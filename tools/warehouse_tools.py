"""
Entity-specific endpoint calls for the Warehouse CAPM OData service.

These are the reusable "tools" each agent composes. They wrap the generic verbs
in tools.odata_client so an endpoint is written exactly once and shared across
agents (footfall, roster, promotion readiness, ...).
"""

from tools.odata_client import (
    odata_get,
    odata_post,
    odata_upsert,
    odata_batch,
    odata_post_custom,
)

# Max ids per $filter to keep the URL length bounded.
_FILTER_CHUNK = 50


# ── Footfall ────────────────────────────────────────────────────────────────
def fetch_footfall_master(top: int = 1000) -> list:
    """All FootFall Master rows."""
    return odata_get("FootfallMaster", {"$format": "json", "$top": top})


# ── Roster (consolidated out of roster_api_prod.py) ───────────────────────────
def fetch_store_master() -> list:
    return odata_get("StoreMaster")


def fetch_workforce_master() -> list:
    return odata_get("WorkforceMaster")


def post_roster_batch(records: list) -> dict:
    """POST a chunk of roster records to RosterMaster in one $batch request."""
    return odata_batch("RosterMaster", records)


def post_store_job_id(job_id: str) -> dict:
    return odata_post("StoreJobId", {"jobId": job_id})


def _job_id_of(row: dict):
    return row.get("jobId") or row.get("JobId") or row.get("jobid") or row.get("JOBID")


def fetch_latest_job_id() -> str | None:
    """Return the most recent roster job id from the StoreJobId backend table.

    StoreJobId is (cuid, managed), so order by createdAt desc and take the top row.
    Falls back to a plain GET (last non-empty jobId) if ordering isn't supported.
    """
    try:
        rows = odata_get("StoreJobId", {"$orderby": "createdAt desc", "$top": 1})
        if rows:
            return _job_id_of(rows[0])
    except Exception as e:
        print(f"[TOOLS] StoreJobId orderby failed ({e}); falling back to full scan")

    job_id = None
    for r in odata_get("StoreJobId"):
        jid = _job_id_of(r)
        if jid:
            job_id = jid
    return job_id


def delete_existing_roster(from_date: str, to_date: str) -> dict:
    """Call the custom CAP action that clears roster records in a date range."""
    return odata_post_custom("deleteRosterIfExists", {"fromDate": from_date, "toDate": to_date})


# ── Promotion Readiness chain ─────────────────────────────────────────────────
def fetch_promotion_readiness(top: int = 1000) -> list:
    """All Promotion Readiness rows (includes ArticleId & PromotionId, no descriptions)."""
    return odata_get("PromotionReadinessAgent", {"$format": "json", "$top": top})


def _chunks(seq: list, size: int):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def _escape(value) -> str:
    """Escape a single quote for an OData string literal."""
    return str(value).replace("'", "''")


def coerce_decimal(value):
    """Best-effort convert to a number for Edm.Decimal fields (e.g. Experience).

    Returns None when the value isn't numeric, so CAP receives null instead of a
    string like "None"/"5 yrs" which would be rejected with a 400.
    """
    if value is None:
        return None
    try:
        return float(str(value).strip())
    except (ValueError, TypeError):
        return None


def _lookup_descriptions(entity: str, key_field: str, desc_field: str, ids: list) -> dict:
    """Generic id->description lookup. `key_field` is a text field (values quoted)."""
    unique_ids = [i for i in dict.fromkeys(i for i in ids if i not in (None, ""))]
    result: dict = {}
    for chunk in _chunks(unique_ids, _FILTER_CHUNK):
        clause = " or ".join(f"{key_field} eq '{_escape(i)}'" for i in chunk)
        rows = odata_get(entity, {"$format": "json", "$filter": clause})
        for row in rows:
            key = row.get(key_field)
            if key is not None:
                result[key] = row.get(desc_field)
    return result


def fetch_article_descriptions(article_ids: list) -> dict:
    """Map {ArticleId: ArticleDescription} for the given ids (filtered against ArticleMaster)."""
    return _lookup_descriptions("ArticleMaster", "ArticleId", "ArticleDescription", article_ids)


def fetch_promotion_descriptions(promotion_ids: list) -> dict:
    """Map {PromotionId: PromoDescription} for the given ids (filtered against PromotionMaster)."""
    return _lookup_descriptions("PromotionMaster", "PromotionId", "PromoDescription", promotion_ids)


# ── Promotion Roster ReGeneration chain ───────────────────────────────────────
def fetch_promotion_master(top: int = 1000) -> list:
    """All Promotion Master rows (ArticleId, PromotionId, PromoStartDate, PromoEndDate, ...)."""
    return odata_get("PromotionMaster", {"$format": "json", "$top": top})


def fetch_roster_on_date(roster_date: str) -> list:
    """Baseline roster rows for a single date (RosterMaster filtered by RosterDate eq 'date')."""
    clause = f"RosterDate eq '{_escape(roster_date)}'"
    return odata_get("RosterMaster", {"$format": "json", "$filter": clause})


def fetch_roster_range(start_date: str, end_date: str) -> list:
    """Roster rows for an inclusive date range (RosterDate ge 'start' and le 'end').

    RosterDate is stored as an ISO YYYY-MM-DD string, so ge/le compare correctly.
    A single-day range (start == end) works too.
    """
    clause = (f"RosterDate ge '{_escape(start_date)}' "
              f"and RosterDate le '{_escape(end_date)}'")
    return odata_get("RosterMaster", {"$format": "json", "$filter": clause})


def post_promotion_roster(record: dict) -> dict:
    """Upsert one generated Promotion Support row to PromotionRosterMaster."""
    return odata_upsert("PromotionRosterMaster", record, record["RoasterId"])


# ── Workforce Exception Roster chain ──────────────────────────────────────────
def fetch_exception_master(top: int = 1000) -> list:
    """All Workforce Exception rows (EmplId, EmplName, CurrentDept, ExceptionCase,
    ExceptionDate, DayOff, Delete_mc, Update_mc)."""
    return odata_get("ExceptionMaster", {"$format": "json", "$top": top})


def post_exception_roster(record: dict) -> dict:
    """Upsert one exception roster row into ExceptionRosterMaster (POST, fallback PUT)."""
    return odata_upsert("ExceptionRosterMaster", record, record["RoasterId"])
