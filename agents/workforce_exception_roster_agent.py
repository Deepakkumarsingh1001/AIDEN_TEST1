"""
Workforce Roster Exception ReGeneration Agent  (SAP Joule -> Python port)
=========================================================================

Updates an ALREADY GENERATED roster when employee exceptions occur (leave, sick
leave, absence). It marks the impacted baseline rows, preserves everything else,
and creates NEW replacement rows when shift coverage is required. It never builds
a roster from scratch and never deletes rows.

Data sources (all via tools/):
  - Workforce Exception : GET /ExceptionMaster   (exceptions; auto mode source)
  - Read Roster on Date : GET /RosterMaster?$filter=RosterDate eq 'date' (baseline)
  - Workforce Master    : GET /WorkforceMaster    (SOURCE OF TRUTH for replacement)
Posting target:
  - UpdateExceptionRoster: POST /ExceptionRosterMaster (one row at a time)

AUTOMATIC EXCEPTION MODE:
  Input "generate exception roster" (or no explicit date) -> detect TODAY, fetch
  exceptions for TODAY or any FUTURE date, ignore PAST exceptions, process each
  exception date in chronological order.

Two-phase, confirmation-gated flow (POSTING RULE):
  Phase 1 (generate): build the exception-adjusted rows as JSON, render a table,
    stash under a session id, ask for confirmation.
  Phase 2 (post): on confirmation, POST the stashed rows to ExceptionRosterMaster
    ONE BY ONE. Rows are NOT regenerated.

Run:
    python agents/workforce_exception_roster_agent.py

    # Phase 1 - generate (automatic mode)
    POST /exception-roster   { "query": "generate exception roster" }
    # or a specific date
    POST /exception-roster   { "date": "2026-03-02" }

    # Phase 2 - confirm + post
    POST /exception-roster   { "session_id": "<id>", "confirm": "yes" }
"""

import os
import re
import sys
import json
import uuid
from datetime import datetime

from dateutil import parser as dateparser

# Make the project root importable so `from tools...` / `from utils import Utils`
# work when this file is run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify
from dotenv import load_dotenv

from utils import Utils
from tools.warehouse_tools import (
    fetch_workforce_master,
    fetch_exception_master,
    fetch_roster_on_date,
    post_exception_roster,
    coerce_decimal,
)

load_dotenv()

# ── LLM ───────────────────────────────────────────────────────────────────────
u = Utils()
llm = u.initialize_llm4()
if not llm:
    raise ValueError("Failed to initialize LLM.")

app = Flask(__name__)

# In-memory store of generated rows awaiting confirmation (session_id -> {...}).
SESSION_STORE: dict = {}

_CONFIRM_WORDS = {"yes", "ok", "okay", "go ahead", "confirm", "true"}

# Fields on each posted row (matches the ExceptionRosterMaster entity). RoasterId is the key.
ROW_FIELDS = [
    "Mch1", "Shift", "EmplId", "StoreId", "WeekDay", "EmplName",
    "RoasterId", "WeeklyOff", "Experience", "RosterDate", "RosterMonth",
]

# Schema string-length limits for ExceptionRosterMaster — enforced before every POST/PUT
# to prevent CAPM 500 errors caused by values exceeding the column width.
# Source: schema.txt Entity Exception_Roster_Master.
_STRING_LIMITS = {
    "EmplId": 10,
    "StoreId": 4,
    "RosterDate": 10,
    "EmplName": 40,
    "WeekDay": 10,
    "RosterMonth": 10,  # Workforce_Master has String(15) — must trim here
    "Mch1": 40,
    "WeeklyOff": 10,    # Workforce_Master has String(15) — must trim here
    "Shift": 30,
}


# ── Agent instructions (ported verbatim from the Joule agent) ───────────────────
EXC_EXPERTISE = """\
You are a Workforce Roster Exception ReGeneration Agent.
Your responsibility is to update an already generated workforce roster when employee
exceptions occur (e.g., leave, sick leave, absence).
You must modify only the impacted rows, preserve the existing roster, and create
replacement roster rows when shift coverage is required.
"""

EXC_INSTRUCTIONS = """\
You are a Workforce Roster ReGeneration Joule Agent.
You operate ONLY on an ALREADY GENERATED roster.
You NEVER create a roster from scratch.

CORE OBJECTIVES
- Read the existing baseline roster.
- Apply valid workforce exceptions.
- Preserve ALL unaffected roster rows exactly as-is.
- Mark the exception correctly on the impacted row.
- NEVER delete existing rows.
- Ensure ALL required shifts remain covered.
- ALWAYS create NEW roster rows when coverage is required.
- Add new rows to the existing roster.
- Use Workforce Master as the ONLY source for replacement selection.

AUTOMATIC EXCEPTION MODE
If the user input is "generate exception roster" (or no explicit date), then:
- Automatically detect TODAY's date.
- Fetch employees who have exceptions scheduled for TODAY only.
- Ignore past and future exception records.
- Generate the exception-adjusted roster for today only and display the FULL updated rows.
- Ask for confirmation before posting (handled by the system).

VALID DEPARTMENT NAMES
Always use ONLY these exact department names for the Mch1 field:
  SUB-KIDS-BAGS-BACKPACKS
  SUB-WOMEN-ACCESSORIES-KEYCHAINS-BAG-CHARMS
  SUB-WOMEN-BAGS-SHOULDER-BAGS
  SUB-WOMEN-BAGS-TOTE-BAGS
  SUB-WOMEN-SHOES-SLIDE-SANDALS
  SUB-WOMEN-SHOES-SLINGBACK-PUMPS
  SUB-WOMEN-SHOES-STRAPPY-SANDALS
  SUB-WOMEN-WALLETS-SMALL-WALLETS

If any source data contains legacy department names (e.g. Fresh Food, Apparel, Electronics,
General Merchandize, Gas Station, Health & Beauty, Toys and Games, Hard Toys), map each
employee to the closest valid department above using their CurrDept from Workforce Master.
Do NOT write any legacy department name in the output rows or in any text summary.

INPUT SOURCES
Baseline Roster: provided per exception date.
Employee Exceptions: provided (auto-fetched from the Workforce Exception Table).
Workforce Master Table (SOURCE OF TRUTH): the ONLY source allowed for Employee Name,
Department, Experience, Weekly Off, and replacement selection. Generated roster data
MUST NOT be used to determine replacement eligibility.

CRITICAL DATE FORMAT RULE
Roster Date MUST be YYYY-MM-DD (e.g. 2026-03-02). Never YYYYMMDD / DD-MM-YYYY / MM-DD-YYYY.
If a source returns YYYYMMDD, convert it to YYYY-MM-DD before using.

PROCESS LOGIC
STEP 1 - MARK EXCEPTION
- Keep original Roaster_ID unchanged.
- Replace Shift with the exception value (example: SICK_LEAVE).
- DO NOT delete the row. IGNORE past exceptions.

BASELINE IMMUTABILITY RULE
After Step 1 rows are IMMUTABLE. ONLY Shift may change.
NEVER change Empl_ID, Empl_Name, Department, Roaster_ID.
NEVER reuse an existing row for coverage.

STEP 2 - CHECK COVERAGE
If a working shift is removed, replacement is MANDATORY.

STEP 3 - SELECT REPLACEMENT
Replacement MUST: belong to the SAME department; come ONLY from Workforce Master;
NOT already have ANY shift on the SAME date; NOT be working the SAME shift on the SAME date;
NOT be selected using generated roster data as availability source.
Selection Priority:
  1. Same department employee whose Weekly_Off matches the roster Day.
  2. Same department employee working on a different working day.
  3. Same department employee with highest Experience.

STEP 4 - CREATE NEW ROSTER ROW (MANDATORY)
ALWAYS create a NEW row. Roaster_ID format: R-YYYYMMDD-NEW
(multiple: R-YYYYMMDD-NEW-1, R-YYYYMMDD-NEW-2, ...). Roaster_ID must be unique.
NEVER overwrite or reuse existing rows.

STEP 5 - ADD NEW ROW
- Date, Day, Month, Store_ID must be the same as the exception row.
- Empl_ID, Empl_Name, Department, Experience, Weekly_Off must come from Workforce Master.
- Shift must be the uncovered shift.
- Place the NEW row immediately below the exception row.

CHAIN REPLACEMENT RULE (NO DOUBLE SHIFTS)
If a selected replacement already has a shift on the SAME date:
- Mark their original row Shift as REASSIGNED. Keep original Roaster_ID unchanged.
- Treat this as a new exception and repeat STEP 2 to STEP 5.
- Stop only when no employee has more than one shift per date.

WEEKLY OFF OVERRIDE RULE
If assigned on a Weekly Off: still CREATE a new row; mark internally as
WEEKLY_OFF_OVERRIDE; do NOT modify Workforce Master.

FRONTEND DISPLAY FILTER RULE
Shift=REASSIGNED is internal only. Do NOT display, return, or post REASSIGNED rows.
Final output: Exception rows + NEW rows only.
- Output COMPLETE final changed rows only.
- Include ONLY exception rows and NEW rows.
- No narration. No explanation.
- Never drop a shift. Never regenerate the full roster.
- Never modify Workforce Master data. Never delete rows.
"""

# Output contract: force machine-readable rows so the API can render a table AND
# post them one-by-one on confirmation. REASSIGNED rows must be excluded.
OUTPUT_CONTRACT = """\
================
OUTPUT CONTRACT (STRICT)
================
Return ONLY a JSON array (no prose, no markdown fences) of the final changed rows:
the exception rows (Shift replaced by the exception value) PLUS the NEW replacement rows.
EXCLUDE any row whose Shift is REASSIGNED (internal only).
Each object MUST have EXACTLY these keys:
  "Mch1"        : department name
  "Shift"       : the shift value (e.g. "Morning (8AM-6PM)", "SICK_LEAVE", "Evening (12PM-10PM)")
  "EmplId"      : employee id
  "StoreId"     : store id (same as the exception row)
  "WeekDay"     : weekday name for the date
  "EmplName"    : employee name (from Workforce Master)
  "RoasterId"   : original id for exception rows; "R-YYYYMMDD-NEW[-n]" for new rows (unique)
  "WeeklyOff"   : weekly off (from Workforce Master)
  "Experience"  : experience (from Workforce Master)
  "RosterDate"  : roster date in YYYY-MM-DD
  "RosterMonth" : month name or number for the date
Return [] if there are no applicable exceptions.
"""


# ── Helpers ──────────────────────────────────────────────────────────────────
def _normalize_iso(value) -> str | None:
    """Normalize a date value (YYYY-MM-DD / YYYYMMDD / flexible) to YYYY-MM-DD."""
    if not value:
        return None
    s = str(value).strip()
    m = re.search(r"\b(\d{4}-\d{1,2}-\d{1,2})\b", s)
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y-%m-%d").strftime("%Y-%m-%d")
        except ValueError:
            pass
    m = re.search(r"\b(\d{8})\b", s)
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y%m%d").strftime("%Y-%m-%d")
        except ValueError:
            pass
    try:
        dt = dateparser.parse(s, fuzzy=True, dayfirst=True)
        return dt.strftime("%Y-%m-%d") if dt else None
    except (ValueError, OverflowError):
        return None


def _resolve_date(data: dict) -> str | None:
    """Explicit date if the user named one; None triggers AUTOMATIC mode."""
    raw = data.get("date") or data.get("roster_date")
    if raw:
        return _normalize_iso(raw)
    # A date embedded in the query (e.g. "exception roster for 2026-03-02").
    query = str(data.get("query") or data.get("user_input") or "")
    if re.search(r"\d{4}-\d{1,2}-\d{1,2}|\d{8}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", query):
        return _normalize_iso(query)
    return None


def _select_exceptions(data: dict):
    """Return (exception_rows, dates_to_process) per explicit-date or AUTOMATIC mode."""
    all_exceptions = fetch_exception_master()
    explicit_date = _resolve_date(data)

    if explicit_date:
        rows = [e for e in all_exceptions if _normalize_iso(e.get("ExceptionDate")) == explicit_date]
        return rows, [explicit_date]

    # AUTOMATIC mode: today ONLY, ignore past and future.
    today = datetime.now().strftime("%Y-%m-%d")
    rows = []
    for e in all_exceptions:
        d = _normalize_iso(e.get("ExceptionDate"))
        if d and d == today:
            e = dict(e)
            e["ExceptionDate"] = d  # normalized for the LLM
            rows.append(e)
    dates = [today] if rows else []
    return rows, dates


def _parse_rows(content: str) -> list:
    content = content.strip()
    if content.startswith("```json"):
        content = content[7:]
    elif content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    content = content.strip()
    try:
        rows = json.loads(content)
        return rows if isinstance(rows, list) else []
    except Exception as e:
        print(f"[EXC] JSON parse failed: {e}")
        return []


def _render_table(rows: list) -> str:
    if not rows:
        return "No exception rows generated."
    header = "| " + " | ".join(ROW_FIELDS) + " |"
    sep = "| " + " | ".join("---" for _ in ROW_FIELDS) + " |"
    body = ["| " + " | ".join(str(r.get(f, "")) for f in ROW_FIELDS) + " |" for r in rows]
    return "\n".join([header, sep, *body])


def generate_rows(data: dict):
    """Phase 1 logic: fetch sources and ask the LLM to generate the changed rows."""
    workforce = fetch_workforce_master()
    exceptions, dates = _select_exceptions(data)
    if not exceptions:
        return [], dates

    # Baseline roster for every date we are processing.
    baseline = []
    for d in dates:
        baseline.extend(fetch_roster_on_date(d))

    today = datetime.now().strftime("%Y-%m-%d")
    prompt = f"""{EXC_EXPERTISE}

{EXC_INSTRUCTIONS}

{OUTPUT_CONTRACT}

================
TODAY
================
{today}

================
EXCEPTION DATES TO PROCESS (chronological)
================
{json.dumps(dates, default=str)}

================
WORKFORCE MASTER (source of truth)
================
{json.dumps(workforce, indent=2, default=str)}

================
EMPLOYEE EXCEPTIONS (today or future only)
================
{json.dumps(exceptions, indent=2, default=str)}

================
BASELINE ROSTER (Read Roster on Date)
================
{json.dumps(baseline, indent=2, default=str)}

Follow the instructions and OUTPUT CONTRACT exactly. Return ONLY the JSON array.
"""
    response = llm.invoke(prompt)
    content = response.content if hasattr(response, "content") else str(response)
    return _parse_rows(content), dates


def post_rows(rows: list) -> dict:
    """Phase 2 logic: POST each row to ExceptionRosterMaster one by one."""
    posted, errors = [], []
    for row in rows:
        # Defensive: never post internal REASSIGNED rows.
        if str(row.get("Shift", "")).strip().upper() == "REASSIGNED":
            continue
        payload = {f: row.get(f) for f in ROW_FIELDS}
        payload["Experience"] = coerce_decimal(payload.get("Experience"))
        # Enforce schema string-length limits — prevents CAPM 500 when source
        # fields (e.g. Workforce_Master.WeeklyOff String(15)) exceed the target
        # column width in ExceptionRosterMaster.
        for field, limit in _STRING_LIMITS.items():
            val = payload.get(field)
            if isinstance(val, str) and len(val) > limit:
                payload[field] = val[:limit]
        try:
            post_exception_roster(payload)
            posted.append(row.get("RoasterId"))
        except Exception as e:
            print(f"[EXC] POST failed for {row.get('RoasterId')}: {e}")
            errors.append({"RoasterId": row.get("RoasterId"), "error": str(e)})
    return {"posted": posted, "errors": errors}


# ── API ────────────────────────────────────────────────────────────────────────
@app.route("/exception-roster", methods=["POST"])
def exception_roster():
    data = request.get_json(silent=True) or {}
    confirm_raw = str(data.get("confirm", "")).strip().lower()
    session_id = data.get("session_id")

    # ── Explicit cancel ───────────────────────────────────────────────────────────
    if confirm_raw in ("no", "false"):
        return jsonify({"status": "cancelled", "message": "Posting cancelled. No rows were posted."}), 200

    # ── Phase 2: confirmation -> post previously generated rows (no regeneration) ──
    if confirm_raw in _CONFIRM_WORDS:
        session = SESSION_STORE.get(session_id)
        if not session:
            return jsonify({"error": "Unknown or expired session_id. Generate the roster again."}), 404
        result = post_rows(session["rows"])
        session["posted"] = True
        return jsonify({
            "status": "posted",
            "session_id": session_id,
            "posted_count": len(result["posted"]),
            "posted_ids": result["posted"],
            "errors": result["errors"],
        }), 200

    # ── Phase 1: generate rows and ask for confirmation ───────────────────────────
    try:
        rows, dates = generate_rows(data)
    except Exception as e:
        print(f"[EXC] ERROR: {e}")
        return jsonify({"error": str(e)}), 500

    new_session = str(uuid.uuid4())
    SESSION_STORE[new_session] = {"rows": rows, "dates": dates, "posted": False}

    if not rows:
        return jsonify({
            "status": "no_exceptions",
            "session_id": new_session,
            "dates_processed": dates,
            "rows": [],
            "table": "No applicable exceptions (today or future) found.",
            "message": "No exception roster changes to post.",
        }), 200

    return jsonify({
        "status": "generated",
        "session_id": new_session,
        "dates_processed": dates,
        "rows": rows,
        "table": _render_table(rows),
        "message": (
            "Do you confirm posting this updated exception roster? "
            "Reply with {\"session_id\": \"%s\", \"confirm\": \"yes\"} to post, "
            "or \"confirm\": \"no\" to cancel." % new_session
        ),
    }), 200


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "agent": "workforce-exception-roster"}), 200


if __name__ == "__main__":
    # use_reloader=False: the debug reloader watches the whole workspace (incl. venv)
    # and restarts endlessly when .pyc files are touched, interrupting requests.
    app.run(host="0.0.0.0", port=5005, debug=True, use_reloader=False)
