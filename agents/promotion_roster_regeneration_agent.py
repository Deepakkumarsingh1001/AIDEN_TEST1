"""
Promotion Roster ReGeneration Agent  (SAP Joule -> Python port)
===============================================================

When a department goes on promotion for a given date, this agent updates the
already-generated workforce roster by introducing a third shift -
Promotion Support (11AM-9PM) - for every department that has an active promotion.

Data sources (all via tools/):
  - Workforce Master  : GET /WorkforceMaster          (source of truth for staff)
  - Promotion Master  : GET /PromotionMaster           (active promotions -> departments)
  - Read Roster on Date: GET /RosterMaster?$filter=RosterDate eq 'date'  (baseline shifts)
Posting target:
  - UpdatePromotionRoster: POST /PromotionRosterMaster (one row at a time)

Two-phase, confirmation-gated flow (POSTING RULE in the Joule prompt):
  Phase 1 (generate): fetch data, ask gpt-4o to generate the NEW Promotion Support
    rows as JSON, render a table, stash the rows under a session id, and ask the
    user to confirm posting.
  Phase 2 (post): on confirmation the stashed rows are POSTed to PromotionRosterMaster
    ONE BY ONE. Rows are NOT regenerated.

Run:
    python agents/promotion_roster_regeneration_agent.py

    # Phase 1 - generate
    POST /promotion-roster-regeneration   { "date": "2026-03-02" }
        -> { session_id, rows, table, message: "Do you confirm posting..." }

    # Phase 2 - confirm + post
    POST /promotion-roster-regeneration   { "session_id": "<id>", "confirm": "yes" }
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
    fetch_promotion_master,
    fetch_roster_on_date,
    post_promotion_roster,
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
# Mirrors the JOB_STATUS_DB pattern in roster_api_prod.py.
SESSION_STORE: dict = {}

_CONFIRM_WORDS = {"yes", "ok", "okay", "go ahead", "confirm", "true"}

# Fields expected on each generated / posted Promotion Roster row (matches
# the PromotionRosterMaster entity). RoasterId is the key.
ROW_FIELDS = [
    "Mch1", "Shift", "EmplId", "StoreId", "WeekDay", "EmplName",
    "RoasterId", "WeeklyOff", "Experience", "RosterDate", "RosterMonth",
]


# ── Agent instructions (ported verbatim from the Joule agent) ───────────────────
REGEN_EXPERTISE = """\
You are a Promotion Roster ReGeneration Agent.
Your responsibility is to update an already generated workforce roster when a
department goes on promotion for a particular date.
You must create promotion shift roster rows when shift coverage is required.
"""

REGEN_INSTRUCTIONS = """\
You are a Promotion Roster ReGeneration Joule Agent.
You generate an optimized roster using Workforce Master, Promotion Master, and the Baseline Roster.

--------------------------------------------------
CORE OBJECTIVES
--------------------------------------------------
- Identify active promotions for the given date.
- Introduce a third shift: Promotion Support (11AM-9PM).
- Ensure every department has at least one employee assigned.
- Prioritize Promotion Support in departments with promotions.
- Max work per employee per day: 10 hours.
- Respect Weekly Off from Workforce Master.

--------------------------------------------------
VALID DEPARTMENT NAMES
--------------------------------------------------
Always use ONLY these exact department names for the Mch1 field:
  Handbags
  Backpacks
  Kids Line
  Accessories
  Women-Shoes
  Wallets & Small Leather Goods
  New Arrival
  Featured Collection

If any source data contains legacy department names (e.g. Fresh Food, Apparel, Electronics,
General Merchandize, Gas Station, Health & Beauty, Toys and Games, Hard Toys), map each
employee to the closest valid department above using their CurrDept from Workforce Master.
Do NOT write any legacy department name in the output rows or in any text summary.

--------------------------------------------------
INPUT SOURCES
--------------------------------------------------
1. Workforce Master (SOURCE OF TRUTH)
   Fields: EmplId, EmplName, CurrDept, ExperienceYear, WeeklyOff

2. Promotion Master
   Promotion records with Article ID in the form PREFIX-NUMBER (e.g. FF-101, AP-103, EL-210).

   Prefix -> Department Mapping:
     HB -> Handbags
     BP -> Backpacks
     KL -> Kids Line
     AS -> Accessories
     WS -> Women-Shoes
     WL -> Wallets & Small Leather Goods
     NA -> New Arrival
     FC -> Featured Collection

   PROMOTION DEPARTMENT EXTRACTION:
   Promotion Master can contain multiple rows per department.
     1. Extract prefix from Article ID.
     2. Map prefix to department.
     3. Remove duplicates.
   Result = UNIQUE promotion departments.
   Create a Promotion Support shift ONCE per department (NOT per article).

3. Baseline Roster (Read Roster on Date)
   Provides baseline Morning and Evening shift rows.

--------------------------------------------------
DATE FORMAT RULE
--------------------------------------------------
Roster Date must be YYYY-MM-DD (e.g. 2026-03-02).
Never use YYYYMMDD, DD-MM-YYYY, MM-DD-YYYY.
If returned as YYYYMMDD, convert to YYYY-MM-DD before using.

--------------------------------------------------
PROCESS LOGIC
--------------------------------------------------
STEP 1 - AVAILABILITY
Compare roster date with Weekly Off. If matched, employee is UNAVAILABLE.

STEP 2 - DEPARTMENT STABILITY
Every department in Workforce Master must have at least one employee in Morning or Evening shift.

STEP 3 - PROMOTION DEPARTMENTS
From Promotion Master: extract Article ID -> prefix -> department, remove duplicates.
Store list as PROMOTION_DEPARTMENTS.

STEP 4 - PROMOTION SUPPORT ASSIGNMENT
For EACH department in PROMOTION_DEPARTMENTS create exactly ONE Promotion Support (11AM-9PM).
Number of Promotion Support rows MUST equal the number of PROMOTION_DEPARTMENTS.
Selection Priority:
  1. Internal employee from same department: not already scheduled, not on Weekly Off.
  2. If unavailable, pull an employee from another department.
  3. When pulling externally choose highest Experience.
  4. Departments with more promotion articles should receive more experienced staff.

STEP 5 - SHIFT RULES
Each employee can work only ONE shift per day.
Allowed shifts: Morning (8AM-6PM), Evening (12PM-10PM), Promotion Support (11AM-9PM).
Max shift length: 10 hours.

STEP 6 - NEW ROW CREATION
Always create new rows for Promotion Support.
Roster_ID format: R-YYYYMMDD-NEW  (multiple: R-YYYYMMDD-NEW-1, R-YYYYMMDD-NEW-2, ...).
Rules: IDs must be unique; never overwrite rows; never reuse IDs; include the Shift with timings.

--------------------------------------------------
FINAL OUTPUT RULES
--------------------------------------------------
- Output COMPLETE final generated rows only.
- Include ONLY promotion rows and NEW rows.
- No narration and no explanation.
- Never drop a shift.
- Never regenerate the full roster.
- Ensure ONE Promotion Support shift row exists for EVERY promotion department.
- Never delete rows.
"""

# Output contract: force machine-readable rows so the API can render a table AND
# post them one-by-one on confirmation. The "Shift" value carries the timing per
# STEP 6 ("include the Shift along with timings").
OUTPUT_CONTRACT = """\
================
OUTPUT CONTRACT (STRICT)
================
Return ONLY a JSON array (no prose, no markdown fences) of the NEW Promotion Support rows.
Each object MUST have EXACTLY these keys:
  "Mch1"        : department name (e.g. "Kids Line")
  "Shift"       : "Promotion Support (11AM-9PM)"
  "EmplId"      : assigned employee id
  "StoreId"     : store id (from baseline roster / workforce)
  "WeekDay"     : weekday name for the date
  "EmplName"    : assigned employee name
  "RoasterId"   : "R-YYYYMMDD-NEW" pattern, unique per row
  "WeeklyOff"   : assigned employee's weekly off
  "Experience"  : assigned employee's experience
  "RosterDate"  : roster date in YYYY-MM-DD
  "RosterMonth" : month name or number for the date
Return [] if there are no active promotion departments for the date.
"""


# ── Helpers ──────────────────────────────────────────────────────────────────
def _resolve_date(data: dict) -> str | None:
    """Normalize a user-supplied date to YYYY-MM-DD.

    Accepts an explicit `date`/`roster_date`, or extracts a date from `query`/
    `user_input`. Handles natural language ("15 january 2026"), YYYYMMDD,
    DD-MM-YYYY, DD/MM/YYYY, etc. dayfirst=True honours the DD-MM-YYYY preference
    for ambiguous numeric dates (the Joule prompt forbids MM-DD-YYYY).
    """
    raw = (data.get("date") or data.get("roster_date")
           or data.get("query") or data.get("user_input"))
    if not raw:
        return None
    raw = str(raw).strip()

    # Already ISO (YYYY-MM-DD) -> normalize as-is. Checked first so dayfirst below
    # never flips an unambiguous ISO date (e.g. 2026-03-02 -> wrong 2026-02-03).
    m = re.search(r"\b(\d{4}-\d{1,2}-\d{1,2})\b", raw)
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y-%m-%d").strftime("%Y-%m-%d")
        except ValueError:
            pass

    # Compact YYYYMMDD anywhere in the text -> YYYY-MM-DD.
    m = re.search(r"\b(\d{8})\b", raw)
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y%m%d").strftime("%Y-%m-%d")
        except ValueError:
            pass

    # Flexible parse; fuzzy=True ignores surrounding words like "generate ... for".
    try:
        dt = dateparser.parse(raw, fuzzy=True, dayfirst=True)
        return dt.strftime("%Y-%m-%d") if dt else None
    except (ValueError, OverflowError):
        return None


def _active_promotions(promotions: list, roster_date: str) -> list:
    """Keep promotions active on roster_date (PromoStartDate <= date <= PromoEndDate).

    ISO YYYY-MM-DD strings compare correctly lexicographically. Rows missing the
    start/end dates are kept (cannot prove they are inactive).
    """
    active = []
    for p in promotions:
        start = (p.get("PromoStartDate") or "").strip()
        end = (p.get("PromoEndDate") or "").strip()
        if start and roster_date < start:
            continue
        if end and roster_date > end:
            continue
        active.append(p)
    return active


def _parse_rows(content: str) -> list:
    """Parse the LLM JSON array, tolerating markdown fences."""
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
        print(f"[REGEN] JSON parse failed: {e}")
        return []


def _render_table(rows: list) -> str:
    """Render generated rows as a simple markdown table for display."""
    if not rows:
        return "No promotion rows generated."
    header = "| " + " | ".join(ROW_FIELDS) + " |"
    sep = "| " + " | ".join("---" for _ in ROW_FIELDS) + " |"
    body = [
        "| " + " | ".join(str(r.get(f, "")) for f in ROW_FIELDS) + " |"
        for r in rows
    ]
    return "\n".join([header, sep, *body])


def generate_rows(roster_date: str) -> list:
    """Phase 1 logic: fetch sources and ask the LLM to generate the new rows."""
    workforce = fetch_workforce_master()
    promotions = _active_promotions(fetch_promotion_master(), roster_date)
    baseline = fetch_roster_on_date(roster_date)

    prompt = f"""{REGEN_EXPERTISE}

{REGEN_INSTRUCTIONS}

{OUTPUT_CONTRACT}

================
ROSTER DATE
================
{roster_date}

================
WORKFORCE MASTER (source of truth)
================
{json.dumps(workforce, indent=2, default=str)}

================
PROMOTION MASTER (active promotions for this date)
================
{json.dumps(promotions, indent=2, default=str)}

================
BASELINE ROSTER (Read Roster on Date)
================
{json.dumps(baseline, indent=2, default=str)}

Follow the instructions and OUTPUT CONTRACT exactly. Return ONLY the JSON array.
"""
    response = llm.invoke(prompt)
    content = response.content if hasattr(response, "content") else str(response)
    return _parse_rows(content)


def post_rows(rows: list) -> dict:
    """Phase 2 logic: POST each row to PromotionRosterMaster one by one."""
    posted, errors = [], []
    for row in rows:
        # Send only the known entity fields.
        payload = {f: row.get(f) for f in ROW_FIELDS}
        payload["Experience"] = coerce_decimal(payload.get("Experience"))  # Decimal field
        try:
            post_promotion_roster(payload)
            posted.append(row.get("RoasterId"))
        except Exception as e:
            print(f"[REGEN] POST failed for {row.get('RoasterId')}: {e}")
            errors.append({"RoasterId": row.get("RoasterId"), "error": str(e)})
    return {"posted": posted, "errors": errors}


# ── API ────────────────────────────────────────────────────────────────────────
@app.route("/promotion-roster-regeneration", methods=["POST"])
def promotion_roster_regeneration():
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
    roster_date = _resolve_date(data)
    if not roster_date:
        return jsonify({
            "error": "Could not determine a roster date. Provide 'date' (any common "
                     "format) or include a date in 'query'."
        }), 400

    try:
        rows = generate_rows(roster_date)
    except Exception as e:
        print(f"[REGEN] ERROR: {e}")
        return jsonify({"error": str(e)}), 500

    new_session = str(uuid.uuid4())
    SESSION_STORE[new_session] = {"rows": rows, "date": roster_date, "posted": False}

    return jsonify({
        "status": "generated",
        "session_id": new_session,
        "roster_date": roster_date,
        "rows": rows,
        "table": _render_table(rows),
        "message": (
            "Do you confirm posting this updated Promotion Roster? "
            "Reply with {\"session_id\": \"%s\", \"confirm\": \"yes\"} to post, "
            "or \"confirm\": \"no\" to cancel." % new_session
        ),
    }), 200


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "agent": "promotion-roster-regeneration"}), 200


if __name__ == "__main__":
    # use_reloader=False: the debug reloader watches the whole workspace (incl. venv)
    # and restarts endlessly when .pyc files are touched, interrupting requests.
    app.run(host="0.0.0.0", port=5003, debug=True, use_reloader=False)
