"""
Roster View Agent
=================

Read-only agent that fetches and displays the workforce roster for a single
date or a date range, from RosterMaster (via tools.warehouse_tools).

It understands natural-language date asks, e.g.:
  - "display the roster for next 2 days"
  - "show roster for today" / "tomorrow" / "yesterday"
  - "roster for this week"
  - "roster from 15 jan 2026 to 17 jan 2026"
  - "roster for 2026-01-15"
or explicit fields { "start_date": ..., "end_date": ... } / { "date": ... }.

No LLM is used — this is a deterministic data fetch + table render.

Relative phrasing convention:
  "next N days" = today through today+(N-1)  (N days, INCLUDING today)
  "last N days" = today-(N-1) through today

Run:
    python agents/roster_view_agent.py
    POST /roster   { "query": "display the roster for next 2 days" }
"""

import os
import re
import sys
from datetime import datetime, timedelta

from dateutil import parser as dateparser

# Make the project root importable so `from tools...` works when run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify

from tools.warehouse_tools import fetch_roster_range

app = Flask(__name__)

# Columns shown in the rendered table (RosterMaster fields).
ROW_FIELDS = [
    "RoasterId", "RosterDate", "WeekDay", "StoreId", "Mch1", "Shift",
    "EmplId", "EmplName", "Experience", "WeeklyOff", "RosterMonth",
]


# ── Date-range resolution ──────────────────────────────────────────────────────
def _fmt(d) -> str:
    return d.strftime("%Y-%m-%d")


def _parse_one(token: str):
    """Parse a single date token to a date object (ISO / YYYYMMDD / flexible)."""
    token = token.strip()
    m = re.search(r"\b(\d{4}-\d{1,2}-\d{1,2})\b", token)
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y-%m-%d").date()
        except ValueError:
            pass
    m = re.search(r"\b(\d{8})\b", token)
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y%m%d").date()
        except ValueError:
            pass
    try:
        dt = dateparser.parse(token, fuzzy=True, dayfirst=True)
        return dt.date() if dt else None
    except (ValueError, OverflowError):
        return None


def resolve_range(data: dict):
    """Return (start_iso, end_iso) from explicit fields or a natural-language query."""
    # 1. Explicit start/end fields.
    if data.get("start_date") and data.get("end_date"):
        s, e = _parse_one(str(data["start_date"])), _parse_one(str(data["end_date"]))
        if s and e:
            return _fmt(min(s, e)), _fmt(max(s, e))

    text = (data.get("query") or data.get("user_input")
            or data.get("date") or data.get("roster_date") or "").strip()
    if not text:
        return None
    low = text.lower()
    today = datetime.now().date()

    # 2. Relative ranges.
    m = re.search(r"(?:next|coming|following|upcoming)\s+(\d+)\s*day", low)
    if m:
        n = max(1, int(m.group(1)))
        return _fmt(today), _fmt(today + timedelta(days=n - 1))
    m = re.search(r"(?:last|past|previous)\s+(\d+)\s*day", low)
    if m:
        n = max(1, int(m.group(1)))
        return _fmt(today - timedelta(days=n - 1)), _fmt(today)
    if "day after tomorrow" in low:
        d = today + timedelta(days=2)
        return _fmt(d), _fmt(d)
    if "tomorrow" in low:
        d = today + timedelta(days=1)
        return _fmt(d), _fmt(d)
    if "yesterday" in low:
        d = today - timedelta(days=1)
        return _fmt(d), _fmt(d)
    if "today" in low:
        return _fmt(today), _fmt(today)
    if "this week" in low:
        start = today - timedelta(days=today.weekday())  # Monday
        return _fmt(start), _fmt(start + timedelta(days=6))
    if "next week" in low:
        start = today - timedelta(days=today.weekday()) + timedelta(days=7)
        return _fmt(start), _fmt(start + timedelta(days=6))

    # 3. Explicit two-ended range ("X to Y", "between X and Y", "X - Y", ...).
    cleaned = re.sub(r"\b(from|between|for|the|roster|display|show|give\s+me|please)\b",
                     " ", low)
    parts = re.split(r"\s+(?:to|till|until|through|thru|and|–|—|-)\s+", cleaned)
    if len(parts) >= 2:
        s, e = _parse_one(parts[0]), _parse_one(parts[-1])
        if s and e:
            return _fmt(min(s, e)), _fmt(max(s, e))

    # 4. Single date anywhere in the text.
    one = _parse_one(text)
    if one:
        return _fmt(one), _fmt(one)
    return None


# ── Rendering ────────────────────────────────────────────────────────────────
def _render_table(rows: list) -> str:
    if not rows:
        return "No roster records found."
    header = "| " + " | ".join(ROW_FIELDS) + " |"
    sep = "| " + " | ".join("---" for _ in ROW_FIELDS) + " |"
    body = ["| " + " | ".join(str(r.get(f, "")) for f in ROW_FIELDS) + " |" for r in rows]
    return "\n".join([header, sep, *body])


def _sort_key(r: dict):
    return (str(r.get("RosterDate", "")), str(r.get("Shift", "")),
            str(r.get("Mch1", "")), str(r.get("EmplId", "")))


# ── API ────────────────────────────────────────────────────────────────────────
@app.route("/roster", methods=["POST"])
def roster():
    data = request.get_json(silent=True) or {}
    rng = resolve_range(data)
    if not rng:
        return jsonify({
            "error": "Could not determine a date or date range. Provide 'start_date'+'end_date', "
                     "'date', or a 'query' like 'roster for next 2 days'."
        }), 400

    start_date, end_date = rng
    try:
        rows = fetch_roster_range(start_date, end_date)
    except Exception as e:
        print(f"[ROSTER VIEW] ERROR: {e}")
        return jsonify({"error": str(e)}), 500

    rows = sorted(rows, key=_sort_key)
    return jsonify({
        "start_date": start_date,
        "end_date": end_date,
        "count": len(rows),
        "rows": rows,
        "table": _render_table(rows),
    }), 200


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "agent": "roster-view"}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5004, debug=True, use_reloader=False)
