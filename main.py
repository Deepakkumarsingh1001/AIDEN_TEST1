"""
Warehouse CoPilot 2.0 — Chatbot Router (main entry)
====================================================

A single chat endpoint backed by a LangGraph ReAct agent that selects tools
freely based on the user's query — replacing the old intent-classifier +
handler dispatch pattern.

The agent maintains per-session conversation history and handles cross-domain
questions (Workforce + Footfall + Promotion) in a single turn.

Two-phase write operations (Promotion Roster, Exception Roster) are managed
via a confirmation_token: the preview tool generates rows + a token, which
main.py stores in the session. On the next turn, if the user confirms, the
agent calls confirm_write_operation with that token.

Run:
    python main.py            # runs on http://127.0.0.1:5000
    POST /chatbot   { "query": "...", "session_id": "<optional, returned on first call>" }
"""

import os
import sys
import uuid
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from agents.chatbot_agent import run_agent
from agents.formatting import text_to_html
from tools.warehouse_logger import get_logger

_log = get_logger("warehouse.main")

# Module-level constant — avoids allocating a new list on every preview response.
_ROSTER_PREVIEW_COLUMNS = [
    "RoasterId", "EmplId", "EmplName", "StoreId", "RosterDate",
    "WeekDay", "RosterMonth", "Mch1", "Experience", "WeeklyOff", "Shift",
]

load_dotenv()

app = Flask(__name__, static_url_path='/static', static_folder='static')
CORS(app, resources={r"/*": {"origins": "*", "send_wildcard": "True"}})
app.config['CORS_HEADERS'] = 'Content-Type'
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'dummy-flask-secret-key')
app.config['SESSION_TYPE'] = 'filesystem'
app.config["SESSION_PERMANENT"] = False

# chat_id -> {
#   "history":       [BaseMessage, ...],   # rolling window of prior turns
#   "last_job_id":   str | None,           # most recent roster generation job
#   "pending_token": str | None,           # confirmation token for two-phase writes
# }
CHAT_SESSIONS: dict = {}

# Keep at most this many messages in the session history sent to the agent
# (10 full turns = 20 messages: 10 Human + 10 AI).
_HISTORY_WINDOW = 20


# ── Helpers ───────────────────────────────────────────────────────────────────

def _html(text: str) -> str:
    """
    Converts a plain text string into an HTML-formatted string.

    Args:
        text: The plain text to convert.

    Returns:
        A string containing HTML paragraph tags.
    """
    return text_to_html(text)


def _build_response_payload(chat_id: str, result: dict, sess: dict) -> dict:
    """Map tool results + agent reply to the structured payload the frontend expects.

    Preserves the same payload shape as the old intent handlers so the
    frontend requires no changes. This function iterates through the tool results
    from the agent, identifies the type of result (e.g., 'footfall', 'roster_view'),
    and builds a JSON payload that the frontend can use to render text, tables,
    charts, and other UI elements.

    Args:
        chat_id: The unique identifier for the current user session.
        result: The dictionary output from the `run_agent` function.
        sess: The current session dictionary, used to manage state like pending tokens.

    Returns:
        A dictionary representing the final JSON payload for the frontend.
    """
    payload: dict = {"session_id": chat_id}

    for tool_msg in result["tool_results"]:
        try:
            tc = (
                json.loads(tool_msg.content)
                if isinstance(tool_msg.content, str)
                else tool_msg.content
            )
        except Exception:
            tc = {}

        rt = tc.get("_result_type", "")

        if rt == "footfall":
            columns = tc.get("columns", [])
            rows = tc.get("rows", [])
            payload.update({
                "reply_html": tc.get("reply_html", ""),
                "summary": tc.get("summary", ""),
                "table": {"columns": columns, "rows": rows},
                "heatmap": {"columns": columns, "rows": rows},
                "type": ["text", "table", "heatmap"],
            })

        elif rt == "promotion_readiness":
            columns = tc.get("columns", [])
            rows = tc.get("rows", [])
            payload.update({
                "reply_html": tc.get("reply_html", ""),
                "summary": tc.get("summary", ""),
                "table": {"columns": columns, "rows": rows},
                "piechart": {"columns": columns, "rows": rows},
                "barchart": {"columns": columns, "rows": rows},
                "type": ["text", "table", "piechart", "barchart"],
            })

        elif rt == "roster_view":
            rows = tc.get("rows", [])
            start_date = tc.get("start_date")
            end_date = tc.get("end_date")
            count = tc.get("count", 0)
            # Use defined column order; fall back to whatever the row has
            columns = [c for c in _ROSTER_PREVIEW_COLUMNS if rows and c in rows[0]] or (
                list(rows[0].keys()) if rows else _ROSTER_PREVIEW_COLUMNS
            )
            payload.update({
                "start_date": start_date,
                "end_date": end_date,
                "count": count,
                "rows": rows,
                "table": {"columns": columns, "rows": rows},
                "type": ["text", "table"],
            })
            # Build HTML table directly from structured rows so ALL fields appear
            # in a proper <table> regardless of how the LLM formatted its reply.
            if rows:
                from agents.formatting import render_html_table
                date_range = (
                    f"{start_date} to {end_date}"
                    if start_date and end_date and start_date != end_date
                    else (start_date or "")
                )
                heading = (
                    f"<p><strong>Roster for {date_range}"
                    f" — {count} record{'s' if count != 1 else ''}</strong></p>"
                )
                payload["_roster_html"] = heading + render_html_table(columns, rows)

        elif rt == "roster_generation":
            job_id = tc.get("job_id")
            if job_id:
                sess["last_job_id"] = job_id
            payload.update({
                "job_id": job_id,
                "start_date": tc.get("start_date"),
                "end_date": tc.get("end_date"),
                "type": ["text"],
            })

        elif rt == "roster_status":
            payload.update({
                "job_id": tc.get("job_id"),
                "job_status": tc.get("status"),
                "type": ["text"],
            })

        elif rt in ("promotion_roster_preview", "exception_roster_preview"):
            token = tc.get("confirmation_token")
            if token:
                sess["pending_token"] = token
            rows = tc.get("rows", [])
            payload.update({
                "rows": rows,
                "table": {"columns": _ROSTER_PREVIEW_COLUMNS, "rows": rows},
                "piechart": {"columns": _ROSTER_PREVIEW_COLUMNS, "rows": rows},
                "dates_processed": (
                    tc.get("dates_processed") or tc.get("roster_date")
                ),
                "type": ["text", "table", "piechart"],
            })

        elif rt == "confirm_write":
            if "error" not in tc:
                sess["pending_token"] = None
            payload.update({
                "posted_ids": tc.get("posted", []),
                "errors": tc.get("errors", []),
                "posted_count": tc.get("posted_count", 0),
                "type": ["text"],
            })

    # Agent's final text reply goes into text.reply.
    # For roster_view, we override with a directly-rendered HTML table so ALL
    # fields appear in a proper <table> regardless of the LLM's markdown output.
    if "_roster_html" in payload:
        reply_html = payload.pop("_roster_html")
    else:
        reply_html = _html(result["reply"])
    payload["text"] = {"reply": reply_html}
    payload.setdefault("type", ["text"])

    return payload


# ── API ───────────────────────────────────────────────────────────────────────

@app.route("/chatbot", methods=["POST"])
def chatbot():
    """
    Main chatbot endpoint that handles all user interactions.

    It receives a user query and a session_id, retrieves the conversation history,
    and invokes the main LangGraph agent (`run_agent`) to process the request.
    It manages session state, including history and tokens for multi-turn operations.

    Returns:
        A JSON response containing the agent's reply and any associated data, or an error.
    """
    data = request.get_json(silent=True) or {}
    query = (data.get("query") or data.get("message") or "").strip()
    if not query:
        return jsonify({"error": "Missing 'query' in request body", "type": ["text"]}), 400

    chat_id = data.get("session_id") or str(uuid.uuid4())
    sess = CHAT_SESSIONS.setdefault(
        chat_id,
        {"history": [], "last_job_id": None, "pending_token": None},
    )

    # Rolling window: only send the last N messages to the agent
    history = sess["history"][-_HISTORY_WINDOW:]

    # If a two-phase write is pending, inject the token as a system hint so
    # the agent can reference it when the user confirms without relying on
    # it appearing verbatim in the conversation history.
    if sess.get("pending_token"):
        history = history + [
            SystemMessage(
                content=(
                    f"There is a pending write operation with "
                    f"confirmation_token: {sess['pending_token']}. "
                    f"If the user confirms, call confirm_write_operation "
                    f"with this exact token."
                )
            )
        ]

    _log.info(f"[{chat_id[:8]}] REQUEST  | query: {query[:120]}")
    req_start = time.time()

    try:
        result = run_agent(query, history, session_id=chat_id)
    except Exception as e:
        _log.error(f"[{chat_id[:8]}] ERROR    | {e}", exc_info=True)
        return jsonify({"session_id": chat_id, "error": str(e), "type": ["text"]}), 500

    # Persist this turn to the session history
    sess["history"].append(HumanMessage(content=query))
    sess["history"].append(AIMessage(content=result["reply"]))

    payload = _build_response_payload(chat_id, result, sess)

    elapsed = time.time() - req_start
    resp_type = payload.get("type", [])
    _log.info(f"[{chat_id[:8]}] RESPONSE | elapsed={elapsed:.2f}s | type={resp_type} | history_len={len(sess['history'])}")

    return jsonify(payload), 200


@app.route("/bookmarks", methods=["GET"])
def bookmarks():
    """
    Provides a list of suggested prompts for the frontend.

    These "bookmarks" are displayed in the UI to help guide users and showcase
    the chatbot's capabilities across different domains like Footfall, Promotion
    Readiness, and Workforce Management.

    Returns:
        A JSON object containing a categorized list of prompt suggestions.
    """
    data = {
        "categories": [
            {
                "name": "Footfall & Sales Conversion",
                "prompts": [
                    {"label": "Department Performance Overview", "query": "Show me the department footfall and sales conversion performance"},
                    {"label": "Backpacks – Why Did Conversion Drop?", "query": "Why did conversion for Backpacks drop despite higher footfall?"},
                    {"label": "Underperforming Departments", "query": "Which departments are underperforming in sales conversion?"},
                    {"label": "Article Sales Conversion Analysis", "query": "Show me article sales conversion analysis for all departments"},
                    {"label": "Improvement Recommendations", "query": "Give me recommendations to improve sales conversion for underperforming departments"},
                ],
            },
            {
                "name": "CALLA TOTE BAG – Promotion Readiness",
                "prompts": [
                    {"label": "CALLA TOTE BAG – Readiness Status", "query": "What is the promotion readiness status for CALLA TOTE BAG?"},
                    {"label": "CALLA TOTE BAG – Labelling Status", "query": "What is the labelling completion and labelling status for CALLA TOTE BAG?"},
                    {"label": "CALLA TOTE BAG – Order Quantity", "query": "How many units of CALLA TOTE BAG need to be ordered for the promotion?"},
                    {"label": "CALLA TOTE BAG – Pricing Status", "query": "What is the pricing status for CALLA TOTE BAG?"},
                    {"label": "CALLA TOTE BAG – Readiness Recommendations", "query": "What are the recommendations to improve promotion readiness for CALLA TOTE BAG?"},
                ],
            },
            {
                "name": "CALLA TOTE BAG – Promotion Plans",
                "prompts": [
                    {"label": "CALLA TOTE BAG – Suggest Promotion Plan", "query": "Suggest a promotion plan to increase footfall for CALLA TOTE BAG"},
                    {"label": "CALLA TOTE BAG – Detailed Promotion Plan", "query": "Suggest a detailed promotion plan to increase the footfall in CALLA TOTE BAG with start date, end date, quantities and discount %"},
                    {"label": "CALLA TOTE BAG – Footfall Analysis", "query": "Show me the footfall and sales conversion analysis for CALLA TOTE BAG"},
                    {"label": "CALLA TOTE BAG – Full Analysis", "query": "Give me a complete analysis of CALLA TOTE BAG including readiness, footfall performance and promotion recommendations"},
                ],
            },
            {
                "name": "Promotion Plans – By Department",
                "prompts": [
                    {"label": "Backpacks – Detailed Promotion Plan", "query": "Suggest a detailed promotion plan to increase the footfall in Backpacks with articles, start date, end date, quantities and discount %"},
                    {"label": "Handbags – Detailed Promotion Plan", "query": "Suggest a detailed promotion plan for Handbags with articles, start date, end date, quantities and discount %"},
                    {"label": "Accessories – Detailed Promotion Plan", "query": "Suggest a detailed promotion plan for Accessories with articles, start date, end date, quantities and discount %"},
                    {"label": "All Departments – Promotion Plan", "query": "Suggest a detailed promotion plan for all departments with articles, start date, end date, quantities and discount %"},
                ],
            },
            {
                "name": "Promotion Readiness – All Articles",
                "prompts": [
                    {"label": "Full Promotion Readiness", "query": "Show me the full promotion readiness for all articles"},
                    {"label": "Articles Not Ready", "query": "Which articles are NOT READY for the upcoming promotion?"},
                    {"label": "Articles At Risk", "query": "Which articles are AT RISK for the upcoming promotion?"},
                    {"label": "Pricing Issues", "query": "Which articles have pricing issues for the promotion?"},
                    {"label": "Replenishment Quantities", "query": "Show me the replenishment order quantities for all promotion articles"},
                ],
            },
            {
                "name": "Workforce & Roster",
                "prompts": [
                    {"label": "Morning Briefing", "query": "What requires my attention today?"},
                    {"label": "Today's Staffing Risks", "query": "Show me today's workforce risks and understaffed departments"},
                    {"label": "Workforce Summary", "query": "Give me a workforce summary by department"},
                ],
            },
        ]
    }
    return jsonify(data), 200


@app.route("/health", methods=["GET"])
def health():
    """
    A simple health check endpoint to confirm the service is running.

    Returns:
        A JSON response indicating the status is "ok".
    """
    return jsonify({"status": "ok", "service": "warehouse-copilot-chatbot", "type": ["text"]}), 200


if __name__ == "__main__":
    cf_port = os.getenv("PORT")
    port = int(cf_port) if cf_port else 5000
    app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)
