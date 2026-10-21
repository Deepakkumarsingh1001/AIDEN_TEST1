"""
Promotion Readiness Agent  (SAP Joule -> Python port)
=====================================================

Standalone Flask agent that reproduces the "Promotion Readiness Agent" built in
SAP Joule (AIDEN Store Co-pilot).

Data flow (chained OData calls, all via tools/):
  1. GET /PromotionReadinessAgent          -> readiness rows (ArticleId, PromotionId, ...)
  2. GET /ArticleMaster   ?$filter=ArticleId   -> ArticleDescription per ArticleId
  3. GET /PromotionMaster ?$filter=PromotionId -> PromoDescription   per PromotionId
  4. Merge descriptions into the readiness rows
  5. Feed enriched data + the deterministic agent instructions + the user query to gpt-4o.

The agent operates STRICTLY on the retrieved skill data — analysis, recommendation
and quantity logic are all driven by the prompt below (mirroring the Joule agent).

Run:
    python agents/promotion_readiness_agent.py
    POST /promotion-readiness   { "query": "how to improve promotion readiness" }
"""

import os
import sys
import json

# Make the project root importable so `from tools...` / `from utils import Utils`
# work when this file is run directly (python agents/promotion_readiness_agent.py).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, Response, jsonify
from dotenv import load_dotenv

from utils import Utils
from tools.warehouse_tools import (
    fetch_promotion_readiness,
    fetch_article_descriptions,
    fetch_promotion_descriptions,
)
from agents.formatting import render_html_reply

load_dotenv()

# ── LLM ───────────────────────────────────────────────────────────────────────
u = Utils()
llm = u.initialize_llm4()
if not llm:
    raise ValueError("Failed to initialize LLM.")

app = Flask(__name__)


# ── Agent instructions (ported verbatim from the Joule agent) ───────────────────
PROMO_EXPERTISE = """\
You are a Promotion Readiness Recommendation Agent.
You specialize in identifying readiness issues such as pricing gaps, labeling status,
and overall readiness status across records.
You focus only on actionable strategies to improve promotion execution and readiness
at department level.
"""

PROMO_INSTRUCTIONS = """\
You are a Promotion Readiness Joule Agent.

You operate STRICTLY on the Promotion Readiness data provided to you in this prompt.
You MUST NOT assume, infer, derive, or invent any value.
ALL outputs MUST be computed ONLY from that data using the EXPLICIT logic below.

====================
ROLE & GUARANTEES
====================
- Deterministic analytics agent
- No hallucinated data
- NO synthetic row or duplicate rows
- NO grain changes
- If data unavailable -> "Not Available"

====================
AVAILABLE SKILL
====================
Promotion Readiness:
ArticleId, ArticleDescription, Department, PromotionId, PromoDescription,
ReadinessPct, PricingStatus, LabellingStatus, ReadinessStatus,
LabellingCompletionPct, OrderToBePlaced

=================
USER INPUT (ONLY)
=================
- Interpret the user input

====================
STEP 1
====================
Read the Promotion Readiness data and use it as the ONLY data source.

=============================
STEP 2: PROMOTION STRATEGY
=============================
Fields available:
ReadinessPct, PricingStatus, LabellingStatus, ReadinessStatus,
LabellingCompletionPct, OrderToBePlaced

Only generate recommendations if the user query explicitly asks for:
- improvement
- strategy
- suggestions
- recommendation
- how to improve promotion readiness

If the user query is ONLY asking for metrics, DO NOT generate recommendations.

------------------------------------
OVERALL RECOMMENDATION RULE

When recommendations are requested:
- DO NOT generate recommendations per row
- DO NOT generate per ArticleId or PromotionId
- Analyze ALL matched rows collectively

Identify overall issues:
- Low ReadinessPct -> poor promotion readiness
- PricingStatus = "No" -> pricing gaps
- LabellingStatus = Red/Yellow -> labeling issues
- Low LabellingCompletionPct -> incomplete execution
- ReadinessStatus = Not Ready / Borderline -> execution gaps

=========================
QUANTITY SUGGESTION RULE
=========================
If the user asks about:
- quantity
- order quantity
- stock required
- how much order
- replenishment

Then:
- Use ONLY OrderToBePlaced from skill data
- Map OrderToBePlaced to QuantityOrderSuggestion
- Do NOT calculate new values
- Negative values indicate excess inventory
- Positive values indicate additional quantity required
- If unavailable -> "Not Available"

==============
FINAL OUTPUT
==============
OUTPUT FORMAT (CARD STYLE)
- Group data by ArticleId
- Generate ONE overall recommendation per ArticleId

Return as TABLE:
ArticleId | ArticleDescription | ReadinessPct | QuantityOrderSuggestion | Recommendation

Rules:
- Include ArticleId
- Include ArticleDescription
- If ArticleDescription unavailable -> "Not Available"
- Numeric values are allowed ONLY for QuantityOrderSuggestion and ReadinessPct
- One row per ArticleId
- Keep it clean and readable

------------------------------------
RECOMMENDATION STYLE:
- Short, clear, actionable points
- Business-friendly language
- No repetition
- Focus on improving promotion readiness execution

------------------------------------
RECOMMENDATION GUIDELINES
Provide 3-5 actionable promotion readiness improvements such as:
- Ensure pricing is completed before promotion launch
- Improve labeling accuracy and completion across products
- Prioritize fixing items with low readiness status
- Increase labeling completion through better store execution
- Validate promotion setup before activation
- Focus on high-impact products for readiness improvement

Rules:
- Return minimum 3 recommendations
- Keep recommendations concise and actionable
- Do NOT repeat similar suggestions

====================
PROHIBITED
====================
Do NOT:
- invent articles or departments
- derive new values
- summarize data
- change field names
- stop after first row if multiple rows exist
"""

PROMO_OUTPUT_CONTRACT = """\
================
OUTPUT CONTRACT (STRICT — overrides any display wording above)
================
Return ONLY a JSON object (no prose, no markdown, no code fences):
{
  "columns": [ "<ordered column headers>" ],
  "rows":    [ { "<column>": <value>, ... }, ... ],
  "message": "<optional one-line note, or 'Not Available'>"
}

Shape per mode:
- METRICS / ANALYSIS -> columns chosen from the available fields (ArticleId, ArticleDescription,
  Department, ReadinessPct, PricingStatus, LabellingStatus, ReadinessStatus,
  LabellingCompletionPct, etc.); one object per matched row.
- QUANTITY -> include "QuantityOrderSuggestion" sourced ONLY from OrderToBePlaced.
- RECOMMENDATIONS -> columns: ["ArticleId","ArticleDescription","ReadinessPct","QuantityOrderSuggestion","Recommendation"];
  ONE row per ArticleId; put the 3-5 actions in "Recommendation" as a SINGLE string with
  points separated by "; " (NEVER split across rows, NEVER leave blank cells).
- If ArticleDescription is unavailable use "Not Available".
- No match -> {"columns": [], "rows": [], "message": "Not Available"}.
Numeric values allowed ONLY for ReadinessPct / QuantityOrderSuggestion and must be numbers.
"""


def _enrich_with_descriptions(rows: list) -> list:
    """
    Enriches promotion readiness data with article and promotion descriptions.

    This function takes the raw readiness rows, extracts all unique ArticleIds and
    PromotionIds, fetches their corresponding descriptions in two batch lookups,
    and then merges the descriptions back into the original rows.

    Args:
        rows: A list of promotion readiness data dictionaries.

    Returns:
        A new list of enriched data dictionaries.
    """
    article_ids = [r.get("ArticleId") for r in rows]
    promotion_ids = [r.get("PromotionId") for r in rows]

    art_map = fetch_article_descriptions(article_ids)
    promo_map = fetch_promotion_descriptions(promotion_ids)

    enriched = []
    for r in rows:
        row = dict(r)
        row["ArticleDescription"] = art_map.get(r.get("ArticleId")) or "Not Available"
        row["PromoDescription"] = promo_map.get(r.get("PromotionId")) or "Not Available"
        enriched.append(row)
    return enriched


def build_prompt(user_query: str, data: list) -> str:
    """
    Constructs the full prompt for the LLM.

    It combines the agent's expertise, detailed instructions, output contract,
    the enriched data (as a JSON string), and the user's query into a single
    string that will be sent to the language model.

    Returns:
        The complete prompt string.
    """
    return f"""{PROMO_EXPERTISE}

{PROMO_INSTRUCTIONS}

{PROMO_OUTPUT_CONTRACT}

================
PROMOTION READINESS DATA (the ONLY data you may use)
================
{json.dumps(data, indent=2, default=str)}

================
USER INPUT
================
{user_query}

Follow the instructions and OUTPUT CONTRACT exactly. Use only the Promotion Readiness data provided.
"""


def _cap_readiness_pct(rows: list) -> list:
    """
    Caps the 'ReadinessPct' value in the data.

    If any row has a `ReadinessPct` greater than 110, it is reset to 100. This
    is a data cleaning step to handle potential outliers or bad data before
    sending it to the LLM.

    Args:
        rows: A list of promotion readiness data dictionaries.

    Returns:
        A new list of data dictionaries with capped percentages.
    """
    result = []
    for row in rows:
        r = dict(row)
        pct = r.get("ReadinessPct")
        if pct is not None:
            try:
                if float(pct) > 110:
                    r["ReadinessPct"] = 100
            except (ValueError, TypeError):
                pass
        result.append(r)
    return result


def run_promotion_readiness_agent(user_query: str) -> dict:
    """
    Executes the full promotion readiness analysis pipeline.

    This function orchestrates the entire process:
    1. Fetches the base promotion readiness data.
    2. Enriches the data with descriptions.
    3. Cleans the data (e.g., capping percentages).
    4. Builds the final prompt for the LLM.
    5. Invokes the LLM and gets the structured JSON response.
    6. Renders the final result into an HTML payload for the frontend.

    Returns:
        A dictionary containing the HTML reply and structured data.
    """
    rows = fetch_promotion_readiness()
    if not rows:
        return {
            "reply_html": "<p>Not Available</p>",
            "columns": [],
            "rows": []
        }

    enriched = _enrich_with_descriptions(rows)
    enriched = _cap_readiness_pct(enriched)
    prompt = build_prompt(user_query, enriched)
    response = llm.invoke(prompt)
    content = response.content if hasattr(response, "content") else str(response)
    return render_html_reply(content)




# ── API ────────────────────────────────────────────────────────────────────────
@app.route("/promotion-readiness", methods=["POST"])
def promotion_readiness():
    """
    Flask API endpoint to trigger the promotion readiness agent.

    Accepts a POST request with a JSON body containing a "query".
    It calls the main agent runner and returns the result as an HTML response.

    Returns:
        A Flask Response object containing the HTML result, or a JSON error.
    """
    data = request.get_json(silent=True) or {}
    user_query = data.get("query") or data.get("user_input")
    if not user_query:
        return jsonify({"error": "Missing 'query' in request body"}), 400

    try:
        content = run_promotion_readiness_agent(user_query)
        return Response(content, mimetype="text/html")
    except Exception as e:
        print(f"[PROMO] ERROR: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    """
    A simple health check endpoint for the agent service.
    """
    return jsonify({"status": "ok", "agent": "promotion-readiness"}), 200


if __name__ == "__main__":
    # use_reloader=False: the debug reloader watches the whole workspace (incl. venv)
    # and restarts endlessly when .pyc files are touched, interrupting requests.
    app.run(host="0.0.0.0", port=5002, debug=True, use_reloader=False)
