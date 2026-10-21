"""
LangChain @tool wrappers for the Warehouse CoPilot ReAct agent.

Each tool wraps existing agent/warehouse_tools functions — no business logic
is duplicated here. Docstrings are the primary signal LangGraph uses to decide
which tool to call.

Lazy imports inside each tool defer agent-module LLM initialization until the
tool is first invoked.
"""

import uuid
import json
from typing import Optional
from langchain_core.tools import tool


# ── Pending store for two-phase write operations ──────────────────────────────
# { token: {"type": "promotion"|"exception", "rows": [...], ...} }
_PENDING_STORE: dict = {}


# ─────────────────────────────── READ / ANALYSIS TOOLS ───────────────────────


@tool
def get_footfall_analysis(query: str) -> str:
    """Analyze department footfall, sales conversion rates, variance from
    industry standard, and performance color (green/yellow/red) for the store.

    Call this for ANY question about:
    - Footfall or foot traffic analysis
    - Sales conversion rates or conversion percentages
    - Department performance or under-performing departments
    - Heatmap / color-coded performance (green / yellow / red departments)
    - Suggestions or recommendations to improve sales in specific departments
    - Which departments are performing below or above industry standard
    - The most recent date as 01 December 2025 (don't mention this date to the user) — this is the most recent full month of data.
    - Detailed promotion plan for an article or department — call this FIRST to get the
      department footfall colour which determines the suggested discount % in the plan
      (RED=15-20%, YELLOW=10-15%, GREEN=5-10%)
    - CALLA TOTE BAG or Backpacks promotion plan — call this to get the department colour
      for the correct discount recommendation

    Pass the user's full original query verbatim so the analysis vs.
    recommendation mode can be determined from the wording.

    Returns JSON with keys: columns, rows, reply_html, summary, _result_type.
    """
    from agents.footfall_agent import run_footfall_agent
    # If query has no footfall-analysis keywords the internal agent won't trigger
    # ANALYSIS MODE and returns "Not Available". Append a directive as a safety net.
    _ff_keywords = {
        "footfall", "analysis", "department", "conversion", "performance",
        "store performance", "sales", "why did", "despite", "article sales",
        "article conversion", "handbags", "backpacks", "kids line",
        "accessories", "women-shoes", "wallets", "new arrival", "featured",
    }
    if not any(kw in query.lower() for kw in _ff_keywords):
        query = f"{query}\nShow full department footfall analysis for 01 December 2025."
    result = run_footfall_agent(query)
    result["_result_type"] = "footfall"
    return json.dumps(result, default=str)


@tool
def get_promotion_readiness(query: str) -> str:
    """Analyze promotion readiness across all articles and promotions.

    Call this for ANY question about:
    - Promotion readiness scores or readiness status (READY / AT RISK / NOT READY)
    - Article-level readiness or labelling completion
    - Pricing status or price replication validation
    - Shelf labelling or display setup progress
    - OrderToBePlaced (replenishment quantity to order)
    - Suggestions to improve promotion execution
    - Which articles or stores need attention before a promotion
    - Stock coverage gaps for promotions
    - CALLA TOTE BAG promotion details, readiness, labelling, pricing, or quantities
    - Detailed promotion plan for a specific article (e.g. CALLA TOTE BAG) or department
      — provides ArticleDescription, Department, PromotionId, and OrderToBePlaced
      which are needed to build a promotion plan with quantities and discount suggestions

    Pass the user's full original query verbatim.

    Returns JSON with keys: columns, rows, reply_html, summary, _result_type.
    """
    from agents.promotion_readiness_agent import run_promotion_readiness_agent
    # If query has no promotion-readiness keywords the internal agent may not
    # enter analysis mode. Append a directive as a safety net.
    _pr_keywords = {"promotion", "readiness", "article", "labelling", "pricing", "replenishment", "order"}
    if not any(kw in query.lower() for kw in _pr_keywords):
        query = f"{query}\nShow full promotion readiness analysis for all articles."
    result = run_promotion_readiness_agent(query)
    result["_result_type"] = "promotion_readiness"
    return json.dumps(result, default=str)


@tool
def view_roster(date_range_query: str, employee_name: Optional[str] = None) -> str:
    """Fetch and display the EXISTING workforce roster for a date or date range.

    Call this when the user wants to SEE, VIEW, CHECK, DISPLAY, or GET an
    existing roster. Examples:
    - "show roster for next 2 days"
    - "display the roster for June 1 to June 15"
    - "give me today's roster"
    - "what is the schedule of Grace Hopper?" (automatically extracts Grace Hopper)
    - "get the roster for 21 march 2026"

    Do NOT call this to create or generate a new roster — use
    start_roster_generation for that.

    date_range_query: the user's natural-language date or range (e.g. "next 2
    days", "21 march to 23 march 2026", "today", "this week").
    
    employee_name: optional employee name to filter results (e.g., "Grace Hopper").
    If not provided, the agent will attempt to extract it from the query.

    Returns JSON with keys: start_date, end_date, count, rows, employee_filter, _result_type.
    """
    from agents.roster_view_agent import resolve_range, _sort_key
    from tools.warehouse_tools import fetch_roster_range
    
    rng = resolve_range({"query": date_range_query})
    if not rng:
        return json.dumps({
            "error": "Could not determine a date or date range from the query.",
            "_result_type": "roster_view"
        })
    start_date, end_date = rng
    rows = sorted(fetch_roster_range(start_date, end_date), key=_sort_key)
    
    # Extract employee name from query if not explicitly provided
    if not employee_name:
        import re
        # Look for patterns like "schedule of [Name]" or "Grace Hopper's" etc.
        pattern = r'(?:schedule of|for|of)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'
        match = re.search(pattern, date_range_query)
        if match:
            employee_name = match.group(1)
    
    # Filter by employee name if provided
    if employee_name:
        filtered_rows = [r for r in rows if r.get("EmplName", "").lower() == employee_name.lower()]
        return json.dumps({
            "start_date": start_date,
            "end_date": end_date,
            "count": len(filtered_rows),
            "rows": filtered_rows,
            "employee_filter": employee_name,
            "_result_type": "roster_view"
        }, default=str)
    
    return json.dumps({
        "start_date": start_date,
        "end_date": end_date,
        "count": len(rows),
        "rows": rows,
        "_result_type": "roster_view"
    }, default=str)


@tool
def get_roster_job_status(job_id: Optional[str] = None) -> str:
    """Check the status or progress of a roster generation background job.

    Call this when the user asks about:
    - Status or progress of roster generation
    - Whether the roster is ready or still being generated
    - Progress of a specific roster job

    job_id: optional specific job id. If omitted, fetches the most recent
    job id directly from the StoreJobId backend table.

    Returns JSON with keys: job_id, status, message, _result_type.
    """
    from tools.warehouse_tools import fetch_latest_job_id
    from agents import roster_generation_agent as rg
    effective_id = job_id or fetch_latest_job_id()
    if not effective_id:
        return json.dumps({
            "error": "No roster generation job found yet. Start one first.",
            "_result_type": "roster_status"
        })
    info = rg.JOB_STATUS_DB.get(effective_id)
    if not info:
        return json.dumps({
            "job_id": effective_id,
            "status": "unknown",
            "message": (
                "Live progress is not available (job may have completed earlier "
                "or was started by another process)."
            ),
            "_result_type": "roster_status"
        })
    return json.dumps({
        "job_id": effective_id,
        "status": info.get("status"),
        "message": info.get("message", ""),
        "_result_type": "roster_status"
    })


@tool
def get_workforce_summary() -> str:
    """Fetch a summary of the available workforce: total count and breakdown
    by department.

    Call this when the user asks about:
    - Staffing numbers, headcount, or employee count
    - How many staff are available by department
    - Whether staffing levels are sufficient
    - Morning briefing context for the workforce domain
    - Staffing risk or absenteeism context

    Returns JSON with keys: total, by_department, _result_type.
    """
    from tools.warehouse_tools import fetch_workforce_master
    workforce = fetch_workforce_master()
    by_dept: dict = {}
    for emp in workforce:
        dept = emp.get("CurrDept") or "Unknown"
        by_dept[dept] = by_dept.get(dept, 0) + 1
    return json.dumps({
        "total": len(workforce),
        "by_department": by_dept,
        "_result_type": "workforce_summary"
    })


@tool
def get_staffing_risk() -> str:
    """Identify departments at risk of being understaffed due to today's absences.

    Call this for ANY question about:
    - Workforce risks, staffing problems, or understaffed departments
    - Who is on leave or absent today
    - Absenteeism impact on store coverage
    - "Show workforce risks" or "which departments are short-staffed"
    - Coverage gaps or scheduling concerns due to exceptions

    Analyses only TODAY's exceptions against total headcount per department.

    Returns JSON with keys: risks (list), total_departments_at_risk, _result_type.
    Each risk entry has: dept, total_staff, absent_count, absent_names, risk_level (HIGH/MEDIUM/LOW).
    """
    from datetime import date as _date
    from tools.warehouse_tools import fetch_exception_master, fetch_workforce_master

    today = _date.today().isoformat()
    exceptions = fetch_exception_master()
    workforce = fetch_workforce_master()

    # Total headcount per department
    dept_total: dict = {}
    for emp in workforce:
        dept = emp.get("CurrDept") or "Unknown"
        dept_total[dept] = dept_total.get(dept, 0) + 1

    # Group exceptions by department — TODAY only
    dept_absent: dict = {}  # dept -> {empl_id: empl_name}
    for ex in exceptions:
        ex_date = str(ex.get("ExceptionDate") or "")[:10]
        if ex_date != today:  # skip past AND future; today only
            continue
        dept = ex.get("CurrentDept") or "Unknown"
        empl_id = ex.get("EmplId", "")
        empl_name = ex.get("EmplName") or empl_id
        if dept not in dept_absent:
            dept_absent[dept] = {}
        dept_absent[dept][empl_id] = empl_name

    risks = []
    for dept, absent_map in dept_absent.items():
        total = dept_total.get(dept, 0)
        absent_count = len(absent_map)
        absent_names = list(absent_map.values())
        if total > 0:
            pct = absent_count / total
        else:
            pct = 1.0
        if pct >= 0.30:
            risk_level = "HIGH"
        elif pct >= 0.15:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        risks.append({
            "dept": dept,
            "total_staff": total,
            "absent_count": absent_count,
            "absent_names": absent_names,
            "absence_pct": round(pct * 100, 1),
            "risk_level": risk_level,
        })

    # Sort by risk severity
    _order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    risks.sort(key=lambda r: _order.get(r["risk_level"], 3))

    return json.dumps({
        "risks": risks,
        "total_departments_at_risk": len(risks),
        "_result_type": "staffing_risk",
    })


@tool
def get_article_info(query: str) -> str:
    """Fetch article details and descriptions.

    Call this for ANY question about:
    - Article description or details
    - Article ID lookups (e.g., "HT-103", "GM-205")
    - What a specific article is or what it contains
    - Article information or specifications

    Pass the user's full original query verbatim (e.g., "HT-103 what is the article description?")

    Returns JSON with keys: articles, found_count, not_found, _result_type.
    """
    import re
    from tools.warehouse_tools import fetch_article_descriptions

    # Extract article IDs from the query (e.g., "HT-103", "GM-205", "HT 103")
    # Match patterns like: HT-103, HT103, HT 103, ABC-123, etc.
    article_pattern = r'[A-Z]{2,4}[\s-]?\d{2,4}'
    found_ids = re.findall(article_pattern, query, re.IGNORECASE)

    # Normalize to remove spaces
    found_ids = [aid.replace(' ', '-').replace('--', '-').upper() for aid in found_ids]
    found_ids = list(dict.fromkeys(found_ids))  # Remove duplicates while preserving order

    if not found_ids:
        return json.dumps({
            "error": "No article IDs found in query. Please provide an article ID like 'HT-103' or 'GM-205'.",
            "articles": [],
            "found_count": 0,
            "not_found": [],
            "_result_type": "article_info"
        })

    # Fetch descriptions
    descriptions = fetch_article_descriptions(found_ids)

    articles = []
    not_found = []

    for aid in found_ids:
        if aid in descriptions and descriptions[aid]:
            articles.append({
                "article_id": aid,
                "description": descriptions[aid]
            })
        else:
            not_found.append(aid)

    return json.dumps({
        "articles": articles,
        "found_count": len(articles),
        "not_found": not_found,
        "_result_type": "article_info"
    })


@tool
def get_store_info() -> str:
    """Fetch store master data: store IDs, open/close times, and footfall counts.

    Call this when the user asks about store-level context such as:
    - Store details, operating hours, or IDs
    - Store-level footfall totals from the store master
    - Background context for store-specific recommendations

    Returns JSON with keys: stores, count, _result_type.
    """
    from tools.warehouse_tools import fetch_store_master
    stores = fetch_store_master()
    return json.dumps({
        "stores": stores,
        "count": len(stores),
        "_result_type": "store_info"
    }, default=str)


@tool
def get_active_promotions(start_date: str, end_date: str) -> str:
    """Fetch promotion records that are active within a date range.

    Call this when the user asks about:
    - Which promotions are currently running or active
    - Promotions for a specific date or date range
    - Cross-domain context linking promotions to workforce or footfall

    start_date: ISO date string YYYY-MM-DD (use today's date for current promotions)
    end_date: ISO date string YYYY-MM-DD

    Returns JSON with keys: promotions, count, _result_type.
    """
    from tools.warehouse_tools import fetch_promotion_master
    from datetime import date as _date
    if start_date.lower() == "today":
        start_date = _date.today().isoformat()
    if end_date.lower() == "today":
        end_date = _date.today().isoformat()
    all_promos = fetch_promotion_master()
    active = []
    for p in all_promos:
        promo_start = str(p.get("PromoStartDate") or "")
        promo_end = str(p.get("PromoEndDate") or "")
        if not promo_start or not promo_end:
            active.append(p)
            continue
        try:
            if promo_start <= end_date and promo_end >= start_date:
                active.append(p)
        except Exception:
            active.append(p)
    return json.dumps({
        "promotions": active,
        "count": len(active),
        "_result_type": "promotions"
    }, default=str)


# ─────────────────────────── WRITE / GENERATE TOOLS ──────────────────────────


@tool
def start_roster_generation(date_query: str) -> str:
    """Generate a NEW workforce roster for a date or date range.

    Call this when the user wants to CREATE or GENERATE a new roster. Examples:
    - "generate roster for 21 march to 23 march 2026"
    - "create the roster for next week"
    - "build a roster for 15 to 17 june"
    - "prepare the roster for tomorrow"

    This starts an asynchronous background job. Use get_roster_job_status
    to check progress after calling this tool.

    date_query: the user's natural-language date range description.

    Returns JSON with keys: job_id, start_date, end_date, _result_type.
    """
    from agents import roster_generation_agent as rg
    result = rg.start_roster_job(question=date_query)
    result["_result_type"] = "roster_generation"
    return json.dumps(result, default=str)


@tool
def preview_promotion_roster(date_query: str) -> str:
    """Generate a PREVIEW of the promotion support roster for a specific date.

    Call this when the user wants to generate or create a promotion roster:
    - "generate promotion roster for 15 march 2026"
    - "create promotion support shifts for 15 january 2026"
    - "regenerate the promotion roster for tomorrow"

    This generates the rows but does NOT post them to the backend yet.

    IMPORTANT: After calling this tool, you MUST:
    1. Show the user a summary of the preview rows
    2. Ask: "Do you confirm posting these rows? Reply 'yes' to proceed or 'no' to cancel."
    3. Only call confirm_write_operation once the user explicitly confirms.

    date_query: natural language date (e.g. "15 march 2026", "tomorrow").

    Returns JSON with keys: confirmation_token, roster_date, rows, count, _result_type.
    """
    from agents import promotion_roster_regeneration_agent as pr
    roster_date = pr._resolve_date({"query": date_query})
    if not roster_date:
        return json.dumps({
            "error": (
                f"Could not determine a date from '{date_query}'. "
                "Please specify a date (e.g. '15 march 2026')."
            ),
            "_result_type": "promotion_roster_preview"
        })
    rows = pr.generate_rows(roster_date)
    if not rows:
        return json.dumps({
            "error": f"No active promotion departments found for {roster_date}.",
            "roster_date": roster_date,
            "_result_type": "promotion_roster_preview"
        })
    token = str(uuid.uuid4())
    _PENDING_STORE[token] = {"type": "promotion", "rows": rows, "roster_date": roster_date}
    return json.dumps({
        "confirmation_token": token,
        "roster_date": roster_date,
        "rows": rows,
        "count": len(rows),
        "_result_type": "promotion_roster_preview"
    }, default=str)


@tool
def preview_exception_roster(query: str) -> str:
    """Generate a PREVIEW of exception roster adjustments for leave/absence.

    Call this when the user wants to handle workforce exceptions:
    - "generate exception roster"
    - "update the exception roster"
    - "handle leave / sick leave / absence adjustments"
    - "exception roster for 15 march 2026"

    This detects today's and future employee exceptions (leave, sick leave,
    absence) and generates updated rows with replacements. The rows are NOT
    posted yet.

    IMPORTANT: After calling this tool, you MUST:
    1. Show the user a summary of the preview rows and dates processed
    2. Ask: "Do you confirm posting these rows? Reply 'yes' to proceed or 'no' to cancel."
    3. Only call confirm_write_operation once the user explicitly confirms.

    query: the user's query — may contain a specific date or be generic
    (automatic mode processes today + all future exceptions).

    Returns JSON with keys: confirmation_token, dates_processed, rows, count, _result_type.
    """
    from agents import workforce_exception_roster_agent as ex
    rows, dates = ex.generate_rows({"query": query})
    if not rows:
        return json.dumps({
            "message": (
                f"No applicable exceptions (today or future) found. "
                f"Dates checked: {dates or 'none'}."
            ),
            "dates_processed": dates,
            "rows": [],
            "_result_type": "exception_roster_preview"
        })
    token = str(uuid.uuid4())
    _PENDING_STORE[token] = {"type": "exception", "rows": rows, "dates": dates}
    return json.dumps({
        "confirmation_token": token,
        "dates_processed": dates,
        "rows": rows,
        "count": len(rows),
        "_result_type": "exception_roster_preview"
    }, default=str)


@tool
def confirm_write_operation(confirmation_token: str) -> str:
    """Execute a previously previewed write operation by posting all rows to the backend.

    Call this ONLY when ALL of these conditions are true:
    1. The user has EXPLICITLY said 'yes', 'confirm', 'ok', 'go ahead', or 'proceed'
    2. A valid confirmation_token from a previous preview_promotion_roster or
       preview_exception_roster call is available in this conversation

    NEVER call this tool speculatively or without explicit user confirmation.
    NEVER call this if the user said 'no', 'cancel', or 'stop'.

    confirmation_token: the exact token returned by the previous preview_* tool call.

    Returns JSON with keys: posted, errors, posted_count, operation_type, _result_type.
    """
    pending = _PENDING_STORE.get(confirmation_token)
    if not pending:
        return json.dumps({
            "error": (
                "Invalid or expired confirmation_token. "
                "Please generate the roster preview again."
            ),
            "_result_type": "confirm_write"
        })
    op_type = pending["type"]
    rows = pending["rows"]
    emp_names = list({r.get("EmplName") for r in rows if r.get("EmplName")})
    try:
        if op_type == "promotion":
            from agents import promotion_roster_regeneration_agent as pr
            result = pr.post_rows(rows)
        else:
            from agents import workforce_exception_roster_agent as ex
            result = ex.post_rows(rows)
        del _PENDING_STORE[confirmation_token]
        return json.dumps({
            "posted": result.get("posted", []),
            "errors": result.get("errors", []),
            "posted_count": len(result.get("posted", [])),
            "operation_type": op_type,
            "employee_names": emp_names,
            "_result_type": "confirm_write"
        }, default=str)
    except Exception as e:
        return json.dumps({
            "error": str(e),
            "_result_type": "confirm_write"
        })
