"""
Offline stand-in for the gpt-4o chat model (SAP Gen AI Hub), used when MOCK_MODE=true.

It is rule-based, not generative: it recognises each agent's prompt, builds a
plausible answer from the (mock) data embedded in that prompt, and — for the
chatbot's tool-calling agent — routes user messages to the right tool by keyword.
"""

import json
import re
import uuid
from datetime import date, datetime, timedelta
from typing import Any, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

_PREFIX_DEPT = {
    "HB": "Handbags", "BP": "Backpacks", "KL": "Kids Line", "AS": "Accessories",
    "WS": "Women-Shoes", "WL": "Wallets & Small Leather Goods",
    "NA": "New Arrival", "FC": "Featured Collection",
}


def _json_arrays(text: str) -> List[list]:
    """All JSON arrays-of-objects embedded in a prompt."""
    dec, out, i = json.JSONDecoder(), [], 0
    while True:
        i = text.find("[", i)
        if i < 0:
            return out
        try:
            obj, end = dec.raw_decode(text[i:])
            if isinstance(obj, list) and obj and isinstance(obj[0], dict):
                out.append(obj)
            i += max(end, 1)
        except ValueError:
            i += 1


def _find(arrays: List[list], key: str) -> list:
    return next((a for a in arrays if key in a[0]), [])


def _roster_for_day(prompt: str) -> str:
    target = re.search(r"Target Date:\s*(\d{4}-\d{2}-\d{2})", prompt).group(1)
    day = re.search(r"Target Day:\s*(\w+)", prompt).group(1)
    staff = _find(_json_arrays(prompt), "EmplId")
    store = _find(_json_arrays(prompt), "StoreId")
    store_id = store[0]["StoreId"] if store else "S001"
    month = datetime.strptime(target, "%Y-%m-%d").strftime("%B")
    rows = []
    for n, e in enumerate(staff):
        morning = n % 2 == 0
        rows.append({
            "date": target, "day": day, "month": month, "store_id": store_id,
            "department": e.get("CurrDept"), "empl_id": str(e["EmplId"]), "empl_name": e.get("EmplName"),
            "experience": e.get("ExperienceYear"), "weekly_off": e.get("WeeklyOff"),
            "shift_type": "MORNING" if morning else "EVENING",
            "shift_start": "08:00" if morning else "12:00",
            "shift_end": "18:00" if morning else "22:00",
        })
    return json.dumps(rows)


def _promotion_rows(prompt: str) -> str:
    arrays = _json_arrays(prompt)
    promos, base = _find(arrays, "PromoStartDate"), _find(arrays, "RoasterId")
    if not promos or not base:
        return "[]"
    depts = [d for d in dict.fromkeys(_PREFIX_DEPT.get(str(p["ArticleId"]).split("-")[0]) for p in promos) if d]
    ymd = str(base[0]["RosterDate"]).replace("-", "")
    rows = []
    for n, dept in enumerate(depts, 1):
        src = next((r for r in base if r.get("Mch1") == dept), None)
        if not src:
            continue
        rows.append({**{k: src.get(k) for k in ("EmplId", "EmplName", "StoreId", "WeekDay", "WeeklyOff",
                                                  "Experience", "RosterDate", "RosterMonth", "Mch1")},
                     "Shift": "Promotion Support (11AM-9PM)", "RoasterId": f"R-{ymd}-NEW-{n}"})
    return json.dumps(rows)


def _exception_rows(prompt: str) -> str:
    arrays = _json_arrays(prompt)
    exc, base = _find(arrays, "ExceptionCase"), _find(arrays, "RoasterId")
    rows = []
    for x in exc:
        for r in base:
            if str(r.get("EmplId")) == str(x["EmplId"]) and r.get("RosterDate") == x["ExceptionDate"]:
                rows.append({**r, "Shift": x["ExceptionCase"]})
    return json.dumps(rows)


def _user_query(prompt: str) -> str:
    m = re.search(r"USER INPUT\s*\n=+\s*\n(.*?)\n\s*\nFollow the instructions", prompt, re.S)
    return (m.group(1) if m else "").lower()


def _analysis(prompt: str) -> Optional[str]:
    """Mimic the footfall / promotion-readiness agents' JSON OUTPUT CONTRACT."""
    arrays = _json_arrays(prompt)
    q = _user_query(prompt)
    wants_reco = any(k in q for k in ("improve", "recommend", "suggest", "strategy", "why", "drop"))
    ff = _find(arrays, "FootFallCount")
    if ff:
        if wants_reco:
            agg = {}
            for r in ff:
                a = agg.setdefault(r["Department"], {"c": 0.0, "n": 0, "s": r["IndustryStandardPct"]})
                a["c"] += r["SalesConversionPct"]; a["n"] += 1
            rows = [{"Department": d, "SalesConversionPct": round(a["c"] / a["n"], 1),
                     "IndustryStandardPct": a["s"],
                     "Recommendation": "Cross-merchandise with top converters; refresh endcap displays; "
                                       "add staff during peak hours; run a targeted promotion"}
                    for d, a in agg.items() if a["c"] / a["n"] < a["s"]]
            cols = ["Department", "SalesConversionPct", "IndustryStandardPct", "Recommendation"]
        else:
            cols = ["Department", "TransactionDate", "FootFallCount", "ReceiptCount", "SalesConversionPct",
                    "IndustryStandardPct", "Variance", "Colour", "PerformanceComparison"]
            rows = [{c: r.get(c) for c in cols} for r in ff[:100]]
        return json.dumps({"columns": cols, "rows": rows, "message": ""})
    pr = _find(arrays, "ReadinessPct")
    if pr:
        if wants_reco:
            cols = ["ArticleId", "ArticleDescription", "ReadinessPct", "QuantityOrderSuggestion", "Recommendation"]
            rows = [{"ArticleId": r["ArticleId"], "ArticleDescription": r.get("ArticleDescription", "Not Available"),
                     "ReadinessPct": r["ReadinessPct"], "QuantityOrderSuggestion": r.get("OrderToBePlaced"),
                     "Recommendation": "Complete pricing before launch; finish labelling; "
                                       "validate promotion setup; prioritise low-readiness items"}
                    for r in pr]
        elif any(k in q for k in ("quantity", "order", "stock", "replenish")):
            cols = ["ArticleId", "ArticleDescription", "ReadinessPct", "QuantityOrderSuggestion"]
            rows = [{"ArticleId": r["ArticleId"], "ArticleDescription": r.get("ArticleDescription", "Not Available"),
                     "ReadinessPct": r["ReadinessPct"], "QuantityOrderSuggestion": r.get("OrderToBePlaced")}
                    for r in pr]
        else:
            cols = ["ArticleId", "ArticleDescription", "Department", "PromotionId", "ReadinessPct", "PricingStatus",
                    "LabellingStatus", "ReadinessStatus", "LabellingCompletionPct"]
            rows = [{c: r.get(c) for c in cols} for r in pr]
        return json.dumps({"columns": cols, "rows": rows, "message": ""})
    return None

def _summarise_tool_result(body: str) -> str:
    try:
        d = json.loads(body)
    except ValueError:
        return "Here is what I found in the latest warehouse data:\n\n" + body[:3000]
    if not isinstance(d, dict):
        return "Here is what I found in the latest warehouse data."
    kind = d.get("_result_type")
    if kind == "roster_generation":
        return (f"Roster generation has started for {d.get('start_date')} to {d.get('end_date')}. "
                "Ask me for the roster status in a moment.")
    if kind in ("promotion", "exception", "promotion_roster_preview", "exception_roster_preview") \
            or "confirmation_token" in d:
        return (f"I've prepared {d.get('count', 0)} roster rows for your review. "
                "Reply **confirm** to post them, or tell me what to change.")
    if kind == "staffing_risk":
        n = d.get("total_departments_at_risk", 0)
        return "No departments are currently at staffing risk." if not n else f"{n} departments are at staffing risk."
    if "posted" in d or "posted_count" in d:
        return f"Done - {d.get('posted_count', d.get('posted', 0))} rows were posted successfully."
    return "Here is what I found in the latest warehouse data:\n\n" + body[:3000]

def _dates_from(question: str) -> tuple:
    today = date.today()
    iso = re.findall(r"\d{4}-\d{2}-\d{2}", question)
    if len(iso) >= 2:
        return iso[0], iso[1]
    if len(iso) == 1:
        return iso[0], iso[0]
    q = question.lower()
    if "tomorrow" in q:
        d = today + timedelta(days=1)
        return d.isoformat(), d.isoformat()
    if "next week" in q:
        s = today - timedelta(days=today.weekday()) + timedelta(days=7)
        return s.isoformat(), (s + timedelta(days=6)).isoformat()
    if "week" in q:
        s = today - timedelta(days=today.weekday())
        return s.isoformat(), (s + timedelta(days=6)).isoformat()
    return today.isoformat(), (today + timedelta(days=6)).isoformat()


# (keywords, tool, arg-builder) — first match wins
_ROUTES = [
    (("confirm", "yes", "proceed", "go ahead", "post it"), "confirm_write_operation", "token"),
    (("job status", "roster status", "generation status"), "get_roster_job_status", {}),
    (("generate roster", "create roster", "generate the roster"), "start_roster_generation", "date_query"),
    (("promotion roster", "promotion support"), "preview_promotion_roster", "date_query"),
    (("exception", "sick leave", "absent", "leave"), "preview_exception_roster", "query"),
    (("staffing risk", "at risk", "risk"), "get_staffing_risk", {}),
    (("workforce", "headcount", "how many staff", "employees"), "get_workforce_summary", {}),
    (("readiness",), "get_promotion_readiness", "query"),
    (("active promotion", "promotions"), "get_active_promotions", "dates"),
    (("footfall", "conversion", "department performance"), "get_footfall_analysis", "query"),
    (("article", "sku"), "get_article_info", "query"),
    (("store info", "store details", "opening hours", "about the store"), "get_store_info", {}),
    (("roster", "shift", "schedule", "who is working"), "view_roster", "date_range_query"),
]


class MockChatModel(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "mock-gpt-4o"

    def bind_tools(self, tools: Any, **kwargs: Any):  # noqa: D401 - langchain API
        return self

    # ── core ────────────────────────────────────────────────────────────────
    def _generate(self, messages: List[BaseMessage], stop=None, run_manager=None, **kwargs) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self._respond(messages))])

    def _respond(self, messages: List[BaseMessage]) -> AIMessage:
        text = "\n".join(str(m.content) for m in messages)
        last = messages[-1]

        # Chatbot (tool-calling) flow
        if isinstance(last, ToolMessage):
            return AIMessage(content=_summarise_tool_result(str(last.content)))
        if "Warehouse CoPilot" in text or any(getattr(m, "type", "") == "system" for m in messages):
            call = self._route(messages)
            if call:
                return AIMessage(content="", tool_calls=[call])
            if getattr(last, "type", "") == "human":
                return AIMessage(content="Hello! I can help with rosters, footfall analysis, promotion "
                                         "readiness, staffing risk and workforce questions. What would you like to see?")

        # Single-shot agent prompts
        if "START_DATE: YYYY-MM-DD" in text:
            q = text.split("Question:", 1)[-1].split("Provide the response")[0]
            s, e = _dates_from(q)
            return AIMessage(content=f"START_DATE: {s}\nEND_DATE: {e}")
        if "Workforce Roster Generation Agent" in text and "Target Date:" in text:
            return AIMessage(content=_roster_for_day(text))
        if "Promotion Support" in text and "OUTPUT CONTRACT" in text:
            return AIMessage(content=_promotion_rows(text))
        if "ExceptionCase" in text and "OUTPUT CONTRACT" in text:
            return AIMessage(content=_exception_rows(text))
        analysis = _analysis(text)
        if analysis:
            return AIMessage(content=analysis)
        return AIMessage(content="Based on the available warehouse data, operations look stable. "
                                 "Ask me about rosters, footfall, or promotion readiness for details.")

    def _route(self, messages: List[BaseMessage]) -> Optional[dict]:
        human = [m for m in messages if getattr(m, "type", "") == "human"]
        if not human:
            return None
        q = str(human[-1].content)
        ql = q.lower()
        for keys, tool, arg in _ROUTES:
            if any(k in ql for k in keys):
                if arg == "token":
                    tok = re.findall(r"confirmation_token:\s*([0-9a-f\-]{36})",
                                     "\n".join(str(m.content) for m in messages))
                    if not tok:
                        continue
                    args = {"confirmation_token": tok[-1]}
                elif arg == "dates":
                    s, e = _dates_from(q)
                    args = {"start_date": s, "end_date": e}
                elif isinstance(arg, str):
                    s, _ = _dates_from(q)
                    has_iso = re.search(r"\d{4}-\d{2}-\d{2}", q)
                    relative = any(w in ql for w in ("today", "tomorrow", "week"))
                    args = {arg: q if has_iso or not relative else f"{q} ({s})"}
                else:
                    args = arg
                return {"name": tool, "args": args, "id": "call_" + uuid.uuid4().hex[:12], "type": "tool_call"}
        return None
