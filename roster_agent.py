from flask import Flask, request, jsonify
import json
import time
from datetime import datetime, timedelta
from typing import TypedDict, List, Dict, Any

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from prompts import DAILY_ROSTER_PROMPT
from utils import Utils

app = Flask(__name__)
u = Utils()

# ---------------- LLM INIT ---------------- #

def get_llm():
    llm = u.initialize_llm4()
    if not llm:
        raise ValueError("Failed to initialize LLM.")
    return llm

llm = get_llm()

# ---------------- STATE ---------------- #

class RosterState(TypedDict):
    start_date: str
    end_date: str
    current_date: datetime
    store_data: List[Dict[str, Any]]
    all_workforce_data: List[Dict[str, Any]]
    final_roster: List[Dict[str, Any]]

# ---------------- GRAPH NODES ---------------- #

def initialize_roster(state: RosterState) -> Command:
    start_dt = datetime.strptime(state["start_date"], "%Y-%m-%d")
    return Command(
        goto="generate",
        update={
            "current_date": start_dt,
            "final_roster": []
        }
    )

def generate_daily_roster(state: RosterState) -> Command:
    curr_date = state["current_date"]
    day_name = curr_date.strftime("%A")
    date_str = curr_date.strftime("%Y-%m-%d")

    # Filter weekly off
    available_staff = [
        emp for emp in state["all_workforce_data"]
        if str(emp.get("Weekly Off", "")).strip().lower() != day_name.lower()
    ]

    prompt_value = DAILY_ROSTER_PROMPT.format(
        target_date=date_str,
        target_day=day_name,
        store_data_str=json.dumps(state["store_data"], indent=2),
        workforce_data_str=json.dumps(available_staff, indent=2)
    )

    response = llm.invoke(prompt_value)
    content = response.content if hasattr(response, "content") else str(response)

    # Safety cleanup
    content = content.replace("```json", "").replace("```", "").strip()

    try:
        daily_roster = json.loads(content)
        if not isinstance(daily_roster, list):
            raise ValueError("LLM output is not a JSON array")
    except Exception as e:
        raise ValueError(f"Invalid JSON returned by LLM: {e}")

    final_roster = state.get("final_roster", [])
    final_roster.extend(daily_roster)

    return Command(
        goto="increment",
        update={"final_roster": final_roster}
    )

def increment_date(state: RosterState) -> Command:
    next_date = state["current_date"] + timedelta(days=1)
    end_dt = datetime.strptime(state["end_date"], "%Y-%m-%d")

    return Command(
        goto=END if next_date > end_dt else "generate",
        update={"current_date": next_date}
    )

# ---------------- GRAPH BUILD ---------------- #

workflow = StateGraph(RosterState)
workflow.add_node("initialize", initialize_roster)
workflow.add_node("generate", generate_daily_roster)
workflow.add_node("increment", increment_date)
workflow.add_edge(START, "initialize")

roster_agent = workflow.compile()

# ---------------- API ---------------- #

@app.route("/generate-roster", methods=["POST"])
def generate_roster():
    start_time = time.perf_counter()
    data = request.get_json()

    if not data:
        return jsonify({"error": "No JSON payload provided"}), 400

    required = ["start_date", "end_date", "store_data", "workforce_data"]
    if not all(k in data for k in required):
        return jsonify({"error": "Missing required fields"}), 400

    initial_state = {
        "start_date": data["start_date"],
        "end_date": data["end_date"],
        "store_data": data["store_data"],
        "all_workforce_data": data["workforce_data"],
        "final_roster": [],
        "current_date": None
    }

    final_state = roster_agent.invoke(
        initial_state,
        config={"recursion_limit": 100}
    )

    end_time = time.perf_counter()

    return jsonify({
        "start_date": data["start_date"],
        "end_date": data["end_date"],
        "total_records": str(len(final_state["final_roster"])),
        "execution_time_sec": round(end_time - start_time, 2),
        "roster": final_state["final_roster"]
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)