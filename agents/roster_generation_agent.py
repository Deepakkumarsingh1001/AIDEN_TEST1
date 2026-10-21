from flask import Flask, request, jsonify
import json
import time
import requests
import os
import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import TypedDict, List, Dict, Any

# Make the project root importable so `from prompts...`, `from utils import Utils`
# and `from tools...` resolve when this file is run directly
# (python agents/roster_generation_agent.py).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

# --- LOCAL MODULE IMPORTS ---
from prompts import DAILY_ROSTER_PROMPT
from utils import Utils
import BTP_Service  # Your team's BTP SAP module

app = Flask(__name__)
u = Utils()
JOB_STATUS_DB = {}
# ---------------- LLM INIT ---------------- #

def get_llm():
    """Initializes the LLM on demand."""
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
        if str(emp.get("WeeklyOff", "")).strip().lower() != day_name.lower()
    ]

    prompt_value = DAILY_ROSTER_PROMPT.format(
        target_date=date_str,
        target_day=day_name,
        store_data_str=json.dumps(state["store_data"], indent=2),
        workforce_data_str=json.dumps(available_staff, indent=2)
    )

    app.logger.info(f"Generating JSON roster for {date_str} ({day_name})...")

    response = llm.invoke(prompt_value)
    content = response.content if hasattr(response, "content") else str(response)

    # Clean up markdown formatting just in case the LLM includes it
    content = content.strip()
    if content.startswith("```json"):
        content = content[7:]
    elif content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    content = content.strip()

    try:
        daily_roster = json.loads(content)
        if not isinstance(daily_roster, list):
            raise ValueError("LLM output is not a JSON array")
    except Exception as e:
        app.logger.error(f"Error parsing JSON on {date_str}: {e}")
        daily_roster = []

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


# ---------------- DATE EXTRACTION HELPER (LLM-BASED) ---------------- #

def extract_dates_from_question(question: str) -> Dict[str, str]:
    """
    Extracts start_date and end_date from a natural language question using the LLM.
    Returns: {"start_date": "YYYY-MM-DD", "end_date": "YYYY-MM-DD"}

    Example:
        >>> extract_dates_from_question("Generate roster from January 1 to February 15")
        {'start_date': '2026-01-01', 'end_date': '2026-02-15'}
    """
    try:
        extraction_prompt = f"""Extract the start date and end date from the following question.
            Return the dates in YYYY-MM-DD format. Use the current date context if needed (today is {datetime.now().strftime('%Y-%m-%d')}).

            Question: {question}

            Provide the response in this exact format:
            START_DATE: YYYY-MM-DD
            END_DATE: YYYY-MM-DD"""

        response = llm.invoke(extraction_prompt)
        content = response.content if hasattr(response, "content") else str(response)

        # Parse the response
        lines = content.strip().split('\n')
        start_date = None
        end_date = None

        for line in lines:
            if 'START_DATE' in line.upper():
                start_date = line.split(':')[-1].strip()
            elif 'END_DATE' in line.upper():
                end_date = line.split(':')[-1].strip()

        if not start_date or not end_date:
            raise ValueError(f"Failed to extract dates from LLM response: {content}")

        # Validate date format
        datetime.strptime(start_date, "%Y-%m-%d")
        datetime.strptime(end_date, "%Y-%m-%d")

        app.logger.info(f"Extracted dates - Start: {start_date}, End: {end_date}")
        return {"start_date": start_date, "end_date": end_date}

    except Exception as e:
        app.logger.error(f"Date extraction failed: {e}")
        raise


# ---------------- SAP DATA HELPERS ---------------- #
# The OData transport + token logic now lives in tools/ (single source of truth,
# with cached OAuth). These thin adapters keep the existing call sites + signatures
# working while delegating to the shared tools. `destination_name` is accepted for
# backwards compatibility but no longer used (base URL resolved via tools.auth).

from tools.warehouse_tools import (
    fetch_store_master,
    fetch_workforce_master,
    post_roster_batch,
    post_store_job_id,
    delete_existing_roster as _delete_roster_tool,
    coerce_decimal,
)
from tools.odata_client import odata_get, odata_post, odata_batch

_ENTITY_FETCHERS = {
    "StoreMaster": fetch_store_master,
    "WorkforceMaster": fetch_workforce_master,
}


def fetch_capm_entity(destination_name: str, entity_name: str) -> list:
    fetcher = _ENTITY_FETCHERS.get(entity_name)
    return fetcher() if fetcher else odata_get(entity_name)


def post_capm_entity(destination_name: str, entity_name: str, payload: dict) -> dict:
    if entity_name == "StoreJobId":
        return post_store_job_id(payload.get("jobId"))
    return odata_post(entity_name, payload)


def post_capm_batch(destination_name: str, entity_name: str, payload_list: list) -> dict:
    if entity_name == "RosterMaster":
        return post_roster_batch(payload_list)
    return odata_batch(entity_name, payload_list)


def delete_existing_roster(destination_name: str, start_date: str, end_date: str):
    return _delete_roster_tool(start_date, end_date)


def _generate_day(date_str: str, day_name: str, store_data: list, workforce_data: list) -> list:
    """Generates roster for a single day. Called concurrently by ThreadPoolExecutor."""
    available_staff = [
        emp for emp in workforce_data
        if str(emp.get("WeeklyOff", "")).strip().lower() != day_name.lower()
    ]
    prompt_value = DAILY_ROSTER_PROMPT.format(
        target_date=date_str,
        target_day=day_name,
        store_data_str=json.dumps(store_data, indent=2),
        workforce_data_str=json.dumps(available_staff, indent=2)
    )
    app.logger.info(f"Generating JSON roster for {date_str} ({day_name})...")
    response = llm.invoke(prompt_value)
    content = response.content if hasattr(response, "content") else str(response)
    content = content.strip()
    if content.startswith("```json"):
        content = content[7:]
    elif content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    content = content.strip()
    try:
        daily_roster = json.loads(content)
        if not isinstance(daily_roster, list):
            raise ValueError("LLM output is not a JSON array")
    except Exception as e:
        app.logger.error(f"Error parsing JSON on {date_str}: {e}")
        daily_roster = []
    return daily_roster


def run_roster_pipeline_async(job_id: str, initial_state: dict, dest_name: str):
    """Runs the LangGraph agent and HANA POST operations in a background thread."""

    # Grab the dates from the state dictionary
    start_date = initial_state["start_date"]
    end_date = initial_state["end_date"]

    JOB_STATUS_DB[job_id] = {"status": "processing", "message": f"Cleaning up existing records from {start_date} to {end_date}..."}
    print(f"🚀 [Job {job_id}] Background generation started...")
    start_time = time.perf_counter()

    try:

        print(f" [Job {job_id}] Deleting existing records...")
        delete_existing_roster(dest_name, start_date, end_date)

        # 1. Generate all days in parallel instead of sequentially via LangGraph
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt   = datetime.strptime(end_date,   "%Y-%m-%d")
        all_dates = []
        d = start_dt
        while d <= end_dt:
            all_dates.append(d)
            d += timedelta(days=1)

        results_map = {}
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_date = {
                executor.submit(
                    _generate_day,
                    d.strftime("%Y-%m-%d"),
                    d.strftime("%A"),
                    initial_state["store_data"],
                    initial_state["all_workforce_data"]
                ): d for d in all_dates
            }
            for future in as_completed(future_to_date):
                results_map[future_to_date[future]] = future.result()

        generated_roster = []
        for d in all_dates:
            generated_roster.extend(results_map[d])

        print(f" [Job {job_id}] LLM generated {len(generated_roster)} shifts. Mapping data...")
        JOB_STATUS_DB[job_id]["message"] = f"Mapping {len(generated_roster)} shifts to HANA format..."

        # 2. Map the data safely
        hana_payload_list = []
        for record in generated_roster:
            try:
                store_id_clean = str(int(record.get("store_id", "1")))
            except ValueError:
                store_id_clean = record.get("store_id", "1")

            shift_type = str(record.get("shift_type", "")).upper()
            if shift_type == "MORNING":
                shift_string = "Morning (8AM-6PM)"
            elif shift_type == "EVENING":
                shift_string = "Evening (12PM-10PM)"
            else:
                shift_string = f"{shift_type.capitalize()} ({record.get('shift_start')}-{record.get('shift_end')})"

            raw_date = str(record.get("date", "")).replace("-", "")
            empl_id = str(record.get("empl_id", ""))
            empl_num = empl_id.replace("EMP", "") if empl_id.startswith("EMP") else empl_id
            custom_roster_id = f"R-{raw_date}-{empl_num}"

            hana_payload_list.append({
                "EmplId": empl_id,
                "EmplName": record.get("empl_name"),
                "Experience": coerce_decimal(record.get("experience")),
                "Mch1": record.get("department"),
                "RoasterId": custom_roster_id,
                "RosterDate": record.get("date"),
                "RosterMonth": record.get("month"),
                "Shift": shift_string,
                "StoreId": store_id_clean,
                "WeekDay": record.get("day"),
                "WeeklyOff": record.get("weekly_off")
            })

        # 3. POST the massive batch to SAP HANA in MICRO-BATCHES
        total_records = len(hana_payload_list)

        if total_records > 0:
            chunk_size = 200
            chunks = [hana_payload_list[i:i + chunk_size] for i in range(0, total_records, chunk_size)]
            total_chunks = len(chunks)
            print(f"[Job {job_id}] Preparing to POST {total_records} records in {total_chunks} parallel chunks of {chunk_size}...")
            JOB_STATUS_DB[job_id]["message"] = f"Posting {total_chunks} batches to SAP HANA in parallel..."

            with ThreadPoolExecutor(max_workers=total_chunks) as executor:
                futures = [executor.submit(post_capm_batch, dest_name, 'RosterMaster', chunk) for chunk in chunks]
                for f in as_completed(futures):
                    f.result()  # re-raises any SAP error immediately

        end_time = time.perf_counter()

        # 4. Mark Job as Completed
        print(f"✅ [Job {job_id}] SUCCESS! Finished in {round(end_time - start_time, 2)} seconds.")
        JOB_STATUS_DB[job_id] = {
            "status": "completed",
            "message": f"Successfully posted {total_records} records to SAP HANA across {total_chunks if total_records > 0 else 0} batches."
        }

    except Exception as e:
        print(f" [Job {job_id}] FAILED: {str(e)}")
        JOB_STATUS_DB[job_id] = {
            "status": "failed",
            "message": f"Error: {str(e)}"
        }

# ---------------- REUSABLE JOB STARTER ---------------- #

DEST_NAME = "Warehouse-copilot-NewDestination"

# Most recent job id started in THIS process (survives across chat sessions).
_LAST_JOB = {"id": None}


def get_latest_job_id() -> str | None:
    """Latest roster job id: the one started in this process, else from the backend
    StoreJobId table (so status works even when asked in a fresh chat session)."""
    if _LAST_JOB["id"]:
        return _LAST_JOB["id"]
    try:
        from tools.warehouse_tools import fetch_latest_job_id
        return fetch_latest_job_id()
    except Exception as e:
        print(f"[ROSTER] fetch_latest_job_id failed: {e}")
        return None


def start_roster_job(question: str = None, start_date: str = None, end_date: str = None) -> dict:
    """Resolve dates, fetch master data, and kick off the async roster job.

    Returns {"job_id", "start_date", "end_date"}. Reused by the /generate-roster
    route and by the main.py chatbot router so the logic lives in one place.
    """
    if not start_date or not end_date:
        if not question:
            raise ValueError("Provide either (question) OR (start_date and end_date)")
        extracted = extract_dates_from_question(question)
        start_date = extracted["start_date"]
        end_date = extracted["end_date"]

    app.logger.info(f"Using dates - Start: {start_date}, End: {end_date}")

    # Fetch master data from SAP BTP in parallel.
    with ThreadPoolExecutor(max_workers=2) as executor:
        f_store     = executor.submit(fetch_capm_entity, DEST_NAME, 'StoreMaster')
        f_workforce = executor.submit(fetch_capm_entity, DEST_NAME, 'WorkforceMaster')
        store_data     = f_store.result()
        workforce_data = f_workforce.result()

    initial_state = {
        "start_date": start_date,
        "end_date": end_date,
        "store_data": store_data,
        "all_workforce_data": workforce_data,
        "final_roster": [],
        "current_date": None,
    }

    job_id = str(uuid.uuid4())
    worker_thread = threading.Thread(
        target=run_roster_pipeline_async,
        args=(job_id, initial_state, DEST_NAME),
    )
    worker_thread.start()
    post_capm_entity(DEST_NAME, 'StoreJobId', payload={"jobId": job_id})
    _LAST_JOB["id"] = job_id
    return {"job_id": job_id, "start_date": start_date, "end_date": end_date}


# ---------------- API ENDPOINT ---------------- #

@app.route("/generate-roster", methods=["POST"])
def generate_roster():

    try:
        data = request.get_json()
        try:
            result = start_roster_job(
                question=data.get("question"),
                start_date=data.get("start_date"),
                end_date=data.get("end_date"),
            )
        except ValueError as ve:
            return jsonify({"error": str(ve)}), 400
        except Exception as fetch_error:
            app.logger.error(f"SAP Fetch Failed: {str(fetch_error)}")
            return jsonify({"error": "Failed to load master data from SAP", "details": str(fetch_error)}), 502

        return jsonify({
            "status": "accepted",
            "job_id": result["job_id"],
            "message": f"Roster generation for {result['start_date']} to {result['end_date']} has started in the background. It will be automatically posted to SAP HANA upon completion."
        }), 202

    except Exception as e:
        app.logger.exception("Route failed")
        return jsonify({"error": str(e)}), 500

@app.route("/job-status/<job_id>", methods=["GET"])
def get_job_status(job_id):
    """Endpoint to check the status of a background roster generation job."""
    job_info = JOB_STATUS_DB.get(job_id)

    if not job_info:
        return jsonify({"error": "Job ID not found. It may be invalid or the server restarted."}), 404

    return jsonify({
        "job_id": job_id,
        "status": job_info["status"],
        "message": job_info["message"]
    }), 200

if __name__ == "__main__":
    cf_port = os.getenv("PORT")
    if cf_port is None:
        app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
    else:
        app.run(host='0.0.0.0', port=int(cf_port), debug=False)
