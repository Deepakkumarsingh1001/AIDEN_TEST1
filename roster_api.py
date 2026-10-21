from flask import Flask, request, Response, jsonify
import pandas as pd
import json
import os
import time
from utils import Utils

def get_llm():
    """
    Initializes the LLM on demand.
    """
    llm = u.initialize_llm4()
    if not llm:
        raise ValueError("Failed to initialize LLM.")
    return llm



app = Flask(__name__)

u = Utils()
llm = get_llm()  

# Load Excel Master Data at startup 
EXCEL_FILE = "./AIDEN_Dataset.xlsx"

try:
    store_master_df = pd.read_excel(EXCEL_FILE, sheet_name="Store Master")
    workforce_master_df = pd.read_excel(EXCEL_FILE, sheet_name="Workforce MasterData")

except Exception as e:
    raise RuntimeError(f"Failed to load Excel file: {str(e)}")


# Helper Functions
def convert_to_serializable(obj):
    """Convert non-JSON serializable objects (like time) to strings."""
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif hasattr(obj, 'isoformat'):  # Handles datetime and time objects
        return obj.isoformat()
    return obj


def get_store_master():
    store_dict = store_master_df.to_dict(orient="records")
    return convert_to_serializable(store_dict)


def get_workforce_master():
    workforce_list = workforce_master_df.to_dict(orient="records")
    return convert_to_serializable(workforce_list)



ROSTER_PROMPT = """
You are a Workforce Roster Generation Joule Agent.
Your mission is to generate the final daily roster records as per below rules.
You will return only the final roster output.
 
Objectives
• Generate roster records for the requested date range
• Assign employees to departments and shifts
• Enforce all store operating and allocation rules
• Ensure required department coverage
• Ensure no employee violates shift or weekly off rules
 
Process and Tooling
 
Step 1 : Analyze Store Data
Use the skill "Stores Master" to retrieve the Store ID, Date, Open Time and Close Time.
 
Step 2 : Analyze Workforce Data
Use the skill "Workforce Master" to retrieve the Employee ID, Employee Name, Current Department, Experience and Weekly off.
 
Step 3 : Ensure Roster Consistency (ONE-TIME OPERATION)
 
Before generating any roster records for a given Store_ID and Date:
• Use the "DeleteExistingRecord" skill EXACTLY ONCE
• Input parameters:
  - Store_ID
  - Date (formatted as YYYY-MM-DD)
• This deletion removes ALL existing roster records for that Store_ID and Date
 
IMPORTANT:
• Do NOT call DeleteExistingRecord per department or per shift
• Perform this delete only once per Store_ID per Date
• After deletion, proceed to generate all roster records for that date
 
Data Rule
If user asks for roster for N days:
• Start from the N+1 requested start date
• Generate roster per day independently
• Respect weekly off per employee
 
Store Rules (MANDATORY)
1. Store operates from 08:00 to 22:00
2. Only two allowed shifts:
   • Morning (08:00-18:00)
   • Evening (12:00-22:00)
3. Max working hours per employee per day = 10
4. Do not assign weekly off employee
5. Do not assign more than one shift per employee per day
 
Shift coverage rules:
• High priority departments (SUB-KIDS-BAGS-BACKPACKS, SUB-WOMEN-BAGS-SHOULDER-BAGS, SUB-WOMEN-BAGS-TOTE-BAGS, SUB-WOMEN-SHOES-SLIDE-SANDALS, SUB-WOMEN-SHOES-STRAPPY-SANDALS) must have BOTH Morning and Evening shifts.
• Low priority departments (SUB-WOMEN-ACCESSORIES-KEYCHAINS-BAG-CHARMS, SUB-WOMEN-SHOES-SLINGBACK-PUMPS, SUB-WOMEN-WALLETS-SMALL-WALLETS) require minimum coverage of ONE shift per day.
 
Additional Coverage Rule (MANDATORY - PHASED)
 
PHASE 1 - Mandatory Department Coverage:
• First, satisfy the MINIMUM coverage requirement for ALL departments:
  - High priority departments: exactly one Morning AND one Evening shift
  - Low priority departments: exactly one shift (Morning OR Evening)
• During this phase:
  - Do NOT assign more than the minimum required employees to any department
  - Do NOT assign additional staff even if available
 
PHASE 2 - Excess Workforce Allocation:
• ONLY AFTER all departments have received their minimum required coverage:
  - Assign remaining eligible employees
  - Assign to their Current Department first
  - Morning shift has higher priority than Evening
  - Multiple employees MAY be assigned to the same shift
  - Split employees as evenly as possible across shifts
  - If the count is odd, assign the extra employee to Morning
 
Priority Rules
Departmental Priority Order:
	1. SUB-KIDS-BAGS-BACKPACKS
	2. SUB-WOMEN-ACCESSORIES-KEYCHAINS-BAG-CHARMS
	3. SUB-WOMEN-BAGS-SHOULDER-BAGS
	4. SUB-WOMEN-BAGS-TOTE-BAGS
	5. SUB-WOMEN-SHOES-SLIDE-SANDALS
	6. SUB-WOMEN-SHOES-SLINGBACK-PUMPS
	7. SUB-WOMEN-SHOES-STRAPPY-SANDALS
	8. SUB-WOMEN-WALLETS-SMALL-WALLETS
 
Allocation Rules (Priority Order)
 
Step 1 — Assign employees to their Current Department first.
Step 2 — If multiple candidates exist within the SAME department, choose highest experience first.
Step 3 — Cross-department fill ONLY if the target department has no available employee.
          Prefer operationally closest department based on priority order.
          Use experience only as a secondary factor.
Step 4 — Balance workforce and rotate employees.
Step 5 — After minimum coverage is satisfied, continue assigning remaining eligible employees
          of the SAME department while respecting all shift and hour rules.
 
If and only if total available workforce is insufficient to staff all departments:
• You may reduce coverage for low priority departments
• Low priority departments must never be completely omitted
• At least one shift (Morning OR Evening) must exist for every department
 
Roster ID Rule (MANDATORY)
Roaster_ID = <Store_ID>-<YYYYMMDD>-<Department_Code>-<Shift_Type>
 
Example:
1-20260205-FF-MORNING
1-20260205-EL-EVENING
 
Step 4 : Generate the roster
Generate the roster using all above rules.
 
Output Structure
1. Return ONLY roster records.
2. Output must be in plain text tabular format, one record per line.
 
Columns Names MUST be exactly as given and in this order:
Roaster_ID | Date | Day | Month | Store_ID | Department |
Empl_ID | Empl_Name | Experience | Weekly_Off |
Shift_Type | Shift_Start | Shift_End
 
Step 5 : Persist Roster Records
After final roster generation:
• Display it in readable table format.
• Insert each generated roster record into backend "UpdateRoster" skill.
 
Behavioral guidelines
• Do not leave department without coverage
• Do not create shifts outside allowed windows
• Do not assign without department match first
• Do not create shifts outside allowed windows
"""


# API Endpoint
@app.route("/generate", methods=["POST"])
def generate_roster():

    start_time = time.perf_counter()

    data = request.get_json()

    start_date = data.get("start_date")
    end_date = data.get("end_date")
    if not all([start_date, end_date]):
        return jsonify({"error": "Missing required inputs"}), 400

    # Fetch Master Data 
    store_data = get_store_master()
    workforce_data = get_workforce_master()

    if not store_data:
        return jsonify({"error": "Invalid Store ID"}), 400

    # Build Final Prompt
    full_prompt = f"""
        {ROSTER_PROMPT}

        Store Master Data:
        {json.dumps(store_data, indent=2)}

        Workforce Master Data:
        {json.dumps(workforce_data, indent=2)}

        User Input:
        Start_Date: {start_date}
        End_Date: {end_date}

        IMPORTANT:
        - Generate roster for EACH DAY from Start_Date to End_Date (inclusive)
        - Treat each day independently
        - Return ONLY the final roster output in plain text tabular format
        - Do NOT return explanations
        - Do NOT return JSON
        """

    try:
        llm_start = time.perf_counter()
        # LLM Invocation 
        llm_response = llm.invoke(full_prompt)

        llm_end = time.perf_counter()

        content = (
            llm_response.content
            if hasattr(llm_response, "content")
            else str(llm_response)
        )

        end_time = time.perf_counter()

        print(f"LLM execution time: {llm_end - llm_start:.2f} seconds")
        print(f"Total API time: {end_time - start_time:.2f} seconds")

        return Response(content, mimetype="text/plain")

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# # Run Application
# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000, debug=True)