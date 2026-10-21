"""
Local mock of the Warehouse CAPM OData v4 service + its XSUAA token endpoint.

Everything here is fictional demo data. It lets the app run end-to-end without
any real SAP BTP / HANA system:

    POST /oauth/token                     -> fake bearer token
    GET  /odata/v4/warehouse/<Entity>     -> seeded rows ($filter eq/ge/le, $top, $orderby)
    POST /odata/v4/warehouse/<Entity>     -> insert (in memory)
    PUT  /odata/v4/warehouse/<Entity>('k')-> update (in memory)
    POST /odata/v4/warehouse/$batch       -> JSON batch of inserts
    POST /odata/v4/warehouse/deleteRosterIfExists

Run standalone:  python mock_services/mock_odata_server.py
(It is also started automatically in a background thread by tools/auth.py when
MOCK_MODE=true.)
"""

import os
import re
import random
import threading
import uuid
from datetime import date, datetime, timedelta, timezone

from flask import Flask, jsonify, request

BASE = "/odata/v4/warehouse"
STORE_ID = "S001"

DEPARTMENTS = [
    "Handbags", "Backpacks", "Kids Line", "Accessories",
    "Women-Shoes", "Wallets & Small Leather Goods", "New Arrival", "Featured Collection",
]
PREFIX = {
    "Handbags": "HB", "Backpacks": "BP", "Kids Line": "KL", "Accessories": "AS",
    "Women-Shoes": "WS", "Wallets & Small Leather Goods": "WL",
    "New Arrival": "NA", "Featured Collection": "FC",
}
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

_FIRST = ["Aarav", "Maya", "Liam", "Sofia", "Noah", "Emma", "Ethan", "Olivia", "Lucas", "Mia",
          "Mason", "Ava", "Logan", "Isla", "James", "Zoe", "Henry", "Nora", "Owen", "Lena",
          "Caleb", "Ruby", "Isaac", "Chloe", "Leo", "Hana", "Jack", "Tara", "Finn", "Nina",
          "Adam", "Eva", "Ravi", "Kira", "Omar", "Lily", "Ben", "Anya", "Theo", "Sara"]
_LAST = ["Sample", "Demo", "Test", "Example"]


def _seed() -> dict:
    rnd = random.Random(42)  # deterministic demo data
    today = date.today()

    # ── Workforce: 40 employees, 5 per department ───────────────────────────
    workforce = []
    for i, name in enumerate(_FIRST):
        dept = DEPARTMENTS[i % len(DEPARTMENTS)]
        workforce.append({
            "EmplId": str(1001 + i),
            "EmplName": f"{name} {_LAST[i % len(_LAST)]}",
            "CurrDept": dept,
            "ExperienceYear": round(rnd.uniform(0.5, 12), 1),
            "WeeklyOff": WEEKDAYS[(i * 3 + i // 7) % 7],
        })

    # ── Store master: one store, 30 days of context ─────────────────────────
    stores = [{
        "StoreId": STORE_ID,
        "StoreDate": (today + timedelta(days=d)).isoformat(),
        "StoreName": "Demo Warehouse Store 001",
        "OpeningTime": "08:00",
        "ClosingTime": "22:00",
        "Region": "North",
        "Capacity": 120,
    } for d in range(-7, 23)]

    # ── Footfall: 8 departments x last 14 days ──────────────────────────────
    footfall = []
    for d in range(14, 0, -1):
        day = today - timedelta(days=d)
        for dept in DEPARTMENTS:
            ff = rnd.randint(400, 1500)
            conv = round(rnd.uniform(8, 38), 1)
            std = 22.0
            var = round(conv - std, 1)
            colour = "GREEN" if var >= 2 else ("YELLOW" if var >= -4 else "RED")
            footfall.append({
                "Department": dept,
                "TransactionDate": day.isoformat(),
                "ReceiptCount": int(ff * conv / 100),
                "FootFallCount": ff,
                "SalesConversionPct": conv,
                "IndustryStandardPct": std,
                "Variance": var,
                "Colour": colour,
                "PerformanceComparison": "Above Industry" if var >= 0 else "Below Industry",
            })

    # ── Articles / promotions / readiness ───────────────────────────────────
    nouns = {
        "Handbags": ["Tote Bag", "Crossbody Bag", "Clutch"],
        "Backpacks": ["Trail Backpack", "Laptop Backpack"],
        "Kids Line": ["Kids Raincoat", "Kids Sneakers"],
        "Accessories": ["Silk Scarf", "Leather Belt"],
        "Women-Shoes": ["Running Shoes", "Ankle Boots"],
        "Wallets & Small Leather Goods": ["Bifold Wallet", "Card Holder"],
        "New Arrival": ["Summer Hat", "Canvas Tote"],
        "Featured Collection": ["Signature Bag", "Limited Edition Watch"],
    }
    articles, promotions, readiness = [], [], []
    pn = 1
    for dept, items in nouns.items():
        for j, item in enumerate(items):
            aid = f"{PREFIX[dept]}-{101 + j}"
            articles.append({"ArticleId": aid, "ArticleDescription": f"Demo {item}", "Department": dept})
            if rnd.random() < 0.7:
                pid = f"PROMO-{pn:03d}"
                pn += 1
                start = today + timedelta(days=rnd.randint(-2, 3))
                promotions.append({
                    "PromotionId": pid, "ArticleId": aid,
                    "PromoDescription": rnd.choice(["Seasonal Sale", "Buy 1 Get 1", "Weekend Special", "Clearance"]),
                    "PromoStartDate": start.isoformat(),
                    "PromoEndDate": (start + timedelta(days=rnd.randint(3, 10))).isoformat(),
                })
                pct = rnd.randint(30, 100)
                readiness.append({
                    "ArticleId": aid, "Department": dept, "PromotionId": pid,
                    "ReadinessPct": pct,
                    "PricingStatus": "Yes" if pct > 55 else "No",
                    "LabellingStatus": "Green" if pct > 80 else ("Yellow" if pct > 55 else "Red"),
                    "ReadinessStatus": "Ready" if pct > 80 else ("Borderline" if pct > 55 else "Not Ready"),
                    "LabellingCompletionPct": min(100, pct + rnd.randint(-10, 10)),
                    "OrderToBePlaced": rnd.randint(-40, 150),
                })

    # ── Roster: -3 .. +10 days, Morning/Evening, honouring weekly off ───────
    roster = []
    for d in range(-3, 11):
        day = today + timedelta(days=d)
        dn = WEEKDAYS[day.weekday()]
        ymd = day.strftime("%Y%m%d")
        n = 0
        for emp in workforce:
            if emp["WeeklyOff"] == dn:
                continue
            n += 1
            roster.append({
                "RoasterId": f"R-{ymd}-{n:03d}",
                "RosterDate": day.isoformat(),
                "WeekDay": dn,
                "RosterMonth": day.strftime("%B"),
                "StoreId": STORE_ID,
                "Mch1": emp["CurrDept"],
                "Shift": "Morning (8AM-6PM)" if n % 2 else "Evening (12PM-10PM)",
                "EmplId": emp["EmplId"],
                "EmplName": emp["EmplName"],
                "Experience": emp["ExperienceYear"],
                "WeeklyOff": emp["WeeklyOff"],
            })

    # ── Exceptions (sick leave etc.) ────────────────────────────────────────
    exceptions = []
    for k, (emp, case) in enumerate(zip([workforce[0], workforce[9], workforce[18], workforce[1], workforce[2]], ["SICK_LEAVE", "SICK_LEAVE", "CASUAL_LEAVE", "SICK_LEAVE", "TRAINING"])):
        exceptions.append({
            "EmplId": emp["EmplId"], "EmplName": emp["EmplName"], "CurrentDept": emp["CurrDept"],
            "ExceptionCase": case,
            "ExceptionDate": (today + timedelta(days=k)).isoformat(),
            "DayOff": WEEKDAYS[(today + timedelta(days=k)).weekday()],
            "Delete_mc": "N", "Update_mc": "N",
        })

    return {
        "StoreMaster": stores,
        "WorkforceMaster": workforce,
        "FootfallMaster": footfall,
        "ArticleMaster": articles,
        "PromotionMaster": promotions,
        "PromotionReadinessAgent": readiness,
        "RosterMaster": roster,
        "PromotionRosterMaster": [],
        "ExceptionMaster": exceptions,
        "ExceptionRosterMaster": [],
        "StoreJobId": [{
            "ID": str(uuid.uuid4()), "jobId": "JOB-DEMO-0001",
            "createdAt": datetime.now(timezone.utc).isoformat(),
        }],
    }


DATA = _seed()
KEYS = {"RosterMaster": "RoasterId", "PromotionRosterMaster": "RoasterId",
        "ExceptionRosterMaster": "RoasterId", "StoreJobId": "ID"}

app = Flask(__name__)


# ── tiny $filter evaluator: "<Field> eq|ge|le|gt|lt|ne '<v>'" joined by and/or ──
_COND = re.compile(r"(\w+)\s+(eq|ne|ge|le|gt|lt)\s+(?:'((?:[^']|'')*)'|([\w.\-]+))")
_OPS = {"eq": lambda a, b: a == b, "ne": lambda a, b: a != b, "ge": lambda a, b: a >= b,
        "le": lambda a, b: a <= b, "gt": lambda a, b: a > b, "lt": lambda a, b: a < b}


def _match(row: dict, flt: str) -> bool:
    def one(expr: str) -> bool:
        m = _COND.search(expr)
        if not m:
            return True
        field, op = m.group(1), m.group(2)
        val = m.group(3).replace("''", "'") if m.group(3) is not None else m.group(4)
        return _OPS[op](str(row.get(field, "")), str(val))
    return any(all(one(part) for part in re.split(r"\s+and\s+", grp, flags=re.I))
               for grp in re.split(r"\s+or\s+", flt, flags=re.I))


@app.route("/oauth/token", methods=["POST"])
def token():
    return jsonify({"access_token": "mock-" + uuid.uuid4().hex, "token_type": "bearer",
                    "expires_in": 43199, "scope": "uaa.resource"})


@app.route(f"{BASE}/<entity>", methods=["GET"])
def get_entity(entity):
    if entity not in DATA:
        return jsonify({"error": {"code": "404", "message": f"Resource '{entity}' not found"}}), 404
    rows = list(DATA[entity])
    if request.args.get("$filter"):
        rows = [r for r in rows if _match(r, request.args["$filter"])]
    if request.args.get("$orderby"):
        parts = request.args["$orderby"].split()
        rows.sort(key=lambda r: str(r.get(parts[0], "")), reverse=len(parts) > 1 and parts[1] == "desc")
    if request.args.get("$top"):
        rows = rows[: int(request.args["$top"])]
    return jsonify({"@odata.context": f"$metadata#{entity}", "value": rows})


def _insert(entity: str, record: dict) -> dict:
    if entity not in DATA:
        raise KeyError(entity)
    record = dict(record)
    key = KEYS.get(entity)
    if entity == "StoreJobId":
        record.setdefault("ID", str(uuid.uuid4()))
        record.setdefault("createdAt", datetime.now(timezone.utc).isoformat())
    if key and any(r.get(key) == record.get(key) for r in DATA[entity]):
        raise ValueError("duplicate key")
    DATA[entity].append(record)
    return record


@app.route(f"{BASE}/<entity>", methods=["POST"])
def post_entity(entity):
    if entity == "deleteRosterIfExists":
        body = request.get_json(silent=True) or {}
        lo, hi = body.get("fromDate", ""), body.get("toDate", "~")
        before = len(DATA["RosterMaster"])
        DATA["RosterMaster"] = [r for r in DATA["RosterMaster"] if not (lo <= r["RosterDate"] <= hi)]
        return jsonify({"value": f"Deleted {before - len(DATA['RosterMaster'])} rows"})
    try:
        return jsonify(_insert(entity, request.get_json(force=True))), 201
    except KeyError:
        return jsonify({"error": {"code": "404", "message": "Unknown entity"}}), 404
    except ValueError as e:
        return jsonify({"error": {"code": "409", "message": str(e)}}), 400


@app.route(f"{BASE}/<entity>('<key>')", methods=["PUT", "PATCH"])
def put_entity(entity, key):
    kf = KEYS.get(entity)
    for r in DATA.get(entity, []):
        if kf and r.get(kf) == key:
            r.update(request.get_json(force=True))
            return jsonify(r)
    return jsonify({"error": {"code": "404", "message": "Not found"}}), 404


@app.route(f"{BASE}/$batch", methods=["POST"])
def batch():
    out = []
    for item in (request.get_json(force=True) or {}).get("requests", []):
        try:
            body = _insert(item["url"].lstrip("/"), item.get("body", {}))
            out.append({"id": item["id"], "status": 201, "body": body})
        except Exception as e:  # noqa: BLE001
            out.append({"id": item["id"], "status": 400, "body": {"error": {"message": str(e)}}})
    return jsonify({"responses": out})


@app.route("/health")
def health():
    return jsonify({"status": "ok", "mock": True})


def start_in_background(port: int) -> None:
    """Start the mock server on 127.0.0.1:<port> in a daemon thread (idempotent)."""
    import socket
    with socket.socket() as s:
        if s.connect_ex(("127.0.0.1", port)) == 0:
            return  # something already listening
    import logging
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    threading.Thread(
        target=lambda: app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False),
        daemon=True, name="mock-odata").start()
    import time
    time.sleep(0.5)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.getenv("MOCK_PORT", "5055")))
