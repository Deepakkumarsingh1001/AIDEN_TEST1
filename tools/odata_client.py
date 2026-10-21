"""
Generic OData v4 verbs for the Warehouse CAPM service.

Every call resolves auth + base URL through the cached token manager
(tools.auth.get_auth), so callers never deal with tokens directly.

    odata_get(entity, params=None)   -> list   (the OData `value` array)
    odata_get_raw(entity, params)    -> dict    (full response body)
    odata_post(entity, payload)      -> dict
    odata_batch(entity, records)     -> dict    (OData v4 $batch envelope)
    odata_post_custom(path, payload) -> dict    (non-entity endpoints, e.g. deleteRosterIfExists)

Notes
-----
- $ is kept literal in query keys — SAP OData rejects %24-encoded option names.
- verify=False mirrors the existing project behaviour (local SSL interception).
"""

import os
from urllib.parse import quote

import requests
import urllib3

from tools.auth import get_auth

# Master tables (e.g. StoreMaster keyed by StoreId+StoreDate) can be large, so the
# original code used no timeout. Use a generous, configurable default instead of a
# short one that would abort big reads. Override with ODATA_TIMEOUT (seconds).
DEFAULT_TIMEOUT = int(os.getenv("ODATA_TIMEOUT", "120"))

# verify=False is intentional for this project; silence the per-call warning spam.
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def _qs(params: dict) -> str:
    """Build a query string while keeping `$option` keys literal."""
    return "&".join(f"{k}={quote(str(v), safe='')}" for k, v in params.items())


def _url(base_url: str, entity: str, params: dict | None) -> str:
    url = base_url + entity.lstrip("/")
    if params:
        url = f"{url}?{_qs(params)}"
    return url


def odata_get_raw(entity: str, params: dict | None = None) -> dict:
    base_url, bearer = get_auth()
    url = _url(base_url, entity, params)
    print(f"[ODATA] GET {url}")
    resp = requests.get(url, headers={"Authorization": bearer, "Accept": "application/json"},
                        verify=False, timeout=DEFAULT_TIMEOUT)
    if not resp.ok:
        print(f"[ODATA] GET failed status={resp.status_code} body={resp.text[:500]}")
    resp.raise_for_status()
    return resp.json()


def odata_get(entity: str, params: dict | None = None) -> list:
    """GET an entity set and return its `value` array (or [] / the raw body)."""
    data = odata_get_raw(entity, params)
    if isinstance(data, dict) and "value" in data:
        return data["value"]
    return data if isinstance(data, list) else [data]


def odata_post(entity: str, payload: dict) -> dict:
    base_url, bearer = get_auth()
    url = base_url + entity.lstrip("/")
    print(f"[ODATA] POST {url}")
    resp = requests.post(
        url,
        headers={"Authorization": bearer, "Accept": "application/json",
                 "Content-Type": "application/json"},
        json=payload, verify=False, timeout=DEFAULT_TIMEOUT,
    )
    if not resp.ok:
        print(f"[ODATA] POST failed status={resp.status_code} body={resp.text[:1000]}")
    resp.raise_for_status()
    return resp.json()


def odata_batch(entity: str, records: list) -> dict:
    """POST many records to `entity` in a single OData v4 JSON $batch request."""
    base_url, bearer = get_auth()
    url = base_url + "$batch"
    batch_requests = [
        {
            "id": str(i),
            "method": "POST",
            "url": entity,
            "headers": {"content-type": "application/json"},
            "body": record,
        }
        for i, record in enumerate(records)
    ]
    print(f"[ODATA] BATCH {url} ({len(records)} records -> {entity})")
    resp = requests.post(
        url,
        headers={"Authorization": bearer, "Accept": "application/json",
                 "Content-Type": "application/json"},
        json={"requests": batch_requests}, verify=False, timeout=DEFAULT_TIMEOUT,
    )
    if not resp.ok:
        print(f"[ODATA] BATCH failed status={resp.status_code} body={resp.text[:1000]}")
    resp.raise_for_status()

    # A $batch returns HTTP 200 even when individual sub-requests fail, so the
    # overall status is NOT enough — inspect each sub-response and surface errors
    # (otherwise a failed insert looks like success and nothing reaches the table).
    data = resp.json()
    responses = data.get("responses", []) if isinstance(data, dict) else []
    failures = [r for r in responses
                if isinstance(r.get("status"), int) and r["status"] >= 400]
    if failures:
        sample = failures[0]
        print(f"[ODATA] BATCH sub-request failures: {len(failures)}/{len(responses)}")
        print(f"[ODATA] first failure: status={sample.get('status')} body={str(sample.get('body'))[:1000]}")
        raise RuntimeError(
            f"OData $batch into {entity}: {len(failures)}/{len(responses)} rows rejected. "
            f"First failure status={sample.get('status')}, body={str(sample.get('body'))[:500]}"
        )
    print(f"[ODATA] BATCH ok: {len(responses)} sub-requests succeeded")
    return data


def odata_upsert(entity: str, payload: dict, key_value: str) -> dict:
    """Create (POST) or update (PUT) a single record — safe to call on retry.

    If POST returns 4xx/5xx (duplicate key or validation failure), falls back to
    PUT on the keyed path so the row is updated rather than erroring.
    """
    try:
        return odata_post(entity, payload)
    except requests.HTTPError as e:
        status = e.response.status_code if e.response is not None else 0
        if status in (400, 409, 500):
            base_url, bearer = get_auth()
            safe_key = str(key_value).replace("'", "''")
            url = f"{base_url}{entity.lstrip('/')}('{safe_key}')"
            print(f"[ODATA] POST failed ({status}), trying PUT {url}")
            resp = requests.put(
                url,
                headers={
                    "Authorization": bearer,
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                },
                json=payload,
                verify=False,
                timeout=DEFAULT_TIMEOUT,
            )
            if not resp.ok:
                print(f"[ODATA] PUT failed status={resp.status_code} body={resp.text[:1000]}")
            resp.raise_for_status()
            return resp.json() if resp.text.strip() else {}
        raise


def odata_post_custom(path: str, payload: dict) -> dict:
    """POST to a non-entity endpoint (custom CAP action/function) under the base URL."""
    base_url, bearer = get_auth()
    url = base_url + path.lstrip("/")
    print(f"[ODATA] POST(custom) {url}")
    resp = requests.post(
        url,
        headers={"Authorization": bearer, "Accept": "application/json",
                 "Content-Type": "application/json"},
        json=payload, verify=False, timeout=DEFAULT_TIMEOUT,
    )
    if not resp.ok:
        print(f"[ODATA] POST(custom) failed status={resp.status_code} body={resp.text[:1000]}")
    resp.raise_for_status()
    # Some custom endpoints return empty bodies.
    try:
        return resp.json()
    except ValueError:
        return {"status": resp.status_code}
