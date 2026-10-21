"""
Cached OAuth token manager for the Warehouse CAPM OData service.

Why this exists
---------------
BTP_Service.getDestination() does a fresh OAuth round-trip on EVERY OData call.
This module fetches the token once and reuses it until it is about to expire,
re-fetching only then.

Public API
----------
    get_auth() -> (base_url: str, bearer_header: str)
        base_url     : OData service base URL, guaranteed to end with "/"
        bearer_header: "Bearer <access_token>" — ready for the Authorization header

Two modes (auto-detected):
    Local      — no VCAP_SERVICES: POST clientid/clientsecret to {token_url}/oauth/token,
                 honour the returned `expires_in`. Base URL from WAREHOUSE_ODATA_URL.
    CloudFoundry — VCAP_SERVICES present: delegate to BTP_Service.getDestination
                 (destination service); cache with DEFAULT_TTL since that path does
                 not expose expires_in.
"""

import os
import time
import threading

import requests
from dotenv import load_dotenv

import BTP_Service

load_dotenv()

# MOCK_MODE=true -> start the local mock OData/OAuth server (mock_services/) and use it.
_MOCK = os.getenv("MOCK_MODE", "true").lower() == "true"
_MOCK_PORT = int(os.getenv("MOCK_PORT", "5055"))
if _MOCK:
    from mock_services.mock_odata_server import start_in_background
    start_in_background(_MOCK_PORT)

# Same default as BTP_Service.py so local dev resolves the warehouse service.
_DEFAULT_BASE_URL = "https://warehouse-demo.example.com/odata/v4/warehouse/"
_DEFAULT_TTL = 3600          # seconds, used when expires_in is unavailable (CF path)
_SAFETY_WINDOW = 60          # re-fetch this many seconds before actual expiry
_CF_DESTINATION_NAME = os.getenv("WAREHOUSE_DESTINATION_NAME", "warehouse")
_CF_DESTINATION_SERVICE = os.getenv("DESTINATION_SERVICE", "destination")

_lock = threading.Lock()
_cache = {"bearer": None, "base_url": None, "expires_at": 0.0}


def _normalize(base_url: str) -> str:
    base_url = base_url or _DEFAULT_BASE_URL
    return base_url if base_url.endswith("/") else base_url + "/"


def _fetch_token() -> tuple[str, str, float]:
    """Return (base_url, bearer_header, ttl_seconds). Mode auto-detected."""
    # ── Cloud Foundry: a destination service is bound ──────────────────────────
    if os.getenv("VCAP_SERVICES") and not _MOCK:
        dest = BTP_Service.getDestination(_CF_DESTINATION_SERVICE, _CF_DESTINATION_NAME)
        bearer = str(dest["authTokens"][0]["http_header"]["value"])
        base_url = _normalize(dest["destinationConfiguration"].get("URL"))
        print(f"[AUTH] token fetched (CF destination) ttl={_DEFAULT_TTL}s")
        return base_url, bearer, _DEFAULT_TTL

    # ── Local dev: client-credentials grant against the XSUAA token URL ─────────
    client_id = os.getenv("clientid")
    client_secret = os.getenv("clientsecret")
    token_url = os.getenv("token_url")
    if not (client_id and client_secret and token_url):
        raise RuntimeError(
            "Missing local OAuth credentials. Set clientid, clientsecret, token_url in .env"
        )

    resp = requests.post(
        token_url.rstrip("/") + "/oauth/token",
        data={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=15,
    )
    resp.raise_for_status()
    body = resp.json()
    access_token = body["access_token"]
    ttl = float(body.get("expires_in", _DEFAULT_TTL))
    base_url = _normalize(os.getenv("WAREHOUSE_ODATA_URL", _DEFAULT_BASE_URL))
    print(f"[AUTH] token fetched (local oauth) ttl={int(ttl)}s")
    return base_url, f"Bearer {access_token}", ttl


def get_auth() -> tuple[str, str]:
    """Return (base_url, bearer_header), reusing the cached token until near expiry."""
    now = time.time()
    with _lock:
        if _cache["bearer"] and now < _cache["expires_at"] - _SAFETY_WINDOW:
            print("[AUTH] token cache hit")
            return _cache["base_url"], _cache["bearer"]

        base_url, bearer, ttl = _fetch_token()
        _cache.update(bearer=bearer, base_url=base_url, expires_at=now + ttl)
        return base_url, bearer


def invalidate() -> None:
    """Force the next get_auth() to re-fetch (e.g. after a 401)."""
    with _lock:
        _cache.update(bearer=None, base_url=None, expires_at=0.0)
