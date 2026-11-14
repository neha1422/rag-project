"""Read-only fallback to the public reports used by the official AGMARKNET portal.

No Data.gov.in key is sent to this service. Public reports must identify a market,
commodity, observation date and unit before a price can be shown.
"""
from __future__ import annotations

import math
import re
import time
from datetime import datetime
from functools import lru_cache
from zoneinfo import ZoneInfo

import requests

BASE = "https://api.agmarknet.gov.in/v1"
PORTAL = "https://agmarknet.gov.in/"
HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Origin": PORTAL.rstrip("/"),
    "Referer": PORTAL,
    "User-Agent": "KrishiNyayAI/0.4 (+https://github.com/neha1422/rag-project)",
}


def _get(path: str, params: dict | None = None) -> dict:
    response = requests.get(BASE + path, params=params, headers=HEADERS, timeout=(3, 10))
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("Invalid official market report")
    return payload


@lru_cache(maxsize=4)
def _filters(hour: int) -> dict:
    payload = _get("/daily-price-arrival/filters")
    if payload.get("status") is not True or not isinstance(payload.get("data"), dict):
        raise ValueError("Official commodity directory unavailable")
    return payload["data"]


@lru_cache(maxsize=64)
def _markets(commodity_id: int, hour: int) -> list[dict]:
    rows = _get(f"/list-comm/{commodity_id}").get("data")
    if not isinstance(rows, list):
        raise ValueError("Official market directory unavailable")
    return rows


@lru_cache(maxsize=128)
def _prices(commodity_id: int, state_id: int, market_id: int, bucket: int) -> dict:
    payload = _get("/prices-and-arrivals/commodity-price/lastweek", {
        "commodityId": commodity_id, "stateId": state_id,
        "marketId": market_id, "includeExcel": "false",
    })
    if payload.get("success") is not True or not isinstance(payload.get("data"), list):
        raise ValueError("Official price report unavailable")
    return payload


def _normal(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", value.lower())).strip()


def _market_name(value: str) -> str:
    return _normal(re.sub(r"\b(apmc|mandi)\b", "", _normal(value)))


def _contains(text: str, value: str) -> bool:
    return bool(value) and bool(re.search(r"\b" + re.escape(value) + r"\b", text))


def lookup(commodity: str, question: str, *, state: str | None = None,
           district: str | None = None, market: str | None = None) -> dict:
    hour = int(time.time() // 3600)
    directory = _filters(hour)
    commodity_rows = [r for r in directory.get("cmdt_data", [])
                      if _normal(r.get("cmdt_name", "")) == _normal(commodity)]
    if len(commodity_rows) != 1:
        return {"status": "unavailable_no_records"}
    commodity_id = int(commodity_rows[0]["cmdt_id"])
    candidates = _markets(commodity_id, hour)
    text = _normal(question)
    state_ids = {int(r["state_id"]) for r in directory.get("state_data", [])
                 if (_normal(r.get("state_name", "")) == _normal(state or "")
                     if state else _contains(text, _normal(r.get("state_name", ""))))}
    if state and not state_ids:
        return {"status": "needs_more_input", "suggestions": []}
    if state_ids:
        candidates = [r for r in candidates if int(r["state_id"]) in state_ids]
    if district:
        district_ids = {int(r["id"]) for r in directory.get("district_data", [])
                        if _normal(r.get("district_name", "")) == _normal(district)
                        and (not state_ids or int(r["state_id"]) in state_ids)}
        market_ids = {int(r["id"]) for r in directory.get("market_data", [])
                      if r.get("district_id") in district_ids}
        candidates = [r for r in candidates if int(r["market_id"]) in market_ids]
    matches = [r for r in candidates
               if (_contains(_market_name(market), _market_name(r.get("market_name", "")))
                   if market else _contains(text, _market_name(r.get("market_name", ""))))]
    if matches:
        longest = max(len(_market_name(r.get("market_name", ""))) for r in matches)
        matches = [r for r in matches if len(_market_name(r.get("market_name", ""))) == longest]
    # Deduplicate directory rows, but never choose an arbitrary market/state.
    matches = list({int(r["market_id"]): r for r in matches}.values())
    if len(matches) != 1:
        suggestions = sorted({f"{r['market_name']}, {r['state_name']}" for r in (matches or candidates)})[:5] if state_ids or market else []
        return {"status": "needs_more_input", "suggestions": suggestions}
    selected = matches[0]
    report = _prices(commodity_id, int(selected["state_id"]), int(selected["market_id"]), int(time.time() // 300))
    today = datetime.now(ZoneInfo("Asia/Kolkata")).date()
    observations = []
    for row in report["data"]:
        unit = str(row.get("unitOfPrice", "")).strip()
        if not unit:
            continue
        for key, value in row.items():
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", key):
                continue
            if isinstance(value, bool):
                continue
            try:
                date = datetime.strptime(key, "%Y-%m-%d").date()
                price = float(value)
            except (ValueError, TypeError):
                continue
            if not math.isfinite(price) or price <= 0 or date > today:
                continue
            observations.append((date, str(row.get("variety", "")), price, unit))
    if not observations:
        return {"status": "unavailable_no_records"}
    # Stable variety selection; retain the original variety and report date.
    date, variety, price, unit = sorted(observations, key=lambda x: (-x[0].toordinal(), x[1]))[0]
    age = (today - date).days
    return {
        "status": "success" if age <= 7 else "stale",
        "commodity": commodity_rows[0]["cmdt_name"], "market": selected["market_name"],
        "state": selected["state_name"], "variety": variety,
        "reported_price": price, "price_unit": unit, "arrival_date": date.isoformat(),
        "age_days": age, "is_today": age == 0,
        "source_url": PORTAL, "report_url": BASE + "/prices-and-arrivals/commodity-price/lastweek",
        "records_returned": len(report["data"]),
    }
