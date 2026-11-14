import copy
import os
import unittest
import requests
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from zoneinfo import ZoneInfo

from fastapi.testclient import TestClient
import app
import mandi_public


class PublicMandiTests(unittest.TestCase):
    def setUp(self):
        for fn in (mandi_public._filters, mandi_public._markets, mandi_public._prices):
            fn.cache_clear()
        app.reset_query_rate_limits()
        self.date = (datetime.now(ZoneInfo("Asia/Kolkata")).date() - timedelta(days=2)).isoformat()
        self.directory = {
            "cmdt_data": [{"cmdt_id": 13, "cmdt_name": "Soyabean"}],
            "state_data": [{"state_id": 19, "state_name": "Madhya Pradesh"}, {"state_id": 20, "state_name": "Maharashtra"}],
            "district_data": [{"id": 308, "district_name": "Indore", "state_id": 19}],
            "market_data": [{"id": 3046, "mkt_name": "Indore APMC", "state_id": 19, "district_id": 308}],
        }
        self.markets = [{"market_id": 3046, "market_name": "Indore APMC", "state_id": 19, "state_name": "Madhya Pradesh"}]
        self.report = {"success": True, "data": [{"variety": "Soyabean - Soyabeen", self.date: 7100, "unitOfPrice": "Rs./Quintal"}]}
        self.calls = []

    def fetch(self, path, params=None):
        self.calls.append((path, params))
        if path == "/daily-price-arrival/filters":
            return {"status": True, "data": copy.deepcopy(self.directory)}
        if path == "/list-comm/13":
            return {"data": copy.deepcopy(self.markets)}
        if path.endswith("/lastweek"):
            self.assertEqual(params["marketId"], 3046)
            self.assertEqual(params["stateId"], 19)
            self.assertEqual(params["commodityId"], 13)
            return copy.deepcopy(self.report)
        self.fail("Unexpected official endpoint")

    def test_crop_only_asks_market_then_followup_returns_exact_dated_report(self):
        with patch.dict(os.environ, {"MANDI_PRIMARY_PROVIDER": "agmarknet_public", "DATA_GOV_IN_API_KEY": "private-key"}), patch("mandi_public._get", side_effect=self.fetch), patch("live_data._request_json") as legacy, patch("app.get_chain") as chain:
            client = TestClient(app.app)
            first = client.post("/query", json={"question": "Aaj soybean ka mandi bhav kya hai?"}).json()
            self.assertEqual(first["live_status"], "needs_more_input")
            self.assertFalse(first["evidence_verified"])
            self.assertEqual(first["workflow_state"], "awaiting_input")
            second = client.post("/query", json={"question": "Indore APMC, Madhya Pradesh", "workflow_context": first["workflow_context"]}).json()
            self.assertEqual(second["live_status"], "success")
            self.assertTrue(second["evidence_verified"])
            self.assertEqual(second["live_data"]["commodity"], "Soyabean")
            self.assertEqual(second["live_data"]["reported_price"], 7100)
            self.assertEqual(second["live_data"]["arrival_date"], self.date)
            self.assertFalse(second["live_data"]["is_today"])
            self.assertIn(self.date, second["answer"])
            self.assertIn("₹7,100/quintal", second["answer"])
            self.assertIn("aaj ka live quote nahi", second["answer"])
            self.assertNotIn("private-key", str(second))
            legacy.assert_not_called()
            chain.assert_not_called()

    def test_data_gov_timeout_falls_back_to_official_report(self):
        from live_data import get_mandi_price_snapshot
        with patch.dict(os.environ, {"MANDI_PRIMARY_PROVIDER": "data_gov_in", "DATA_GOV_IN_API_KEY": "private-key"}), patch("live_data._request_json", side_effect=requests.Timeout()), patch("mandi_public._get", side_effect=self.fetch):
            result = get_mandi_price_snapshot("Indore soybean mandi price")
        self.assertEqual(result["live_status"], "success")
        self.assertEqual(result["data_provider"], "agmarknet.gov.in")
        self.assertEqual(result["live_data"]["reported_price"], 7100)

    def test_ambiguous_market_requires_state_instead_of_guessing(self):
        self.markets.append({"market_id": 9999, "market_name": "Indore APMC", "state_id": 20, "state_name": "Maharashtra"})
        with patch("mandi_public._get", side_effect=self.fetch):
            result = mandi_public.lookup("Soyabean", "Indore soybean price")
        self.assertEqual(result["status"], "needs_more_input")
        self.assertFalse(any(path.endswith("/lastweek") for path, _ in self.calls))

    def test_wrong_state_does_not_select_another_states_market(self):
        with patch("mandi_public._get", side_effect=self.fetch):
            result = mandi_public.lookup("Soyabean", "Indore soybean price", state="Maharashtra", market="Indore")
        self.assertEqual(result["status"], "needs_more_input")
        self.assertFalse(any(path.endswith("/lastweek") for path, _ in self.calls))

    def test_unreported_zero_nonfinite_and_future_prices_are_rejected(self):
        today = datetime.now(ZoneInfo("Asia/Kolkata")).date()
        self.report["data"] = [{"variety": "Soyabean", "unitOfPrice": "Rs./Quintal", self.date: "NR", (today - timedelta(days=1)).isoformat(): 0, today.isoformat(): float("nan"), (today + timedelta(days=1)).isoformat(): 9000}]
        with patch("mandi_public._get", side_effect=self.fetch):
            self.assertEqual(mandi_public.lookup("Soyabean", "Indore soybean price")["status"], "unavailable_no_records")

    def test_old_report_is_marked_stale(self):
        date = (datetime.now(ZoneInfo("Asia/Kolkata")).date() - timedelta(days=10)).isoformat()
        self.report["data"] = [{"variety": "Soyabean", "unitOfPrice": "Rs./Quintal", date: 7100}]
        with patch("mandi_public._get", side_effect=self.fetch):
            self.assertEqual(mandi_public.lookup("Soyabean", "Indore soybean price")["status"], "stale")

    def test_reports_are_cached_without_changing_observation_date(self):
        with patch("mandi_public._get", side_effect=self.fetch):
            one = mandi_public.lookup("Soyabean", "Indore soybean price")
            two = mandi_public.lookup("Soyabean", "Indore soybean price")
        self.assertEqual(one["arrival_date"], two["arrival_date"])
        self.assertEqual(len(self.calls), 3)

    def test_public_request_never_receives_private_data_gov_key(self):
        response = Mock()
        response.json.return_value = {"data": []}
        with patch.dict(os.environ, {"DATA_GOV_IN_API_KEY": "private-key"}), patch("requests.get", return_value=response) as get:
            mandi_public._get("/list-comm/13")
        self.assertNotIn("private-key", str(get.call_args))
        self.assertEqual(get.call_args.kwargs["timeout"], (3, 10))

    def test_failed_live_lookup_is_not_verified_or_synthesized(self):
        dynamic = {"mode": "dynamic_router", "route": "dynamic_router", "route_reason": "mandi_price_live_data", "answer": "Price service unavailable.", "sources": [{"category": "market_prices", "url": "https://agmarknet.gov.in/"}], "live_status": "unavailable_api_error"}
        with patch("app.route_dynamic_query", return_value=dynamic), patch("app.get_chain") as chain:
            result = TestClient(app.app).post("/query", json={"question": "Indore soybean mandi price"}).json()
        self.assertFalse(result["evidence_verified"])
        self.assertEqual(result["evidence_verifier"]["status"], "unavailable")
        chain.assert_not_called()
