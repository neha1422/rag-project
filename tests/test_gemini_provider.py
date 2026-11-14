import os
import unittest
from unittest.mock import Mock, patch
from rag_chain import _call_gemini

class GeminiProviderTests(unittest.TestCase):
    def test_generates_with_header_auth_and_existing_dependencies(self):
        response = Mock()
        response.json.return_value = {"candidates": [{"content": {"parts": [{"text": "internal", "thought": True}, {"text": "Grounded answer"}]}}]}
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key", "GEMINI_MODEL": "gemini-2.5-flash-lite"}), patch("requests.post", return_value=response) as post:
            self.assertEqual(_call_gemini("Verified context"), "Grounded answer")
        args, kwargs = post.call_args
        self.assertNotIn("test-key", args[0])
        self.assertEqual(kwargs["headers"]["x-goog-api-key"], "test-key")
        self.assertEqual(kwargs["json"]["contents"][0]["parts"][0]["text"], "Verified context")

    def test_empty_or_blocked_generation_is_not_a_valid_answer(self):
        response = Mock()
        response.json.return_value = {"candidates": []}
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}), patch("requests.post", return_value=response):
            with self.assertRaisesRegex(RuntimeError, "empty response"):
                _call_gemini("Verified context")
