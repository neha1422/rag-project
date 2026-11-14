import unittest
from unittest.mock import patch
from ocr_utils import check_ocr_dependencies

class OCRReadinessTests(unittest.TestCase):
    def test_wrapper_without_executable_is_unavailable(self):
        with patch("ocr_utils._module_available", return_value=True), patch("ocr_utils.shutil.which", return_value=None):
            status = check_ocr_dependencies()
        self.assertFalse(status["available"])
        self.assertEqual(status["engine"], "")

    def test_cli_without_wrapper_is_available(self):
        with patch("ocr_utils._module_available", side_effect=lambda name: name == "pypdfium2"), patch("ocr_utils.shutil.which", return_value="/usr/bin/tesseract"), patch("ocr_utils.installed_tesseract_languages", return_value=["eng"]):
            status = check_ocr_dependencies()
        self.assertTrue(status["available"])
        self.assertEqual(status["engine"], "tesseract_cli")
