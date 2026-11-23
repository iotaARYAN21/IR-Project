import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.indexing.config import load_env_file


class LoadEnvFileTests(unittest.TestCase):
    def write(self, text):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / ".env"
        path.write_text(text)
        return path

    def test_reads_plain_quoted_and_exported_values(self):
        path = self.write(
            '# comment\n\nA_PLAIN=one\nA_QUOTED="two words"\n'
            "export A_EXPORTED='three'\nnot a variable line\n"
        )
        with mock.patch.dict(os.environ, {}, clear=False):
            for name in ("A_PLAIN", "A_QUOTED", "A_EXPORTED"):
                os.environ.pop(name, None)
            load_env_file(path)
            self.assertEqual(os.environ["A_PLAIN"], "one")
            self.assertEqual(os.environ["A_QUOTED"], "two words")
            self.assertEqual(os.environ["A_EXPORTED"], "three")

    def test_existing_environment_wins(self):
        path = self.write("A_KEEP=from-file\n")
        with mock.patch.dict(os.environ, {"A_KEEP": "from-shell"}):
            load_env_file(path)
            self.assertEqual(os.environ["A_KEEP"], "from-shell")

    def test_missing_file_is_ignored(self):
        load_env_file(Path("/nonexistent/.env"))


if __name__ == "__main__":
    unittest.main()
