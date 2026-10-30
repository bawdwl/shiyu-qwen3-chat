import tempfile
import unittest
from pathlib import Path

from secret_store import LocalSecretStore


class LocalSecretStoreTests(unittest.TestCase):
    def test_round_trip_and_delete_without_plaintext_on_disk(self):
        with tempfile.TemporaryDirectory() as temp:
            store = LocalSecretStore(Path(temp) / "secrets.json")
            store.set("deepseek_api_key", "sk-very-secret")
            self.assertTrue(store.has("deepseek_api_key"))
            self.assertEqual(store.get("deepseek_api_key"), "sk-very-secret")
            self.assertNotIn(
                "sk-very-secret",
                (Path(temp) / "secrets.json").read_text(encoding="utf-8"),
            )
            self.assertTrue(store.delete("deepseek_api_key"))
            self.assertFalse(store.has("deepseek_api_key"))

    def test_empty_values_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            store = LocalSecretStore(Path(temp) / "secrets.json")
            with self.assertRaises(ValueError):
                store.set("", "x")
            with self.assertRaises(ValueError):
                store.set("key", "")

    def test_key_survives_new_store_instance_and_process_context(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "secrets.json"
            LocalSecretStore(path).set("deepseek_api_key", "sk-persistent")
            reopened = LocalSecretStore(path)
            self.assertEqual(reopened.get("deepseek_api_key"), "sk-persistent")
            self.assertTrue((Path(temp) / "local-secret.key").is_file())


if __name__ == "__main__":
    unittest.main()
