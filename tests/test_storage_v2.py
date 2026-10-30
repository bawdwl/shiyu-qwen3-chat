import sqlite3
import tempfile
import unittest
from pathlib import Path

from storage import ChatStore


class ChatStoreV2Tests(unittest.TestCase):
    def test_role_conversation_is_reused_and_reset_clears_related_state(self):
        with tempfile.TemporaryDirectory() as temp:
            store = ChatStore(Path(temp) / "chat.sqlite3")
            first = store.get_or_create_role_conversation(
                "companion", model_id="deepseek-flash", provider="deepseek"
            )
            second = store.get_or_create_role_conversation(
                "companion", model_id="deepseek-flash", provider="deepseek"
            )
            self.assertEqual(first["id"], second["id"])
            self.assertEqual(first["provider"], "deepseek")
            store.save_turn(first["id"], "你好", "你好呀")
            store.set_setting("limits", {"request": 1000, "daily": 5000})
            store.set_setting("deepseek_configured", True)
            reset = store.clear_role_conversation("companion")
            self.assertEqual(reset["messages"], [])
            self.assertEqual(
                store.get_setting("limits"), {"request": 1000, "daily": 5000}
            )

    def test_usage_totals_and_idempotent_request_id(self):
        with tempfile.TemporaryDirectory() as temp:
            store = ChatStore(Path(temp) / "chat.sqlite3")
            store.record_usage(
                "request-1",
                "deepseek",
                "deepseek-flash",
                None,
                prompt_tokens=10,
                completion_tokens=4,
                cached_tokens=2,
                usage_day="2026-10-28",
            )
            with self.assertRaises(sqlite3.IntegrityError):
                store.record_usage(
                    "request-1",
                    "deepseek",
                    "deepseek-flash",
                    None,
                    prompt_tokens=10,
                    completion_tokens=4,
                    usage_day="2026-10-28",
                )
            usage = store.usage_for_day("deepseek", "2026-10-28")
            self.assertEqual(usage["calls"], 1)
            self.assertEqual(usage["total_tokens"], 14)
            self.assertEqual(usage["cached_tokens"], 2)

    def test_opening_is_inserted_once_only_for_an_empty_conversation(self):
        with tempfile.TemporaryDirectory() as temp:
            store = ChatStore(Path(temp) / "chat.sqlite3")
            current = store.get_or_create_role_conversation(
                "custom-opening",
                model_id="deepseek-flash",
                provider="deepseek",
            )
            self.assertTrue(
                store.ensure_opening_message(
                    current["id"],
                    "（抬眼看向你）你终于来了。桌上有封只写了你名字的信。",
                )
            )
            self.assertFalse(
                store.ensure_opening_message(
                    current["id"],
                    "不应覆盖第二条开场。",
                )
            )
            messages = store.get(current["id"])["messages"]
            self.assertEqual(len(messages), 1)
            self.assertEqual(messages[0]["role"], "assistant")
            self.assertIn("只写了你名字的信", messages[0]["content"])

    def test_old_session_schema_is_migrated_without_losing_messages(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "chat.sqlite3"
            db = sqlite3.connect(path)
            db.executescript(
                "CREATE TABLE sessions("
                "id TEXT PRIMARY KEY,model_mode TEXT NOT NULL,"
                "created_at TEXT NOT NULL,updated_at TEXT NOT NULL);"
                "CREATE TABLE messages("
                "id INTEGER PRIMARY KEY AUTOINCREMENT,"
                "session_id TEXT NOT NULL,role TEXT NOT NULL,"
                "content TEXT NOT NULL,created_at TEXT NOT NULL);"
                "INSERT INTO sessions VALUES('old','base','t','t');"
                "INSERT INTO messages(session_id,role,content,created_at)"
                " VALUES('old','user','原有消息','t');"
            )
            db.commit()
            db.close()
            store = ChatStore(path)
            session = store.get("old")
            self.assertEqual(session["provider"], "local")
            self.assertEqual(session["messages"][0]["content"], "原有消息")


if __name__ == "__main__":
    unittest.main()
