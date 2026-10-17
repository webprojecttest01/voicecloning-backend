import os
import tempfile
import unittest
import uuid
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient


_test_data = tempfile.TemporaryDirectory()
os.environ["DATA_DIR"] = _test_data.name
os.environ["ENVIRONMENT"] = "test"
os.environ["FLASK_SECRET_KEY"] = "test-only-session-secret"
os.environ["SESSION_COOKIE_SAMESITE"] = "lax"
os.environ["SESSION_COOKIE_SECURE"] = "false"
os.environ["DATABASE_URL"] = ""
os.environ.pop("DEFAULT_USER_EMAIL", None)
os.environ.pop("DEFAULT_USER_PASSWORD", None)

import app as backend


class BackendApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        backend.DATABASE = str(Path(_test_data.name) / "test-users.db")
        backend.ELEVENLABS_API_KEY = None

    def setUp(self):
        self.client_context = TestClient(backend.app)
        self.client = self.client_context.__enter__()
        self.test_email = f"test-{uuid.uuid4().hex}@example.com"

    def tearDown(self):
        self.client_context.__exit__(None, None, None)

    def signup(self):
        return self.client.post(
            "/api/auth/signup",
            json={
                "name": "Test User",
                "email": self.test_email,
                "password": "test-password-123"
            }
        )

    def test_startup_creates_database_and_health_checks_it(self):
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["database"], "connected")

        import sqlite3

        with closing(sqlite3.connect(backend.DATABASE)) as connection:
            tables = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
        self.assertTrue({"users", "transactions", "user_balances"} <= tables)

    def test_default_account_is_seeded_hashed_and_idempotently(self):
        import sqlite3

        email = f"default-{uuid.uuid4().hex}@example.com"
        password = "bootstrap-password-123"
        with patch.dict(
            os.environ,
            {
                "DEFAULT_USER_NAME": "Bootstrap User",
                "DEFAULT_USER_EMAIL": email,
                "DEFAULT_USER_PASSWORD": password
            }
        ):
            backend.init_database()
            backend.init_database()

        with closing(sqlite3.connect(backend.DATABASE)) as connection:
            connection.row_factory = sqlite3.Row
            user = connection.execute(
                "SELECT id, password FROM users WHERE email = ?",
                (email,)
            ).fetchone()
            balance = connection.execute(
                "SELECT user_id FROM user_balances WHERE user_id = ?",
                (user["id"],)
            ).fetchone()

        self.assertIsNotNone(user)
        self.assertNotEqual(user["password"], password)
        self.assertTrue(backend.check_password_hash(user["password"], password))
        self.assertEqual(balance["user_id"], user["id"])
        login = self.client.post(
            "/api/auth/login",
            json={"email": email, "password": password}
        )
        self.assertEqual(login.status_code, 200)

    def test_signup_login_session_and_logout(self):
        response = self.signup()
        self.assertEqual(response.status_code, 201)
        user_id = response.json()["user"]["id"]

        current = self.client.get("/api/auth/me")
        self.assertEqual(current.status_code, 200)
        self.assertEqual(current.json()["user"]["id"], user_id)

        logout = self.client.post("/api/auth/logout")
        self.assertEqual(logout.status_code, 200)
        self.assertEqual(self.client.get("/api/auth/me").status_code, 401)

        login = self.client.post(
            "/api/auth/login",
            json={"email": self.test_email.upper(), "password": "test-password-123"}
        )
        self.assertEqual(login.status_code, 200)
        self.assertEqual(self.client.get("/api/auth/me").status_code, 200)

        duplicate = self.signup()
        self.assertEqual(duplicate.status_code, 409)

    def test_payment_calculation_record_history_and_topup(self):
        self.assertEqual(self.signup().status_code, 201)

        calculation = self.client.post(
            "/api/payments/calculate",
            json={"text": "Hello from the API", "model": "chatterbox_voice_cloning"}
        )
        self.assertEqual(calculation.status_code, 200)
        self.assertGreater(calculation.json()["token_count"], 0)

        recorded = self.client.post(
            "/api/payments/record",
            json={"text": "Hello from the API", "model": "chatterbox_voice_cloning"}
        )
        self.assertEqual(recorded.status_code, 200, recorded.text)
        transaction_id = recorded.json()["transaction_id"]

        history = self.client.get("/api/payments/history")
        self.assertEqual(history.status_code, 200)
        self.assertEqual(history.json()["transactions"][0]["id"], transaction_id)

        topup = self.client.post("/api/payments/topup", json={"tokens": 1000})
        self.assertEqual(topup.status_code, 200)
        self.assertEqual(topup.json()["added_tokens"], 1000)
        self.assertEqual(self.client.get("/api/payments/history").status_code, 200)

    def test_public_routes_and_generation_endpoint_contracts(self):
        self.assertEqual(self.client.get("/").status_code, 200)
        self.assertEqual(self.client.get("/languages").status_code, 200)
        self.assertIn("en", self.client.get("/languages").json()["languages"])

        missing_key_voices = self.client.get("/api/voices")
        self.assertEqual(missing_key_voices.status_code, 500)

        backend.ELEVENLABS_API_KEY = "test-key"
        try:
            invalid_tts = self.client.post("/api/tts", json={"text": "hello"})
        finally:
            backend.ELEVENLABS_API_KEY = None
        self.assertEqual(invalid_tts.status_code, 400)

        self.assertEqual(self.client.post("/tts", data={}).status_code, 422)
        self.assertEqual(self.client.post("/stt", files={}).status_code, 422)
        self.assertEqual(self.client.post("/dub", data={}).status_code, 422)


if __name__ == "__main__":
    unittest.main()
