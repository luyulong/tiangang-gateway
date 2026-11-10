import json
import os
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from unittest.mock import patch

from gateway import OpenAICompatibleExecutor, ServiceUnavailable, configured_executor, create_server


class FakeExecutor:
    def execute(self, endpoint, request, timeout_seconds):
        if endpoint == "/v1/chat/completions":
            return {"id": "chatcmpl-test", "object": "chat.completion", "created": 1,
                    "model": request["model"], "choices": [{"index": 0,
                    "message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}]}
        if endpoint == "/v1/completions":
            return {"id": "cmpl-test", "object": "text_completion", "created": 1,
                    "model": request["model"], "choices": [{"index": 0, "text": "hello", "finish_reason": "stop"}]}
        return {"object": "list", "model": request["model"], "data": [
            {"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}]}


class SlowExecutor:
    def execute(self, endpoint, request, timeout_seconds):
        time.sleep(0.2)
        return FakeExecutor().execute(endpoint, request, timeout_seconds)


class FailingExecutor:
    def execute(self, endpoint, request, timeout_seconds):
        raise ServiceUnavailable("sensitive upstream detail")


class CrashingExecutor:
    def execute(self, endpoint, request, timeout_seconds):
        raise RuntimeError("secret token and internal stack")


class UpstreamHandler(BaseHTTPRequestHandler):
    seen = []

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).seen.append((self.path, self.headers.get("authorization"), payload))
        if self.path == "/v1/embeddings":
            body = {"object": "list", "model": payload["model"], "data": [
                {"object": "embedding", "index": 0, "embedding": [0.25, 0.75]}]}
        elif self.path == "/v1/completions":
            body = {"id": "cmpl-local", "object": "text_completion", "created": 1,
                    "model": payload["model"], "choices": [{"index": 0, "text": "local", "finish_reason": "stop"}]}
        else:
            body = {"id": "chat-local", "object": "chat.completion", "created": 1,
                    "model": payload["model"], "choices": [{"index": 0,
                    "message": {"role": "assistant", "content": "local"}, "finish_reason": "stop"}]}
        encoded = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, fmt, *args):
        pass


class GatewayTests(unittest.TestCase):
    def setUp(self):
        self.server = create_server("127.0.0.1", 0, FakeExecutor(), timeout_seconds=0.1)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=1)

    def request(self, path, body=None, content_type="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode() if body is not None else b""
        req = Request(self.base + path, data=data, headers={"Content-Type": content_type})
        try:
            response = urlopen(req, timeout=2)
        except HTTPError as exc:
            response = exc
        return response.code, response.headers.get_content_type(), json.loads(response.read())

    def test_minimal_requests_return_openai_response_shapes(self):
        cases = [
            ("/v1/chat/completions", {"model": "m", "messages": [{"role": "user", "content": "hi"}]}, "chat.completion"),
            ("/v1/completions", {"model": "m", "prompt": "hi"}, "text_completion"),
            ("/v1/embeddings", {"model": "m", "input": "hi"}, "list"),
        ]
        for path, body, expected in cases:
            with self.subTest(path=path):
                status, content_type, result = self.request(path, body)
                self.assertEqual(status, 200)
                self.assertEqual(content_type, "application/json")
                self.assertEqual(result["object"], expected)

    def test_validation_errors_share_openai_envelope(self):
        cases = [
            ("/v1/chat/completions", {"model": "m"}),
            ("/v1/completions", {"model": "m", "prompt": 4}),
            ("/v1/embeddings", {"input": "hi"}),
            ("/v1/chat/completions", {"model": "m", "messages": [], "stream": True}),
        ]
        for path, body in cases:
            with self.subTest(path=path, body=body):
                status, content_type, result = self.request(path, body)
                self.assertEqual(status, 400)
                self.assertEqual(content_type, "application/json")
                self.assertIn("message", result["error"])
                self.assertIn("type", result["error"])

    def test_bad_json_and_content_type_are_json_errors(self):
        for path, raw, media, expected in [
            ("/v1/completions", b"{", "application/json", 400),
            ("/v1/embeddings", b"{}", "text/plain", 415),
        ]:
            status, content_type, result = self.request(path, raw, media)
            self.assertEqual(status, expected)
            self.assertEqual(content_type, "application/json")
            self.assertIn("error", result)

    def test_unconfigured_executor_and_health_are_unavailable(self):
        server = create_server("127.0.0.1", 0, timeout_seconds=0.1)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base = f"http://127.0.0.1:{server.server_address[1]}"
            with urlopen(base + "/health") as response:
                self.fail(f"expected unhealthy status, received {response.status}")
        except HTTPError as exc:
            self.assertEqual(exc.code, 503)
            self.assertEqual(json.loads(exc.read())["status"], "not_ready")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=1)

    def test_executor_failures_are_sanitized_and_timeouts_bounded(self):
        self.server.executor = FailingExecutor()
        status, _, body = self.request("/v1/completions", {"model": "m", "prompt": "x"})
        self.assertEqual(status, 503)
        self.assertNotIn("sensitive", json.dumps(body))
        self.server.executor = CrashingExecutor()
        status, _, body = self.request("/v1/completions", {"model": "m", "prompt": "x"})
        self.assertEqual(status, 500)
        self.assertNotIn("secret", json.dumps(body))
        self.server.executor = SlowExecutor()
        start = time.monotonic()
        status, _, body = self.request("/v1/completions", {"model": "m", "prompt": "x"})
        self.assertEqual(status, 504)
        self.assertLess(time.monotonic() - start, 0.5)
        self.assertEqual(body["error"]["type"], "timeout_error")

    def test_openai_compatible_upstream_receives_mapped_models_and_bearer_key(self):
        UpstreamHandler.seen = []
        upstream = ThreadingHTTPServer(("127.0.0.1", 0), UpstreamHandler)
        thread = threading.Thread(target=upstream.serve_forever, daemon=True)
        thread.start()
        try:
            executor = OpenAICompatibleExecutor(
                f"http://127.0.0.1:{upstream.server_address[1]}/v1",
                "test-only-key",
                {"/v1/chat/completions": "deepseek-v4.1-flash",
                 "/v1/completions": "deepseek-v4.1-flash",
                 "/v1/embeddings": "qwen3.7-text-embedding"},
            )
            cases = [
                ("/v1/chat/completions", {"model": "alias", "messages": [{"role": "user", "content": "hi"}]}, "deepseek-v4.1-flash"),
                ("/v1/completions", {"model": "alias", "prompt": "hi"}, "deepseek-v4.1-flash"),
                ("/v1/embeddings", {"model": "alias", "input": "hi"}, "qwen3.7-text-embedding"),
            ]
            for path, body, expected_model in cases:
                result = executor.execute(path, body, 2)
                self.assertEqual(result["model"], expected_model)
            self.assertEqual(len(UpstreamHandler.seen), 3)
            seen_by_path = {path: (authorization, payload) for path, authorization, payload in UpstreamHandler.seen}
            self.assertEqual(set(seen_by_path), {path for path, _, _ in cases})
            for path, (authorization, payload) in seen_by_path.items():
                self.assertEqual(authorization, "Bearer test-only-key")
                self.assertEqual(payload["model"], "qwen3.7-text-embedding" if path == "/v1/embeddings" else "deepseek-v4.1-flash")
        finally:
            upstream.shutdown()
            upstream.server_close()
            thread.join(timeout=1)

    def test_runtime_executor_requires_injected_secret_and_uses_configured_models(self):
        with patch.dict(os.environ, {
            "UPSTREAM_BASE_URL": "https://provider.invalid/v1",
            "UPSTREAM_API_KEY": "test-only-key",
            "EMBEDDING_MODEL": "qwen3.7-text-embedding",
        }, clear=True):
            executor = configured_executor()
            self.assertIsInstance(executor, OpenAICompatibleExecutor)
            self.assertEqual(executor.models["/v1/embeddings"], "qwen3.7-text-embedding")
        with patch.dict(os.environ, {"UPSTREAM_BASE_URL": "https://provider.invalid/v1"}, clear=True):
            self.assertIsNone(configured_executor())


if __name__ == "__main__":
    unittest.main()
