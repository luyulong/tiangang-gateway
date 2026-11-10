"""Minimal OpenAI-compatible HTTP gateway with a configurable upstream adapter."""
from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_TIMEOUT_SECONDS = 60.0
MAX_TIMEOUT_SECONDS = 300.0
MAX_BODY_BYTES = 1_048_576
ENDPOINTS = {
    "/v1/chat/completions": "chat.completion",
    "/v1/completions": "text_completion",
    "/v1/embeddings": "embeddings",
}


class ModelExecutor(Protocol):
    def execute(self, endpoint: str, request: dict[str, Any], timeout_seconds: float) -> dict[str, Any]: ...


class ServiceUnavailable(Exception):
    pass


class RequestError(Exception):
    def __init__(self, message: str, param: str | None = None, code: str | None = None):
        super().__init__(message)
        self.param = param
        self.code = code


class OpenAICompatibleExecutor:
    """Forward requests to an OpenAI-compatible provider without logging secrets."""

    def __init__(self, base_url: str, api_key: str, models: dict[str, str]):
        self.base_url = base_url.rstrip("/")
        if self.base_url.endswith("/v1"):
            self.base_url = self.base_url[:-3]
        self.api_key = api_key
        self.models = models

    def execute(self, endpoint: str, request: dict[str, Any], timeout_seconds: float) -> dict[str, Any]:
        model = self.models.get(endpoint)
        if not model:
            raise ServiceUnavailable("No upstream model is configured for this endpoint.")
        payload = dict(request)
        payload["model"] = model
        upstream = Request(
            f"{self.base_url}{endpoint}",
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
        )
        try:
            with urlopen(upstream, timeout=timeout_seconds) as response:
                return json.loads(response.read())
        except HTTPError as exc:
            if exc.code in (401, 403, 404, 429, 503):
                raise ServiceUnavailable(f"Upstream returned HTTP {exc.code}.") from None
            raise RuntimeError(f"Upstream returned HTTP {exc.code}.") from None
        except (URLError, TimeoutError, json.JSONDecodeError):
            raise ServiceUnavailable("The model provider could not be reached or returned invalid JSON.") from None


def configured_executor() -> ModelExecutor | None:
    base_url = os.environ.get("UPSTREAM_BASE_URL", "https://api.iangangaitp.com").rstrip("/")
    api_key = os.environ.get("UPSTREAM_API_KEY")
    models = {
        "/v1/chat/completions": os.environ.get("CHAT_MODEL", "deepseek-v4.1-flash"),
        "/v1/completions": os.environ.get("COMPLETION_MODEL", "deepseek-v4.1-flash"),
        "/v1/embeddings": os.environ.get("EMBEDDING_MODEL", "qwen3.7-text-embedding"),
    }
    if not api_key or not base_url:
        return None
    return OpenAICompatibleExecutor(base_url, api_key, models)


def configured_timeout() -> float:
    raw = os.environ.get("REQUEST_TIMEOUT", str(DEFAULT_TIMEOUT_SECONDS))
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError("REQUEST_TIMEOUT must be a number") from exc
    if value <= 0 or value > MAX_TIMEOUT_SECONDS:
        raise ValueError(f"REQUEST_TIMEOUT must be greater than 0 and at most {MAX_TIMEOUT_SECONDS:g}")
    return value


def validate_request(path: str, payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise RequestError("Request body must be a JSON object.", code="invalid_request")
    model = payload.get("model")
    if not isinstance(model, str) or not model.strip():
        raise RequestError("Field 'model' is required and must be a non-empty string.", "model", "missing_required_parameter")
    if path == "/v1/chat/completions":
        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages:
            raise RequestError("Field 'messages' is required and must be a non-empty array.", "messages", "missing_required_parameter")
        for i, message in enumerate(messages):
            if (not isinstance(message, dict) or not isinstance(message.get("role"), str)
                    or not message["role"].strip() or not isinstance(message.get("content"), str)):
                raise RequestError(f"Each message must contain a non-empty string role and string content (item {i}).", "messages", "invalid_type")
        if payload.get("stream") is True:
            raise RequestError("Streaming is not supported by this gateway version.", "stream", "unsupported_value")
    elif path == "/v1/completions":
        prompt = payload.get("prompt")
        if not (isinstance(prompt, str) and prompt or isinstance(prompt, list) and prompt and all(isinstance(x, str) for x in prompt)):
            raise RequestError("Field 'prompt' must be a non-empty string or non-empty array of strings.", "prompt", "invalid_type")
    elif path == "/v1/embeddings":
        value = payload.get("input")
        if not (isinstance(value, str) and value or isinstance(value, list) and value and all(isinstance(x, str) for x in value)):
            raise RequestError("Field 'input' must be a non-empty string or non-empty array of strings.", "input", "invalid_type")
    return payload


def error_body(message: str, kind: str, param: str | None = None, code: str | None = None) -> dict[str, Any]:
    error: dict[str, Any] = {"message": message, "type": kind}
    if param is not None:
        error["param"] = param
    if code is not None:
        error["code"] = code
    return {"error": error}


def validate_result(path: str, result: Any) -> dict[str, Any]:
    """Reject malformed adapter output instead of claiming SDK compatibility."""
    if not isinstance(result, dict):
        raise TypeError("executor returned an invalid response")
    if path == "/v1/embeddings":
        data = result.get("data")
        valid = (result.get("object") == "list" and isinstance(result.get("model"), str)
                 and isinstance(data, list) and all(
                     isinstance(item, dict) and item.get("object") == "embedding"
                     and isinstance(item.get("index"), int) and isinstance(item.get("embedding"), list)
                     and all(isinstance(value, (int, float)) for value in item["embedding"])
                     for item in data))
    else:
        choices = result.get("choices")
        expected_object = "chat.completion" if path == "/v1/chat/completions" else "text_completion"
        valid = (result.get("object") == expected_object and isinstance(result.get("id"), str)
                 and isinstance(result.get("created"), int) and isinstance(result.get("model"), str)
                 and isinstance(choices, list) and bool(choices))
        for choice in choices if isinstance(choices, list) else []:
            if not isinstance(choice, dict) or not isinstance(choice.get("index"), int):
                valid = False
            elif path == "/v1/chat/completions":
                valid = valid and isinstance(choice.get("message"), dict) and isinstance(choice.get("finish_reason"), (str, type(None)))
            else:
                valid = valid and isinstance(choice.get("text"), str) and isinstance(choice.get("finish_reason"), (str, type(None)))
    if not valid:
        raise TypeError("executor returned a response with an invalid OpenAI-compatible shape")
    return result


class GatewayServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], executor: ModelExecutor | None, timeout_seconds: float):
        super().__init__(address, GatewayHandler)
        self.executor = executor
        self.request_timeout_seconds = timeout_seconds
        self.execution_pool = ThreadPoolExecutor(max_workers=16, thread_name_prefix="model-executor")

    def server_close(self) -> None:
        super().server_close()
        self.execution_pool.shutdown(wait=False, cancel_futures=True)


class GatewayHandler(BaseHTTPRequestHandler):
    server: GatewayServer
    protocol_version = "HTTP/1.1"

    def _write_json(self, status: int, body: dict[str, Any]) -> None:
        encoded = json.dumps(body, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(encoded)
        self.close_connection = True

    def do_GET(self) -> None:
        if self.path == "/health":
            ready = self.server.executor is not None
            self._write_json(200 if ready else 503, {"status": "ok" if ready else "not_ready"})
        else:
            self._write_json(404, error_body("The requested endpoint was not found.", "invalid_request_error", code="not_found"))

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0]
        if path not in ENDPOINTS:
            self._write_json(404, error_body("The requested endpoint was not found.", "invalid_request_error", code="not_found"))
            return
        if not self.headers.get("Content-Type", "").lower().startswith("application/json"):
            self._write_json(415, error_body("Content-Type must be application/json.", "invalid_request_error", "content_type", "unsupported_media_type"))
            return
        try:
            length = int(self.headers.get("Content-Length", "-1"))
        except ValueError:
            length = -1
        if length < 0:
            self._write_json(400, error_body("A valid Content-Length is required.", "invalid_request_error", code="invalid_request"))
            return
        if length > MAX_BODY_BYTES:
            self._write_json(413, error_body("Request body is too large.", "invalid_request_error", code="request_too_large"))
            return
        try:
            payload = json.loads(self.rfile.read(length))
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._write_json(400, error_body("Request body must contain valid JSON.", "invalid_request_error", code="invalid_json"))
            return
        try:
            request = validate_request(path, payload)
        except RequestError as exc:
            self._write_json(400, error_body(str(exc), "invalid_request_error", exc.param, exc.code))
            return
        executor = self.server.executor
        if executor is None:
            self._write_json(503, error_body("No model executor is configured.", "service_unavailable_error", code="executor_unavailable"))
            return
        future = self.server.execution_pool.submit(executor.execute, path, request, self.server.request_timeout_seconds)
        try:
            result = validate_result(path, future.result(timeout=self.server.request_timeout_seconds))
            self._write_json(200, result)
        except FutureTimeout:
            future.cancel()
            self._write_json(504, error_body("The request timed out.", "timeout_error", code="request_timeout"))
        except ServiceUnavailable:
            self._write_json(503, error_body("The model executor is unavailable.", "service_unavailable_error", code="executor_unavailable"))
        except Exception:
            self._write_json(500, error_body("An internal error occurred.", "internal_error", code="internal_error"))


def create_server(host: str = "127.0.0.1", port: int = 8000, executor: ModelExecutor | None = None,
                  timeout_seconds: float | None = None) -> GatewayServer:
    timeout = configured_timeout() if timeout_seconds is None else timeout_seconds
    if timeout <= 0 or timeout > MAX_TIMEOUT_SECONDS:
        raise ValueError(f"timeout_seconds must be greater than 0 and at most {MAX_TIMEOUT_SECONDS:g}")
    return GatewayServer((host, port), executor, timeout)


if __name__ == "__main__":
    server = create_server(host=os.environ.get("HOST", "0.0.0.0"), port=int(os.environ.get("PORT", "8000")),
                           executor=configured_executor())
    print(f"Gateway listening on {server.server_address[0]}:{server.server_address[1]}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
