"""Transport limits apply without contacting a provider or reading real credentials."""

# Standard library
import io
import time
from urllib.error import HTTPError

# Third-party
import pytest


@pytest.fixture
def transport():
    from scripts import knowledge_utility_capture_transport

    return knowledge_utility_capture_transport


def _hang_worker(connection, endpoint, payload, key):
    time.sleep(60)


def _large_worker(connection, endpoint, payload, key):
    from scripts.knowledge_utility_capture_transport import _encode_result, _result

    connection.send_bytes(_encode_result(_result(body=b"a" * 1048576, status=200)))
    connection.close()


def test_deadline_reaps_hanging_worker(transport):
    started = time.monotonic()
    client = transport.LiveTransport("sk-test-sentinel", deadline_seconds=0.2, worker=_hang_worker)
    result = client.post("responses", {"input": "authored"})
    assert result.error_code == "deadline_exceeded"
    assert time.monotonic() - started < 5
    assert result.body == b""


def test_large_ipc_result_does_not_deadlock(transport):
    client = transport.LiveTransport("sk-test-sentinel", deadline_seconds=15, worker=_large_worker)
    result = client.post("responses", {"input": "authored"})
    assert result.error_code is None
    assert len(result.body) == 1048576


@pytest.mark.parametrize("endpoint", ["https://evil.example/", "../responses", "responses/1"])
def test_endpoint_allowlist_rejects_before_worker(transport, endpoint):
    client = transport.LiveTransport("sk-test-sentinel", worker=_hang_worker)
    with pytest.raises(ValueError):
        client.post(endpoint, {})


class FakeResponse(io.BytesIO):
    status = 200

    def __init__(self, body):
        super().__init__(body)
        self.headers = {"x-request-id": "sk-test-sentinel", "authorization": "sk-test-sentinel"}


def test_http_bounds_redacts_and_disables_redirects_proxies(transport, monkeypatch):
    calls = []
    handlers_seen = []

    class Opener:
        def open(self, request, timeout):
            calls.append(request)
            assert timeout <= 45
            return FakeResponse(b"sk-test-sentinel " + b"a" * 1048576)

    def build(*handlers):
        handlers_seen.extend(handlers)
        return Opener()

    monkeypatch.setattr(transport, "build_opener", build)
    monkeypatch.setenv("HTTPS_PROXY", "http://must-not-be-used.invalid")
    result = transport._perform_http("responses", {"input": "authored"}, "sk-test-sentinel")
    assert len(calls) == 1
    assert calls[0].full_url == "https://api.openai.com/v1/responses"
    assert calls[0].get_header("Authorization") == "Bearer sk-test-sentinel"
    assert any(getattr(handler, "proxies", None) == {} for handler in handlers_seen)
    redirect = next(h for h in handlers_seen if isinstance(h, transport.NoRedirect))
    assert redirect.redirect_request(None, None, 302, "", {}, "https://evil.example") is None
    assert result.incomplete and result.redacted and result.error_code == "response_too_large"
    assert len(result.body) <= 1048576
    assert b"sk-test-sentinel" not in result.body
    assert "sk-test-sentinel" not in repr(result)


@pytest.mark.parametrize("error_kind", ["http", "connection"])
def test_errors_are_retained_without_retry_or_secret(transport, monkeypatch, error_kind):
    calls = []

    class Opener:
        def open(self, request, timeout):
            calls.append(1)
            if error_kind == "http":
                raise HTTPError(
                    request.full_url,
                    429,
                    "sk-test-sentinel",
                    {},
                    io.BytesIO(b'{"error":"sk-test-sentinel"}'),
                )
            raise OSError("sk-test-sentinel")

    monkeypatch.setattr(transport, "build_opener", lambda *args: Opener())
    result = transport._perform_http("responses", {}, "sk-test-sentinel")
    assert calls == [1]
    assert result.error_code in ("http_error", "transport_error")
    assert "sk-test-sentinel" not in repr(result)


def test_key_file_is_bounded_and_errors_do_not_echo_contents(transport, tmp_path):
    path = tmp_path / "key"
    path.write_text("sk-test-sentinel\n", encoding="utf-8")
    assert isinstance(transport.make_live_transport(path), transport.LiveTransport)
    path.write_bytes(b"x" * 4097)
    with pytest.raises(ValueError, match="invalid credential"):
        transport.make_live_transport(path)
