import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import boto3
import pytest
from botocore.exceptions import ClientError
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import HumanMessage, SystemMessage

from src.agent import (
    MAX_ATTEMPTS,
    get_cap_brief_llm,
    get_market_brief_llm,
    get_step_llm,
    invoke_with_retries,
)
from src.config import bedrock_client_config, get_settings
from src.llm_schemas import StepDecision


class _ThrottlingHandler(BaseHTTPRequestHandler):
    """Always answers 429/ThrottlingException and counts the POSTs it received."""

    request_count = 0

    def do_POST(self):
        type(self).request_count += 1
        body = b'{"message": "throttled"}'
        self.send_response(429)
        self.send_header("x-amzn-ErrorType", "ThrottlingException")
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture(autouse=True)
def _clear_llm_factory_caches():
    # These factories are @lru_cache in agent.py; this file calls them for
    # real (unlike test_agent.py, which always monkeypatches them), so each
    # test needs a fresh client instead of one left over from another test.
    get_step_llm.cache_clear()
    get_cap_brief_llm.cache_clear()
    get_market_brief_llm.cache_clear()
    yield
    get_step_llm.cache_clear()
    get_cap_brief_llm.cache_clear()
    get_market_brief_llm.cache_clear()


@pytest.fixture
def fake_aws_creds(monkeypatch):
    # No test here may reach real AWS: fake credentials plus no profile
    # means a real Bedrock endpoint would reject the request outright.
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "fake")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "fake")
    monkeypatch.delenv("AWS_PROFILE", raising=False)


@pytest.fixture
def throttling_stub(fake_aws_creds):
    """A local HTTP server standing in for Bedrock: every POST gets a 429
    ThrottlingException. Yields its base URL; _ThrottlingHandler.request_count
    tracks how many requests it actually received.
    """
    _ThrottlingHandler.request_count = 0
    server = HTTPServer(("127.0.0.1", 0), _ThrottlingHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join()


def _underlying_chat_model(structured_llm):
    """Reach through with_structured_output(..., include_raw=True)'s chain
    back to the ChatBedrockConverse instance a factory built, so its real
    client.meta.config can be inspected.
    """
    return structured_llm.first.steps__["raw"].bound


# --- a: the shared config caps botocore's own retries at 1 ---
def test_bedrock_client_config_sends_exactly_one_request_per_invoke(throttling_stub):
    client = boto3.client(
        "bedrock-runtime",
        region_name="us-east-1",
        endpoint_url=throttling_stub,
        config=bedrock_client_config(),
    )

    with pytest.raises(ClientError) as exc_info:
        client.converse(
            modelId="us.anthropic.claude-sonnet-4-6",
            messages=[{"role": "user", "content": [{"text": "hi"}]}],
        )

    assert exc_info.value.response["Error"]["Code"] == "ThrottlingException"
    assert _ThrottlingHandler.request_count == 1


# --- b: app-level MAX_ATTEMPTS retries == real requests (botocore adds none) ---
def test_invoke_with_retries_sends_exactly_max_attempts_requests(
    monkeypatch, throttling_stub
):
    monkeypatch.setattr("src.agent.time.sleep", lambda seconds: None)

    llm = ChatBedrockConverse(
        model=get_settings().bedrock_model_id,
        region_name=get_settings().bedrock_region,
        temperature=0,
        config=bedrock_client_config(),
        max_tokens=get_settings().bedrock_max_tokens,
        endpoint_url=throttling_stub,
    ).with_structured_output(StepDecision, include_raw=True)

    parsed, llm_errors, attempts = invoke_with_retries(
        llm, [SystemMessage(content="sys"), HumanMessage(content="hi")]
    )

    assert parsed is None
    assert attempts == MAX_ATTEMPTS
    assert llm_errors == ["ThrottlingException: throttled"] * MAX_ATTEMPTS
    assert _ThrottlingHandler.request_count == MAX_ATTEMPTS


# --- c: every factory applies the shared config and the max_tokens setting ---
@pytest.mark.parametrize(
    "factory", [get_step_llm, get_cap_brief_llm, get_market_brief_llm]
)
def test_llm_factories_apply_shared_config_and_max_tokens(fake_aws_creds, factory):
    bound = _underlying_chat_model(factory())

    assert bound.client.meta.config.retries["total_max_attempts"] == 1
    assert (
        bound.client.meta.config.connect_timeout
        == get_settings().bedrock_connect_timeout
    )
    assert bound.client.meta.config.read_timeout == get_settings().bedrock_read_timeout
    assert bound.max_tokens == get_settings().bedrock_max_tokens
