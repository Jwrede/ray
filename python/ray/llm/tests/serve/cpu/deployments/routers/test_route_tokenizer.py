import sys
from typing import List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest
from starlette.datastructures import Headers

from ray.llm._internal.serve.core.configs.openai_api_models import (
    ErrorInfo,
    ErrorResponse,
    TokenizeChatRequest,
    TokenizeCompletionRequest,
    TokenizeResponse,
)
from ray.llm._internal.serve.core.ingress.route_tokenizer import (
    RouteTokenizedRequest,
    RouteTokenizer,
)
from ray.llm._internal.serve.core.ingress.router import LLMRouter
from ray.llm._internal.serve.routing_policies.dynamo.kv_router_client import (
    StubKvRouter,
    WorkerScore,
)


class _FakeRequest:
    """Minimal Starlette-request stand-in (mirrors test_router.py)."""

    def __init__(self, body: bytes, headers: Optional[dict] = None):
        self._body = body
        self.headers = Headers(headers or {})

    async def body(self) -> bytes:
        return self._body


def _make_handle_returning(resp):
    """Build a mock deployment handle whose ``.tokenize.remote`` yields ``resp``.

    Returns ``(handle, captured)`` where ``captured`` records the positional
    args passed into ``tokenize.remote`` (so tests can inspect the built
    Tokenize* request).
    """
    captured = {}

    async def _gen(req, raw_request_info):
        captured["req"] = req
        captured["raw_request_info"] = raw_request_info
        yield resp

    handle = MagicMock()
    # options(stream=True) returns a handle exposing tokenize.remote.
    streamed = MagicMock()
    streamed.tokenize.remote = _gen
    handle.options.return_value = streamed
    return handle, captured


class TestRouteTokenizer:
    @pytest.mark.asyncio
    async def test_chat_path(self):
        resp = TokenizeResponse(count=3, max_model_len=2048, tokens=[1, 2, 3])
        handle, captured = _make_handle_returning(resp)
        tokenizer = RouteTokenizer(handle)

        body = (
            b'{"model": "m", "messages": [{"role": "user", "content": "hi"}], '
            b'"max_tokens": 16}'
        )
        result = await tokenizer.tokenize(body, body_truncated=False)

        assert result == RouteTokenizedRequest(
            endpoint="chat",
            token_ids=[1, 2, 3],
            expected_output_tokens=16,
            model="m",
        )
        # Inspect the request actually passed to tokenize.remote.
        handle.options.assert_called_once_with(stream=True)
        tok_req = captured["req"]
        assert isinstance(tok_req, TokenizeChatRequest)
        assert tok_req.model == "m"
        assert tok_req.messages == [{"role": "user", "content": "hi"}]
        assert tok_req.add_generation_prompt is True
        assert captured["raw_request_info"] is None

    @pytest.mark.asyncio
    async def test_chat_path_prefers_max_completion_tokens(self):
        resp = TokenizeResponse(count=2, max_model_len=2048, tokens=[7, 8])
        handle, _ = _make_handle_returning(resp)
        tokenizer = RouteTokenizer(handle)

        body = (
            b'{"model": "m", "messages": [{"role": "user", "content": "hi"}], '
            b'"max_completion_tokens": 5, "max_tokens": 16}'
        )
        result = await tokenizer.tokenize(body, body_truncated=False)
        assert result.expected_output_tokens == 5

    @pytest.mark.asyncio
    async def test_completion_path(self):
        resp = TokenizeResponse(count=2, max_model_len=2048, tokens=[10, 11])
        handle, captured = _make_handle_returning(resp)
        tokenizer = RouteTokenizer(handle)

        body = b'{"model": "m", "prompt": "Hello", "max_tokens": 8}'
        result = await tokenizer.tokenize(body, body_truncated=False)

        assert result == RouteTokenizedRequest(
            endpoint="completion",
            token_ids=[10, 11],
            expected_output_tokens=8,
            model="m",
        )
        tok_req = captured["req"]
        assert isinstance(tok_req, TokenizeCompletionRequest)
        assert tok_req.prompt == "Hello"
        assert tok_req.add_special_tokens is True

    @pytest.mark.asyncio
    async def test_truncated_body_returns_none_and_skips_handle(self):
        resp = TokenizeResponse(count=1, max_model_len=2048, tokens=[1])
        handle, _ = _make_handle_returning(resp)
        tokenizer = RouteTokenizer(handle)

        body = b'{"model": "m", "prompt": "Hello"}'
        result = await tokenizer.tokenize(body, body_truncated=True)

        assert result is None
        handle.options.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_body_returns_none(self):
        handle, _ = _make_handle_returning(None)
        tokenizer = RouteTokenizer(handle)
        assert await tokenizer.tokenize(b"", body_truncated=False) is None
        handle.options.assert_not_called()

    @pytest.mark.asyncio
    async def test_multi_prompt_list_returns_none(self):
        resp = TokenizeResponse(count=1, max_model_len=2048, tokens=[1])
        handle, _ = _make_handle_returning(resp)
        tokenizer = RouteTokenizer(handle)

        body = b'{"model": "m", "prompt": ["a", "b"]}'
        result = await tokenizer.tokenize(body, body_truncated=False)

        assert result is None
        handle.options.assert_not_called()

    @pytest.mark.asyncio
    async def test_error_response_returns_none(self):
        err = ErrorResponse(
            error=ErrorInfo(message="boom", type="BadRequest", code=400)
        )
        handle, _ = _make_handle_returning(err)
        tokenizer = RouteTokenizer(handle)

        body = b'{"model": "m", "prompt": "Hello"}'
        result = await tokenizer.tokenize(body, body_truncated=False)
        assert result is None

    @pytest.mark.asyncio
    async def test_invalid_json_returns_none(self):
        handle, _ = _make_handle_returning(None)
        tokenizer = RouteTokenizer(handle)
        assert await tokenizer.tokenize(b"not json {", body_truncated=False) is None
        handle.options.assert_not_called()

    @pytest.mark.asyncio
    async def test_non_dict_json_returns_none(self):
        handle, _ = _make_handle_returning(None)
        tokenizer = RouteTokenizer(handle)
        assert await tokenizer.tokenize(b"[1, 2, 3]", body_truncated=False) is None
        handle.options.assert_not_called()

    @pytest.mark.asyncio
    async def test_neither_messages_nor_prompt_returns_none(self):
        handle, _ = _make_handle_returning(None)
        tokenizer = RouteTokenizer(handle)
        assert await tokenizer.tokenize(b'{"model": "m"}', body_truncated=False) is None
        handle.options.assert_not_called()


class TestStubKvRouter:
    @pytest.mark.asyncio
    async def test_rank_workers_identity_passthrough(self):
        stub = StubKvRouter()
        ranked = await stub.rank_workers([1, 2, 3], allowed_worker_ids=[5, 9, 2])
        assert ranked == [
            WorkerScore(worker_id=5),
            WorkerScore(worker_id=9),
            WorkerScore(worker_id=2),
        ]
        # All scores/fields are zero in the passthrough stub.
        assert all(w.score == 0.0 and w.overlap_blocks == 0 for w in ranked)

    @pytest.mark.asyncio
    async def test_rank_workers_no_allowed_returns_empty(self):
        stub = StubKvRouter()
        assert await stub.rank_workers([1, 2, 3]) == []
        assert await stub.rank_workers([1, 2, 3], allowed_worker_ids=None) == []


class _SpyKvRouter(StubKvRouter):
    """Records the token_ids passed to ``rank_workers``."""

    def __init__(self):
        super().__init__()
        self.rank_calls: List[List[int]] = []

    async def rank_workers(self, token_ids, allowed_worker_ids=None):
        self.rank_calls.append(list(token_ids))
        return await super().rank_workers(token_ids, allowed_worker_ids)


class _FakeRouteTokenizer:
    """Records calls and returns a preset RouteTokenizedRequest."""

    def __init__(self, result: Optional[RouteTokenizedRequest]):
        self._result = result
        self.calls = []

    async def tokenize(self, request_body, body_truncated):
        self.calls.append((request_body, body_truncated))
        return self._result


class TestLLMRouterRoute:
    @pytest.mark.asyncio
    async def test_route_tokenizes_and_ranks_before_pick(self):
        # Bypass the async __init__ (mirrors test_router.py's _new_direct_router).
        router = LLMRouter.__new__(LLMRouter)
        router._handle = MagicMock()

        rt = RouteTokenizedRequest(endpoint="chat", token_ids=[11, 22, 33], model="m")
        fake_tokenizer = _FakeRouteTokenizer(rt)
        spy_kv = _SpyKvRouter()
        router._route_tokenizer = fake_tokenizer
        router._kv_router = spy_kv
        router._pick_replica = AsyncMock(return_value=("h", 1, "rid"))

        body = b'{"model": "m", "messages": [{"role": "user", "content": "hi"}]}'
        request = _FakeRequest(body)

        result = await router.route(request)

        assert result == {"host": "h", "port": 1, "replica_id": "rid"}
        # Tokenizer called with the raw body and truncation flag.
        assert fake_tokenizer.calls == [(body, False)]
        # KvRouter.rank_workers invoked with the tokenized token IDs.
        assert spy_kv.rank_calls == [[11, 22, 33]]
        router._pick_replica.assert_called_once_with(
            handle=router._handle, request_body=body, body_truncated=False
        )

    @pytest.mark.asyncio
    async def test_route_resilient_when_tokenization_returns_none(self):
        """Tokenization failure (None) must not break routing or call rank_workers."""
        router = LLMRouter.__new__(LLMRouter)
        router._handle = MagicMock()

        fake_tokenizer = _FakeRouteTokenizer(None)
        spy_kv = _SpyKvRouter()
        router._route_tokenizer = fake_tokenizer
        router._kv_router = spy_kv
        router._pick_replica = AsyncMock(return_value=("h", 2, "rid2"))

        request = _FakeRequest(b'{"model": "m", "prompt": ["a", "b"]}')
        result = await router.route(request)

        assert result == {"host": "h", "port": 2, "replica_id": "rid2"}
        assert fake_tokenizer.calls == [
            (b'{"model": "m", "prompt": ["a", "b"]}', False)
        ]
        # rank_workers must NOT be called when tokenization yields None.
        assert spy_kv.rank_calls == []


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__]))
