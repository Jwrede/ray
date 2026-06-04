"""Pre-routing tokenization of incoming OpenAI requests.

:class:`RouteTokenizer` turns a raw ChatCompletions / Completions request body
into prompt token IDs by calling the selected vLLM replica's ``/tokenize``
endpoint through the LLMServer deployment handle. The replica applies the chat
template for chat requests, so the resulting token IDs match what the engine
will actually prefill.

The token IDs feed KV-aware routing (see ``KvRouterClient.rank_workers``).
Tokenization is best-effort: any failure returns ``None`` so the caller falls
back to normal (body-unaware) routing.
"""

import json
from dataclasses import dataclass
from typing import List, Optional

from ray.llm._internal.serve.core.configs.openai_api_models import (
    ErrorResponse,
    TokenizeChatRequest,
    TokenizeCompletionRequest,
)
from ray.llm._internal.serve.observability.logging import get_logger
from ray.serve.handle import DeploymentHandle

logger = get_logger(__name__)


@dataclass(frozen=True)
class RouteTokenizedRequest:
    """Result of tokenizing an incoming request for routing purposes."""

    endpoint: str  # "chat" | "completion"
    token_ids: List[int]
    expected_output_tokens: Optional[int] = None
    model: Optional[str] = None


class RouteTokenizer:
    """Tokenizes incoming requests via the replica's ``/tokenize`` endpoint.

    Args:
        handle: A handle to the LLMServer deployment. Its ``tokenize`` method is
            an async generator yielding exactly one ``TokenizeResponse`` (or an
            ``ErrorResponse``).
    """

    def __init__(self, handle: DeploymentHandle):
        self._handle = handle

    async def tokenize(
        self, request_body: bytes, body_truncated: bool
    ) -> Optional[RouteTokenizedRequest]:
        """Tokenize ``request_body`` into prompt token IDs.

        Returns ``None`` (graceful fallback to normal routing) when the body is
        truncated/empty, not a supported request shape, or tokenization fails.
        """
        try:
            if body_truncated or not request_body:
                return None

            payload = json.loads(request_body)
            if not isinstance(payload, dict):
                return None

            model = payload.get("model")
            tok_req: object
            if "messages" in payload:
                endpoint = "chat"
                tok_req = TokenizeChatRequest.model_validate(
                    {
                        "model": model,
                        "messages": payload["messages"],
                        "add_generation_prompt": True,
                    }
                )
            elif "prompt" in payload:
                prompt = payload["prompt"]
                if not isinstance(prompt, str):
                    # Multi-prompt (list) tokenization is out of scope; fall
                    # back to normal routing.
                    return None
                endpoint = "completion"
                tok_req = TokenizeCompletionRequest.model_validate(
                    {
                        "model": model,
                        "prompt": prompt,
                        "add_special_tokens": True,
                    }
                )
            else:
                return None

            gen = self._handle.options(stream=True).tokenize.remote(tok_req, None)
            resp = await gen.__anext__()
            if isinstance(resp, ErrorResponse):
                return None

            expected_output_tokens = payload.get(
                "max_completion_tokens"
            ) or payload.get("max_tokens")

            return RouteTokenizedRequest(
                endpoint=endpoint,
                token_ids=list(resp.tokens),
                expected_output_tokens=expected_output_tokens,
                model=model,
            )
        except Exception as e:
            logger.debug("Pre-routing tokenization failed, falling back: %s", e)
            return None
