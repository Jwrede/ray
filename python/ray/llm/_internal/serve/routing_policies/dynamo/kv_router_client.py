"""KvRouter client interface mirroring Dynamo's ``dynamo._core.KvRouter``.

Defines the abstract :class:`KvRouterClient` (the surface we will back with the
real Dynamo KvRouter) and :class:`StubKvRouter`, a passthrough used until the
bindings are wired in. Keeping it a narrow ABC lets us swap in the real
implementation without touching routing call sites. See each method's docstring
for its Dynamo counterpart.
"""

import abc
from dataclasses import dataclass, field
from typing import List, Optional, Set


@dataclass(frozen=True)
class WorkerScore:
    """A single ranked-worker row returned by :meth:`KvRouterClient.rank_workers`.

    Fields mirror the columns Dynamo's KvRouter produces when it ranks workers:
    the worker (and its data-parallel rank), how many KV-cache blocks overlap
    with the request's prompt, the projected prefill/decode load if the request
    were placed on this worker, and an aggregate score.
    """

    worker_id: int
    dp_rank: int = 0
    overlap_blocks: int = 0
    potential_prefill_tokens: int = 0
    potential_decode_blocks: int = 0
    score: float = 0.0


class KvRouterClient(abc.ABC):
    """Interface mirroring Dynamo's ``dynamo._core.KvRouter`` Python API.

    Implementations are expected to be backed by a real Dynamo KvRouter once
    the bindings are wired in. Until then, see :class:`StubKvRouter`.
    """

    @abc.abstractmethod
    async def rank_workers(
        self,
        token_ids: List[int],
        allowed_worker_ids: Optional[List[int]] = None,
    ) -> List[WorkerScore]:
        """Rank candidate workers for a request's prompt token IDs.

        Args:
            token_ids: The prompt token IDs of the incoming request.
            allowed_worker_ids: If provided, restrict ranking to these workers;
                otherwise all registered workers are candidates.

        Returns:
            Ranked :class:`WorkerScore` rows (best first is implementation
            defined; the stub preserves input order).
        """

    @abc.abstractmethod
    async def add_request(
        self,
        request_id: str,
        token_ids: List[int],
        worker_id: int,
        dp_rank: int = 0,
        overlap_blocks: int = 0,
        expected_output_tokens: Optional[int] = None,
    ) -> None:
        """Register an in-flight request against the chosen worker."""

    @abc.abstractmethod
    async def mark_prefill_complete(self, request_id: str) -> None:
        """Signal that ``request_id`` finished prefill and is now decoding."""

    @abc.abstractmethod
    async def add_output_block(
        self,
        request_id: str,
        decay_fraction: Optional[float] = None,
    ) -> None:
        """Record that ``request_id`` produced another block of output tokens."""

    @abc.abstractmethod
    async def free(self, request_id: str) -> None:
        """Release all router-side bookkeeping for ``request_id``."""

    @abc.abstractmethod
    def register_workers(self, worker_ids: List[int]) -> None:
        """Add ``worker_ids`` to the set the router may route to."""

    @abc.abstractmethod
    def remove_worker(self, worker_id: int) -> None:
        """Remove ``worker_id`` from the set the router may route to."""


@dataclass
class StubKvRouter(KvRouterClient):
    """Passthrough stub standing in for Dynamo's KvRouter until the real
    bindings are wired in.

    ``rank_workers`` performs an identity ranking (returns the allowed workers
    in order, with zero scores) and all lifecycle methods are no-ops. A small
    internal set of active request IDs and registered workers is kept purely for
    realism so callers can exercise the full lifecycle without surprises.
    """

    _active_request_ids: Set[str] = field(default_factory=set)
    _worker_ids: Set[int] = field(default_factory=set)

    async def rank_workers(
        self,
        token_ids: List[int],
        allowed_worker_ids: Optional[List[int]] = None,
    ) -> List[WorkerScore]:
        return [WorkerScore(worker_id=w) for w in (allowed_worker_ids or [])]

    async def add_request(
        self,
        request_id: str,
        token_ids: List[int],
        worker_id: int,
        dp_rank: int = 0,
        overlap_blocks: int = 0,
        expected_output_tokens: Optional[int] = None,
    ) -> None:
        self._active_request_ids.add(request_id)

    async def mark_prefill_complete(self, request_id: str) -> None:
        pass

    async def add_output_block(
        self,
        request_id: str,
        decay_fraction: Optional[float] = None,
    ) -> None:
        pass

    async def free(self, request_id: str) -> None:
        self._active_request_ids.discard(request_id)

    def register_workers(self, worker_ids: List[int]) -> None:
        self._worker_ids.update(worker_ids)

    def remove_worker(self, worker_id: int) -> None:
        self._worker_ids.discard(worker_id)
