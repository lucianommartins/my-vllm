# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch

import vllm.envs as envs
from vllm.model_executor.layers.fused_moe.all2all_utils import get_ep_all2all_manager
from vllm.utils.torch_utils import PIN_MEMORY
from vllm.v1.outputs import (
    AsyncModelRunnerOutput,
    KVConnectorOutput,
    LogprobsTensors,
    ModelRunnerOutput,
    PoolerOutput,
)
from vllm.v1.worker.gpu.sample.output import SamplerOutput, SamplingMaskTensors
from vllm.v1.worker.utils import raise_if_nan_logits

if TYPE_CHECKING:
    from vllm.distributed.aux_output_connector.worker import PendingAuxOutput
    from vllm.v1.worker.gpu.input_batch import InputBatch


@dataclass(frozen=True)
class StepTimingSample:
    forward_ms: float
    drafter_ms: float
    num_target_tokens: int
    num_reqs: int
    full_cudagraph: bool


def _timing_event() -> torch.cuda.Event:
    return torch.cuda.Event(enable_timing=True)


@dataclass
class StepTimingEvents:
    forward_start: torch.cuda.Event = field(default_factory=_timing_event)
    forward_end: torch.cuda.Event = field(default_factory=_timing_event)
    drafter_start: torch.cuda.Event = field(default_factory=_timing_event)
    drafter_end: torch.cuda.Event = field(default_factory=_timing_event)


class StepTimingCollector:
    """Times the steps run inside ``collect``; record calls no-op outside it.

    Every step gets its own events, so the steps queue back-to-back and the
    block resolves them all behind one sync on the way out.
    """

    def __init__(self):
        self._collecting = False
        self._step: StepTimingEvents | None = None
        self._batch = (False, 0, 0)
        self._timed: list[tuple[StepTimingEvents, tuple[bool, int, int]]] = []

    @contextlib.contextmanager
    def collect(self) -> Iterator[list[StepTimingSample]]:
        """Time every step run in this block.

        The yielded list holds one sample per timed step once the block exits;
        it stays empty inside the block, where the timings are still on device.
        """
        samples: list[StepTimingSample] = []
        self._collecting = True
        try:
            yield samples
        finally:
            self._collecting = False
            timed, self._timed, self._step = self._timed, [], None
        if not timed:
            return
        # Same stream, issue order: once the last step is done, so are the rest.
        timed[-1][0].drafter_end.synchronize()
        samples.extend(
            StepTimingSample(
                events.forward_start.elapsed_time(events.forward_end),
                events.drafter_start.elapsed_time(events.drafter_end),
                num_target_tokens,
                num_reqs,
                full_cudagraph,
            )
            for events, (full_cudagraph, num_target_tokens, num_reqs) in timed
        )

    def record_batch(self, input_batch: "InputBatch", full_cudagraph: bool) -> None:
        """Costs from different execution modes must not share a cost curve."""
        self._batch = (full_cudagraph, input_batch.num_tokens, input_batch.num_reqs)

    def forward_start(self) -> None:
        if self._collecting:
            self._step = StepTimingEvents()
            self._step.forward_start.record()

    def forward_end(self) -> None:
        if self._step is not None:
            self._step.forward_end.record()

    def drafter_start(self) -> None:
        if self._step is not None:
            self._step.drafter_start.record()

    def drafter_end(self) -> None:
        """Ends the step: only steps that reach here have a draft cost."""
        if self._step is not None:
            self._step.drafter_end.record()
            self._timed.append((self._step, self._batch))
            self._step = None


_V1_EXT_UNSET = object()
_V1_EXT = _V1_EXT_UNSET


def _get_v1_ext():
    global _V1_EXT
    if _V1_EXT is _V1_EXT_UNSET:
        from vllm.v1.worker.gpu._v1_sm80_ops import get_ext

        _V1_EXT = get_ext()
    return _V1_EXT


class AsyncOutputPool:
    """Reusable pool of blocking CUDA events and pinned D2H staging buffers."""

    def __init__(self, max_num_reqs: int, max_size: int = 8):
        self.max_num_reqs = max_num_reqs
        self.max_size = max_size
        self._events: list[torch.cuda.Event] = []
        self._wait_events: list[torch.cuda.Event] = []
        self._sampled_bufs: list[tuple[torch.Tensor, np.ndarray]] = []
        self._fast_slots: list[list] = []
        self._num_bufs: dict[torch.dtype, list[torch.Tensor]] = {}
        self._ones_np = np.ones(max_num_reqs, dtype=np.int32)
        self._ones_np.setflags(write=False)
        self._ones_views: dict[int, np.ndarray] = {}
        self._last_ones_n: int = -1
        self._last_ones_np: np.ndarray | None = None
        self._cached_main_stream: torch.cuda.Stream | None = None
        self._cached_main_stream_h: int = 0
        self._cached_copy_stream: torch.cuda.Stream | None = None
        self._cached_copy_stream_h: int = 0

    def ones_np(self, n: int) -> np.ndarray:
        if n == self._last_ones_n and self._last_ones_np is not None:
            return self._last_ones_np
        view = self._ones_views.get(n)
        if view is None:
            view = self._ones_np[:n]
            if len(self._ones_views) <= 256:
                self._ones_views[n] = view
        self._last_ones_n = n
        self._last_ones_np = view
        return view

    def acquire_fast_slot(
        self,
        main_stream: torch.cuda.Stream,
        copy_stream: torch.cuda.Stream,
    ) -> tuple[list, int, int]:
        if main_stream is not self._cached_main_stream:
            self._cached_main_stream = main_stream
            self._cached_main_stream_h = main_stream.cuda_stream
        if copy_stream is not self._cached_copy_stream:
            self._cached_copy_stream = copy_stream
            self._cached_copy_stream_h = copy_stream.cuda_stream
        if self._fast_slots:
            slot = self._fast_slots.pop()
        else:
            copy_ev = (
                self._events.pop()
                if self._events
                else torch.cuda.Event(blocking=True)
            )
            wait_ev = (
                self._wait_events.pop()
                if self._wait_events
                else torch.cuda.Event(blocking=False)
            )
            entry = (
                self._sampled_bufs.pop()
                if self._sampled_bufs
                else self._new_sampled_entry()
            )
            slot = [copy_ev, copy_ev.cuda_event, wait_ev, wait_ev.cuda_event, entry]
        if slot[3] == 0:
            slot[2].record(main_stream)
            slot[3] = slot[2].cuda_event
        if slot[1] == 0:
            slot[0].record(copy_stream)
            slot[1] = slot[0].cuda_event
        return slot, self._cached_main_stream_h, self._cached_copy_stream_h

    def release_fast_slot(self, slot: list) -> None:
        if len(self._fast_slots) < self.max_size:
            self._fast_slots.append(slot)
        else:
            self.release(slot[0], slot[4], None, wait_event=slot[2])

    def _new_sampled_entry(self) -> tuple[torch.Tensor, np.ndarray]:
        buf = torch.empty(
            (self.max_num_reqs, 1),
            dtype=torch.int64,
            device="cpu",
            pin_memory=PIN_MEMORY,
        )
        return buf, buf.numpy()

    def acquire_event(self) -> torch.cuda.Event:
        if self._events:
            return self._events.pop()
        return torch.cuda.Event(blocking=True)

    def acquire_wait_event(self) -> torch.cuda.Event:
        if self._wait_events:
            return self._wait_events.pop()
        return torch.cuda.Event(blocking=False)

    def acquire_sampled_entry(self) -> tuple[torch.Tensor, np.ndarray]:
        if self._sampled_bufs:
            return self._sampled_bufs.pop()
        buf = torch.empty(
            (self.max_num_reqs, 1),
            dtype=torch.int64,
            device="cpu",
            pin_memory=PIN_MEMORY,
        )
        return buf, buf.numpy()

    def copy_sampled_to_np(
        self, x: torch.Tensor
    ) -> tuple[tuple[torch.Tensor, np.ndarray] | None, np.ndarray]:
        if (
            x.ndim == 2
            and x.shape[1] == 1
            and x.dtype == torch.int64
            and x.shape[0] <= self.max_num_reqs
        ):
            entry = self.acquire_sampled_entry()
            buf, buf_np = entry
            buf[: x.shape[0]].copy_(x, non_blocking=True)
            return entry, buf_np[: x.shape[0]]
        return None, async_copy_to_np(x)

    def copy_num_to_np(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor | None, np.ndarray]:
        if x.ndim == 1 and x.shape[0] <= self.max_num_reqs:
            bufs = self._num_bufs.get(x.dtype)
            if bufs:
                buf = bufs.pop()
            else:
                buf = torch.empty(
                    self.max_num_reqs,
                    dtype=x.dtype,
                    device="cpu",
                    pin_memory=PIN_MEMORY,
                )
            view = buf[: x.shape[0]]
            view.copy_(x, non_blocking=True)
            return buf, view.numpy()
        return None, async_copy_to_np(x)

    def release(
        self,
        event: torch.cuda.Event,
        sampled_buf: tuple[torch.Tensor, np.ndarray] | torch.Tensor | None,
        num_buf: torch.Tensor | None,
        wait_event: torch.cuda.Event | None = None,
    ) -> None:
        if len(self._events) < self.max_size:
            self._events.append(event)
        if wait_event is not None and len(self._wait_events) < self.max_size:
            self._wait_events.append(wait_event)
        if sampled_buf is not None and len(self._sampled_bufs) < self.max_size:
            entry = (
                sampled_buf
                if isinstance(sampled_buf, tuple)
                else (sampled_buf, sampled_buf.numpy())
            )
            self._sampled_bufs.append(entry)
        if num_buf is not None:
            bufs = self._num_bufs.setdefault(num_buf.dtype, [])
            if len(bufs) < self.max_size:
                bufs.append(num_buf)


class AsyncOutput(AsyncModelRunnerOutput):
    def __init__(
        self,
        model_runner_output: ModelRunnerOutput,
        sampler_output: SamplerOutput,
        num_sampled_tokens: torch.Tensor,
        main_stream: torch.cuda.Stream,
        copy_stream: torch.cuda.Stream,
        check_ep_fault: bool,
        pending_aux_output: "PendingAuxOutput | None",
        pool: AsyncOutputPool | None = None,
    ):
        # NOTE(woosuk): We must retain references to the GPU tensors,
        # as the copy operations are performed on a different CUDA stream than
        # the one where the tensors were created.
        self.model_runner_output = model_runner_output
        self.sampler_output = sampler_output
        self.num_sampled_tokens = num_sampled_tokens
        self.pending_aux_output = pending_aux_output
        self._pool = pool
        self._has_fault: torch.Tensor | None = None
        self._pooled_sampled_buf: tuple[torch.Tensor, np.ndarray] | torch.Tensor | None = None
        self._pooled_num_buf: torch.Tensor | None = None
        self._pooled_wait_event: torch.cuda.Event | None = None
        self._fast_slot: list | None = None

        sampled_ids = sampler_output.sampled_token_ids
        if (
            pool is not None
            and not check_ep_fault
            and pending_aux_output is None
            and sampler_output.logprobs_tensors is None
            and sampler_output.num_nans is None
            and sampler_output.sampling_mask_tensors is None
            and not model_runner_output.prompt_logprobs_dict
            and not model_runner_output.prompt_token_id_logprobs_dict
            and getattr(sampler_output, "num_sampled_all_ones", False)
            and sampled_ids.ndim == 2
            and sampled_ids.shape[1] == 1
            and sampled_ids.dtype == torch.int64
            and sampled_ids.stride(0) == 1
            and (n_reqs := sampled_ids.shape[0]) <= pool.max_num_reqs
            and num_sampled_tokens.shape[0] <= pool.max_num_reqs
            and (ext := _get_v1_ext()) is not None
        ):
            self.logprobs_tensors = None
            self.num_nans = None
            self.sampling_mask_tensors = None
            self.prompt_logprobs_dict = {}
            self.prompt_token_id_logprobs_dict = {}
            self._num_sampled_all_ones = True
            self.num_sampled_tokens_np = pool.ones_np(num_sampled_tokens.shape[0])
            slot, ms_h, cs_h = pool.acquire_fast_slot(main_stream, copy_stream)
            self._fast_slot = slot
            self.copy_event = slot[0]
            entry = slot[4]
            self.sampled_token_ids = entry[1][:n_reqs]
            ext.async_copy_sampled_and_record(
                sampled_ids,
                entry[0],
                n_reqs,
                ms_h,
                cs_h,
                slot[3],
                slot[1],
            )
            return

        # Blocking (sleep) event to avoid busy-polling the CUDA driver lock.
        self.copy_event = (
            pool.acquire_event() if pool is not None else torch.cuda.Event(blocking=True)
        )

        with stream(copy_stream, main_stream):
            if pool is not None:
                wait_event = pool.acquire_wait_event()
                wait_event.record(main_stream)
                copy_stream.wait_event(wait_event)
                self._pooled_wait_event = wait_event
            else:
                copy_stream.wait_stream(main_stream)

            if pool is not None:
                self._pooled_sampled_buf, self.sampled_token_ids = (
                    pool.copy_sampled_to_np(sampled_ids)
                )
            else:
                self.sampled_token_ids = async_copy_to_np(sampled_ids)
            self.logprobs_tensors: LogprobsTensors | None = None
            if sampler_output.logprobs_tensors is not None:
                self.logprobs_tensors = (
                    sampler_output.logprobs_tensors.to_cpu_nonblocking()
                )
            self.num_nans: np.ndarray | None = None
            if sampler_output.num_nans is not None:
                self.num_nans = async_copy_to_np(sampler_output.num_nans)
            self._num_sampled_all_ones = bool(
                getattr(sampler_output, "num_sampled_all_ones", False)
                and sampled_ids.ndim == 2
                and sampled_ids.shape[1] == 1
            )
            if (
                pool is not None
                and self._num_sampled_all_ones
                and num_sampled_tokens.shape[0] <= pool.max_num_reqs
            ):
                self._pooled_num_buf = None
                self.num_sampled_tokens_np = pool.ones_np(
                    num_sampled_tokens.shape[0]
                )
            elif pool is not None:
                self._pooled_num_buf, self.num_sampled_tokens_np = pool.copy_num_to_np(
                    num_sampled_tokens
                )
            else:
                self.num_sampled_tokens_np = async_copy_to_np(num_sampled_tokens)
            self.sampling_mask_tensors: SamplingMaskTensors | None = None
            if sampler_output.sampling_mask_tensors is not None:
                self.sampling_mask_tensors = (
                    sampler_output.sampling_mask_tensors.to_cpu_nonblocking()
                )
            self.prompt_logprobs_dict = {
                k: v.to_cpu_nonblocking() if v is not None else None
                for k, v in self.model_runner_output.prompt_logprobs_dict.items()
            }
            token_id_logprobs = self.model_runner_output.prompt_token_id_logprobs_dict
            self.prompt_token_id_logprobs_dict = {
                k: v.to("cpu", non_blocking=True) for k, v in token_id_logprobs.items()
            }
            if self.pending_aux_output is not None:
                self.pending_aux_output.enqueue_cpu_copy(
                    num_sampled=self.num_sampled_tokens_np,
                    num_rejected=async_copy_to_np(sampler_output.num_rejected),
                )
            if check_ep_fault:
                has_fault = get_ep_all2all_manager().query_fault()
                self._has_fault = has_fault.to("cpu", non_blocking=True)
            self.copy_event.record(copy_stream)

    def get_output(self) -> ModelRunnerOutput:
        self.copy_event.synchronize()

        # NOTE(woosuk): The following code is to ensure compatibility with
        # the existing model runner.
        # Going forward, we should keep the data structures as NumPy arrays
        # rather than Python lists.
        sampled_token_ids: list[list[int]] = self.sampled_token_ids.tolist()
        if not self._num_sampled_all_ones:
            num_sampled_tokens: list[int] = self.num_sampled_tokens_np.tolist()
            for token_ids, num_tokens in zip(sampled_token_ids, num_sampled_tokens):
                del token_ids[num_tokens:]
        self.model_runner_output.sampled_token_ids = sampled_token_ids

        if self.sampling_mask_tensors is not None:
            self.model_runner_output.sampling_masks = (
                self.sampling_mask_tensors.tolists(self.num_sampled_tokens_np)
            )

        if self.num_nans is not None:
            self.model_runner_output.num_nans_in_logits = dict(
                zip(self.model_runner_output.req_ids, self.num_nans.tolist())
            )
            if envs.VLLM_RAISE_ON_LOGIT_NANS:
                raise_if_nan_logits(self.model_runner_output.num_nans_in_logits)

        if self.logprobs_tensors is not None:
            self.model_runner_output.logprobs = self.logprobs_tensors.tolists()
        self.model_runner_output.prompt_logprobs_dict = self.prompt_logprobs_dict
        self.model_runner_output.prompt_token_id_logprobs_dict = (
            self.prompt_token_id_logprobs_dict
        )
        if self.pending_aux_output is not None:
            self.model_runner_output.aux_output_connector_output = (
                self.pending_aux_output.process_output()
            )

        if self._has_fault is not None and self._has_fault.item():
            mask = get_ep_all2all_manager().query_active_mask()
            raise RuntimeError(
                "Fault detected in EP all2all communication: "
                "one or more ranks timed out during dispatch/combine. "
                f"Mask: {mask.cpu().tolist()}"
            )

        if self._pool is not None:
            pool = self._pool
            self._pool = None
            fast_slot = self._fast_slot
            if fast_slot is not None:
                self._fast_slot = None
                pool.release_fast_slot(fast_slot)
            else:
                sampled_buf = self._pooled_sampled_buf
                num_buf = self._pooled_num_buf
                wait_event = self._pooled_wait_event
                self._pooled_sampled_buf = None
                self._pooled_num_buf = None
                self._pooled_wait_event = None
                pool.release(self.copy_event, sampled_buf, num_buf, wait_event=wait_event)

        return self.model_runner_output

    def get_kv_connector_output(self) -> KVConnectorOutput | None:
        """Get the KV connector output for this async output."""
        return self.model_runner_output.kv_connector_output


class AsyncPoolingOutput(AsyncModelRunnerOutput):
    def __init__(
        self,
        model_runner_output: ModelRunnerOutput,
        pooler_output: PoolerOutput,
        finished_mask: list[bool],
        main_stream: torch.cuda.Stream,
        copy_stream: torch.cuda.Stream,
    ):
        self.model_runner_output = model_runner_output
        self.pooler_output = pooler_output
        # Blocking (sleep) event to avoid busy-polling the CUDA driver lock.
        self.copy_event = torch.cuda.Event(blocking=True)

        with stream(copy_stream, main_stream):
            copy_stream.wait_stream(main_stream)
            if isinstance(self.pooler_output, torch.Tensor) and all(finished_mask):
                self.pooler_output_cpu: PoolerOutput = self.pooler_output.to(
                    "cpu", non_blocking=True
                )
            else:
                outputs = (
                    self.pooler_output.unbind()
                    if isinstance(self.pooler_output, torch.Tensor)
                    else self.pooler_output
                )
                self.pooler_output_cpu = [
                    None
                    if output is None or not is_finished
                    else output.to("cpu", non_blocking=True)
                    for output, is_finished in zip(outputs, finished_mask, strict=True)
                ]
            self.copy_event.record(copy_stream)

    def get_output(self) -> ModelRunnerOutput:
        if isinstance(self.pooler_output_cpu, torch.Tensor):
            pooler_output = list(self.pooler_output_cpu.unbind(dim=0))
        else:
            pooler_output = self.pooler_output_cpu
        self.copy_event.synchronize()
        self.model_runner_output.pooler_output = pooler_output
        return self.model_runner_output

    def get_kv_connector_output(self) -> KVConnectorOutput | None:
        """Get the KV connector output for this async output."""
        return self.model_runner_output.kv_connector_output


def async_copy_to_np(x: torch.Tensor) -> np.ndarray:
    return x.to("cpu", non_blocking=True, copy=True).numpy()


@contextlib.contextmanager
def stream(to_stream: torch.cuda.Stream, from_stream: torch.cuda.Stream):
    """Lightweight version of torch.cuda.stream() context manager which
    avoids current_stream and device lookups.
    """
    try:
        torch.cuda.set_stream(to_stream)
        yield
    finally:
        torch.cuda.set_stream(from_stream)
