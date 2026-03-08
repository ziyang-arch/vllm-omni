"""
Co-Location Aware DVFS Scheduler for vllm-omni.

Controls GPU clock frequencies based on how many pipeline stages are
actively computing on a shared GPU.  Uses a cross-process shared counter
(multiprocessing.Value) so that stages running in separate OS processes
can coordinate without heavyweight IPC.

All pynvml / NVTX calls are wrapped defensively so that a failure in the
DVFS path never crashes the inference pipeline.
"""

import ctypes
import logging
import multiprocessing
import os
import time as _time
from typing import Optional

logger = logging.getLogger(__name__)

_nvml_available = False
_nvtx_available = False

try:
    import pynvml

    _nvml_available = True
except ImportError:
    pynvml = None  # type: ignore[assignment]

try:
    import nvtx  # cuda-python / nvtx package

    _nvtx_available = True
except ImportError:
    nvtx = None  # type: ignore[assignment]


def get_gpu_freq_bounds(gpu_id: int = 0) -> tuple[int, int]:
    """Auto-discover the min and max SM clock frequencies for *gpu_id*.

    Returns (min_freq_mhz, max_freq_mhz).  Falls back to conservative
    Ada Lovelace defaults (210, 2520) on any error.
    """
    default_min, default_max = 210, 2520
    if not _nvml_available:
        return default_min, default_max
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        max_freq = pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)

        # Supported graphics clocks are returned per memory-clock level.
        # We pick the lowest supported graphics clock across all mem clocks.
        mem_clocks = pynvml.nvmlDeviceGetSupportedMemoryClocks(handle)
        min_freq = max_freq
        for mc in mem_clocks:
            try:
                gfx_clocks = pynvml.nvmlDeviceGetSupportedGraphicsClocks(handle, mc)
                if gfx_clocks:
                    min_freq = min(min_freq, min(gfx_clocks))
            except pynvml.NVMLError:
                continue

        return int(min_freq), int(max_freq)
    except Exception as exc:
        logger.warning("get_gpu_freq_bounds failed for GPU %d: %s; using defaults", gpu_id, exc)
        return default_min, default_max


class CoLocationDVFSScheduler:
    """Cross-process GPU clock scheduler driven by an active-stage counter.

    Parameters
    ----------
    gpu_id : int
        CUDA device index whose clocks will be controlled.
    min_freq, max_freq : int
        Locked clock bounds in MHz.
    shared_counter : multiprocessing.Value
        ``ctypes.c_int`` Value shared across all stage processes on this GPU.
    shared_lock : multiprocessing.Lock
        Lock protecting *shared_counter*.
    """

    def __init__(
        self,
        gpu_id: int,
        min_freq: int,
        max_freq: int,
        shared_counter: "multiprocessing.Value[ctypes.c_int]",
        shared_lock: multiprocessing.Lock,
        stage_id: int = -1,
    ):
        self.gpu_id = gpu_id
        self.stage_id = stage_id
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.active_stages = shared_counter
        self.lock = shared_lock

        self._handle: Optional[object] = None
        self._nvml_ok = False
        self._idle_nvtx_id: Optional[int] = None

        if _nvml_available:
            try:
                pynvml.nvmlInit()
                self._handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                self._nvml_ok = True
                logger.info(
                    "[DVFS] Scheduler initialised for GPU %d  (freq range %d–%d MHz)",
                    gpu_id,
                    min_freq,
                    max_freq,
                )
            except Exception as exc:
                logger.warning("[DVFS] pynvml init failed for GPU %d: %s — running in no-op mode", gpu_id, exc)
        else:
            logger.warning("[DVFS] pynvml not installed — running in no-op mode")

    # ------------------------------------------------------------------
    # Public API (called from the stage worker hot path)
    # ------------------------------------------------------------------

    def stage_sleep(self) -> None:
        """Call **before** the stage enters a blocking IPC wait."""
        ts = _time.time()
        if _nvtx_available:
            nvtx.range_push("dvfs_idle")

        with self.lock:
            self.active_stages.value -= 1
            count = self.active_stages.value
        logger.info(
            "[DVFS] Stage-%d GPU-%d SLEEP  t=%.6f  active_after=%d",
            self.stage_id, self.gpu_id, ts, count,
        )
        if count == 0:
            self._set_clocks(self.min_freq)

    def stage_wakeup(self) -> None:
        """Call **after** the stage exits a blocking IPC wait."""
        ts = _time.time()
        with self.lock:
            self.active_stages.value += 1
            count = self.active_stages.value
        logger.info(
            "[DVFS] Stage-%d GPU-%d WAKEUP t=%.6f  active_after=%d",
            self.stage_id, self.gpu_id, ts, count,
        )
        if count == 1:
            self._set_clocks(self.max_freq)

        if _nvtx_available:
            nvtx.range_pop()

    def cleanup(self) -> None:
        """Reset GPU clocks to driver defaults.  Safe to call multiple times."""
        if not self._nvml_ok:
            return
        try:
            pynvml.nvmlDeviceResetGpuLockedClocks(self._handle)
            logger.info("[DVFS] GPU %d clocks reset to defaults", self.gpu_id)
        except Exception as exc:
            logger.warning("[DVFS] Failed to reset clocks on GPU %d: %s", self.gpu_id, exc)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _set_clocks(self, freq_mhz: int) -> None:
        if not self._nvml_ok:
            return
        try:
            pynvml.nvmlDeviceSetGpuLockedClocks(self._handle, freq_mhz, freq_mhz)
            logger.info(
                "[DVFS] GPU %d clocks -> %d MHz  (ok)",
                self.gpu_id, freq_mhz,
            )
        except Exception as exc:
            logger.warning(
                "[DVFS] GPU %d clocks -> %d MHz  FAILED: %s",
                self.gpu_id, freq_mhz, exc,
            )


# ------------------------------------------------------------------
# Factory — called once per GPU in the orchestrator (parent process)
# ------------------------------------------------------------------

_gpu_shared_state: dict[int, dict] = {}


def get_or_create_dvfs_shared(gpu_id: int, ctx: Optional[multiprocessing.context.BaseContext] = None) -> dict:
    """Return ``{"counter": Value, "lock": Lock}`` for *gpu_id*.

    Creates the shared objects on first call for a given GPU.  Subsequent
    calls return the same objects so that all stages on the same GPU share
    one counter.

    Parameters
    ----------
    gpu_id : int
        Physical GPU index.
    ctx : multiprocessing context, optional
        The spawn context used by the orchestrator.  Falls back to
        ``multiprocessing.get_context("spawn")``.
    """
    if gpu_id in _gpu_shared_state:
        return _gpu_shared_state[gpu_id]

    if ctx is None:
        ctx = multiprocessing.get_context("spawn")

    shared = {
        "counter": ctx.Value(ctypes.c_int, 0),
        "lock": ctx.Lock(),
    }
    _gpu_shared_state[gpu_id] = shared
    logger.info("[DVFS] Created shared counter for GPU %d", gpu_id)
    return shared
