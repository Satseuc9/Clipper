"""
Processing queue: finished segments -> worker processes -> clips.

- Workers are separate processes (ProcessPoolExecutor), so heavy analysis
  can never block the recorder threads or the monitor loop.
- Segments of one session are processed strictly in order (each segment
  uses the previous one as context and dedupes against its clips);
  different streamers are processed in parallel up to ANALYSIS_WORKERS.
- State lives in the segment sidecars, so the queue survives restarts:
  "queued" -> "processing" -> "done" | "failed" (after PROCESSING_MAX_ATTEMPTS),
  with PROCESSING_RETRY_DELAY_SECONDS between attempts.
- A crashed worker process (BrokenProcessPool) is replaced automatically.
"""

import logging
import multiprocessing
import os
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool

from clipping_engine import config
from clipping_engine.capture.segment_manager import recover_interrupted
from clipping_engine.processing import config_snapshot, process_segment
from clipping_engine.storage import clip_storage, recording_storage as storage

log = logging.getLogger(__name__)
CLEANUP_EVERY = 600


def _performance_cores():
    """Performance-core count on Apple silicon Macs, else None."""
    if sys.platform != "darwin":
        return None
    try:
        out = subprocess.run(["sysctl", "-n", "hw.perflevel0.physicalcpu"], capture_output=True,
                             text=True, timeout=5)
        return int(out.stdout.strip()) if out.returncode == 0 else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def resolve_workers(setting=None):
    """ANALYSIS_WORKERS as a number ("auto" -> based on this computer's CPU)."""
    setting = config.ANALYSIS_WORKERS if setting is None else setting
    if isinstance(setting, int) and setting > 0:
        return setting
    perf = _performance_cores()
    if perf:
        return max(1, min(3, perf // 2))
    return max(1, min(3, (os.cpu_count() or 4) // 4))


def _worker_init(snapshot):
    signal.signal(signal.SIGINT, signal.SIG_IGN)     # CTRL+C is handled by the main process
    from clipping_engine.processing import apply_snapshot
    apply_snapshot(snapshot)
    if config.ANALYSIS_NICENESS and hasattr(os, "nice"):
        try:
            os.nice(int(config.ANALYSIS_NICENESS))     # FFmpeg/Whisper children inherit it
        except OSError:
            pass
    try:
        logs = config.ENGINE_DIR / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(level=logging.INFO, filename=str(logs / "worker.log"),
                            format="%(asctime)s %(process)d %(name)s %(levelname)s %(message)s")
    except OSError:
        pass                      # logging must never stop a worker from starting


class Pipeline:
    def __init__(self, workers=None, process_fn=None, use_processes=True):
        self.workers = workers or resolve_workers()
        self.process_fn = process_fn or process_segment
        self.use_processes = use_processes
        self.executor = None
        self.inflight = {}                 # (session, index) -> future
        self.events = deque(maxlen=8)      # recent outcomes for the status display
        self.clips_sent = 0
        self._finished_sessions = set()    # fully processed, ended sessions (skipped when scanning)
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread = None
        self._last_cleanup = 0.0

    # ---- lifecycle -------------------------------------------------------------------------

    def start(self, active_sessions=()):
        queued = recover_interrupted(active_sessions)
        if queued:
            self._event(f"recovered {queued} segment(s) from a previous run")
        self._thread = threading.Thread(target=self._loop, name="pipeline", daemon=True)
        self._thread.start()

    def notify(self, *_):
        self._wake.set()

    def stop(self, wait=False):
        """Stop the queue. Unfinished segments stay "processing" and are re-queued on the next start."""
        self._stop.set()
        self._wake.set()
        if self._thread:
            self._thread.join(10)
        if self.executor:
            if not wait:
                for proc in list(getattr(self.executor, "_processes", {}).values()):
                    proc.terminate()
            self.executor.shutdown(wait=wait, cancel_futures=True)
            self.executor = None

    def drain(self, timeout=None):
        """Wait until every segment is finished, retries included (mock mode and tests)."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            unfinished = any(side.get("status") not in storage.FINISHED
                             for session in storage.list_sessions()
                             for side in storage.session_segments(session))
            if not self.inflight and not unfinished:
                return True
            if deadline and time.monotonic() > deadline:
                return False
            self._wake.set()
            time.sleep(0.5)

    # ---- loop ----------------------------------------------------------------------------------

    def _executor(self):
        if self.executor is None:
            ctx = multiprocessing.get_context("spawn")
            self.executor = ProcessPoolExecutor(max_workers=self.workers, mp_context=ctx,
                                                initializer=_worker_init, initargs=(config_snapshot(),))
        return self.executor

    def _loop(self):
        while not self._stop.is_set():
            try:
                self._collect()
                self._dispatch()
                if time.monotonic() - self._last_cleanup > CLEANUP_EVERY:
                    self._last_cleanup = time.monotonic()
                    storage.cleanup_recordings()
                    clip_storage.cleanup_clips()
            except Exception:
                log.exception("pipeline loop error")
            self._wake.wait(2)
            self._wake.clear()

    def _pending(self):
        """Segments ready to process now, respecting per-session order and retry delays."""
        now = time.time()
        ready = []
        for session in storage.list_sessions():
            if session in self._finished_sessions:
                continue
            if storage.session_fully_processed(session):
                self._finished_sessions.add(session)
                continue
            for side in storage.session_segments(session):
                status = side.get("status")
                if status in storage.FINISHED:
                    continue
                key = (str(session), side["index"])
                if status == "queued" and key not in self.inflight and now >= side.get("retry_at", 0):
                    ready.append((session, side))
                break          # strict order: nothing after an unfinished segment
        return ready

    def _dispatch(self):
        for session, side in self._pending():
            if len(self.inflight) >= self.workers:
                return
            path = storage.sidecar_for(storage.segment_file(session, side["index"]))
            storage.update_json(path, status="processing", attempts=side.get("attempts", 0) + 1,
                                started=time.time())
            future = self._submit(str(session), side["index"])
            self.inflight[(str(session), side["index"])] = future

    def _submit(self, session, index):
        if self.use_processes:
            return self._executor().submit(self.process_fn, session, index, config_snapshot())
        return _InlineFuture(self.process_fn, session, index, config_snapshot())

    def _collect(self):
        for key, future in list(self.inflight.items()):
            if not future.done():
                continue
            del self.inflight[key]
            session, index = key
            path = storage.sidecar_for(storage.segment_file(session, index))
            side = storage.read_json(path, {}) or {}
            label = "/".join(Path(session).parts[-3:]) + f"/segment_{index:04d}"
            try:
                result = future.result()
            except BrokenProcessPool:
                self.executor = None                # replace the crashed pool
                self._retry(path, side, "worker process crashed", label)
                continue
            except Exception as exc:
                self._retry(path, side, f"{type(exc).__name__}: {str(exc)[:200]}", label)
                continue
            storage.update_json(path, status="done", finished=time.time(), error=None, result=result)
            self.clips_sent += result.get("sent", 0)
            self._event(f"{label}: {describe(result)}")

    def _retry(self, path, side, error, label):
        attempts = side.get("attempts", 1)
        if attempts >= config.PROCESSING_MAX_ATTEMPTS:
            storage.update_json(path, status="failed", error=error, finished=time.time())
            self._event(f"{label}: FAILED after {attempts} attempts ({error})")
            log.error("%s failed permanently: %s", label, error)
        else:
            storage.update_json(path, status="queued", error=error,
                                retry_at=time.time() + config.PROCESSING_RETRY_DELAY_SECONDS)
            self._event(f"{label}: error, retrying ({error})")
            log.warning("%s attempt %s failed: %s", label, attempts, error)

    def _event(self, text):
        self.events.append(f"[{time.strftime('%H:%M:%S')}] {text}")

    def backlog(self):
        """(segments waiting, GB waiting) across all sessions."""
        count, size = 0, 0
        for session in storage.list_sessions():
            if session in self._finished_sessions:
                continue
            for side in storage.session_segments(session):
                if side.get("status") in ("queued", "processing"):
                    count += 1
                    size += side.get("size", 0)
        return count, size / 1024 ** 3

    def status_lines(self):
        waiting, gb = self.backlog()
        workers = f"{self.workers} worker{'' if self.workers == 1 else 's'}"
        lines = [f"  ⚙ analysis ({workers}): {len(self.inflight)} running, {waiting} waiting "
                 f"({gb:.1f} GB), {self.clips_sent} clip(s) sent"]
        if waiting > max(3, 2 * self.workers):
            lines.append(f"  ⚠ analysis is falling behind ({self.workers} worker(s)): raise ANALYSIS_WORKERS, "
                         f"record fewer streamers or use a lower STREAM_QUALITY")
        lines += [f"    {e}" for e in list(self.events)[-5:]]      # compact: last 5 results
        return lines


def describe(result):
    """One-line summary of a processed segment, including which signals it had."""
    best = result.get("best_score")
    text = f"{result.get('candidates', 0)} moments, best {best if best is not None else '-'}, " \
           f"{len(result.get('clips', []))} clip(s)"
    signals = result.get("signals") or {}
    have = [k for k in ("audio", "chat", "video") if signals.get(k)]
    have += [k for k in ("speech", "ai") if result.get(k)]
    missing = [k for k in ("chat", "speech", "ai") if k not in have]
    text += f" [{'+'.join(have) or 'no signals'}" + (f"; no {', no '.join(missing)}" if missing else "") + "]"
    problems = [n for n in result.get("notes") or [] if "unavailable" in n or "failed" in n]
    if problems:
        text += f" ⚠ {problems[0][:90]}"
    return text


class _InlineFuture:
    """Runs a function synchronously; used by tests to inject failures."""

    def __init__(self, fn, *args):
        try:
            self._result, self._exc = fn(*args), None
        except BaseException as exc:      # noqa: BLE001 - re-raised from result()
            self._result, self._exc = None, exc

    def done(self):
        return True

    def result(self):
        if self._exc:
            raise self._exc
        return self._result
