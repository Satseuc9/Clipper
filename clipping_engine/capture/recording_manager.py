"""
Recording manager: one Streamlink -> FFmpeg pipeline per live streamer.

- start when the monitor reports LIVE, stop when it reports OFFLINE
  (UNKNOWN never stops a recording)
- at most one recorder per streamer, at most MAX_CONCURRENT_RECORDINGS total
- reconnects with back-off when the connection drops or stalls; segment
  numbering continues in the same session folder
- runs in its own threads and child processes, so analysis (a separate
  process pool) can never block or interrupt capture
"""

import logging
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from clipping_engine import config
from clipping_engine.capture import segment_manager
from clipping_engine.capture.streamlink_manager import (
    can_handle_url, ffmpeg_segment_command, platform_support, streamlink_command,
)
from clipping_engine.storage.recording_storage import (
    list_segment_files, new_session_dir, read_json, segment_index, update_json, write_json,
)

log = logging.getLogger(__name__)


@dataclass
class StreamTarget:
    platform: str
    name: str                    # as written in stream-monitor/config.py
    url: str                     # what Streamlink opens
    display_name: str = ""
    title: str | None = None
    extra: dict = field(default_factory=dict)   # platform IDs from the monitor

    @property
    def key(self):
        return (self.platform, self.name.lower())


def _popen_group_kwargs():
    """Own process group, so CTRL+C reaches us first and we stop children in order."""
    if sys.platform == "win32":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def _stop_process(proc, timeout):
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(5)


class StreamlinkLog:
    """
    Follows Streamlink's log for its ad-filter state. During Twitch ad breaks
    Streamlink drops the ad segments and deliberately writes nothing until
    the ad ends ("Filtering out segments and pausing stream output" ...
    "Resuming stream output"), so no data then is not a stall.
    """

    PAUSE = b"Filtering out segments and pausing stream output"
    RESUME = b"Resuming stream output"

    def __init__(self, path):
        self.path = path
        self.pos = 0
        self.paused_since = None
        self.ad_breaks = 0

    def poll(self):
        try:
            with open(self.path, "rb") as fh:
                fh.seek(self.pos)
                chunk = fh.read()
        except OSError:
            return
        cut = chunk.rfind(b"\n") + 1
        self.pos += cut
        for line in chunk[:cut].splitlines():
            if self.PAUSE in line and self.paused_since is None:
                self.paused_since = time.monotonic()
                self.ad_breaks += 1
            elif self.RESUME in line:
                self.paused_since = None

    @property
    def paused(self):
        return self.paused_since is not None


def free_gb(path):
    path = Path(path)
    while not path.exists() and path != path.parent:
        path = path.parent
    return shutil.disk_usage(path).free / 1024 ** 3


class Recorder(threading.Thread):
    def __init__(self, target, on_segment=None):
        super().__init__(name=f"recorder-{target.platform}-{target.name}", daemon=True)
        self.target = target
        self.on_segment = on_segment
        self.session_dir = new_session_dir(target.platform, target.name, {
            "platform": target.platform, "streamer": target.name,
            "display_name": target.display_name or target.name, "url": target.url,
            "title": target.title, "extra": target.extra,
        })
        self._stop_event = threading.Event()
        self.state = "starting"
        self.last_error = None
        self.segments = 0
        self.run_number = 0
        self.ad_breaks = 0
        self._procs = ()
        self._run_had_data = False

    # ---- control ---------------------------------------------------------------------

    def stop(self):
        self._stop_event.set()

    @property
    def stopping(self):
        return self._stop_event.is_set()

    def kill(self):
        """Hard stop (second CTRL+C): kill the capture processes immediately."""
        self._stop_event.set()
        for proc in self._procs:
            if proc.poll() is None:
                proc.kill()

    # ---- main loop -------------------------------------------------------------------

    def run(self):
        delays = config.RECONNECT_DELAYS_SECONDS or [30]
        attempt = 0
        failing_since = None
        final_state = "stopped"
        try:
            while not self._stop_event.is_set():
                self.run_number += 1
                started = time.time()
                self._run_had_data = False
                outcome = self._record_run(self.run_number)
                if self._stop_event.is_set():
                    break
                if outcome.startswith("disk almost full"):
                    self.last_error, final_state = outcome, "stopped: disk almost full"
                    break
                if time.time() - started > 300:
                    attempt = 0          # a healthy run resets the back-off
                if self._run_had_data:
                    failing_since = None
                else:
                    failing_since = failing_since or started
                    if time.time() - failing_since > config.RECORDING_GIVE_UP_MINUTES * 60:
                        self.last_error = f"gave up after {config.RECORDING_GIVE_UP_MINUTES} min without data ({outcome})"
                        final_state = "gave up"
                        log.warning("%s: %s", self.name, self.last_error)
                        break
                delay = delays[min(attempt, len(delays) - 1)]
                attempt += 1
                self.state = "reconnecting"
                self.last_error = outcome
                log.warning("%s: %s; reconnecting in %ss", self.name, outcome, delay)
                self._stop_event.wait(delay)
        except Exception:
            log.exception("%s crashed", self.name)
            self.last_error = "recorder crashed (see engine.log)"
        finally:
            update_json(self.session_dir / "session.json", status="ended", ended=time.time())
            self.state = final_state

    def _record_run(self, run):
        files = list_segment_files(self.session_dir)
        start_number = (segment_index(files[-1]) + 1) if files else 1
        logs = self.session_dir / "logs"
        meta_path = self.session_dir / "session.json"
        meta = read_json(meta_path, {}) or {}
        meta.setdefault("runs", []).append({"run": run, "started": time.time(), "first_segment": start_number})
        write_json(meta_path, meta)

        with open(logs / f"run{run:02d}_streamlink.log", "ab") as sl_log, \
             open(logs / f"run{run:02d}_ffmpeg.log", "ab") as ff_log:
            streamlink = subprocess.Popen(
                streamlink_command(self.target.url), stdout=subprocess.PIPE, stderr=sl_log,
                stdin=subprocess.DEVNULL, **_popen_group_kwargs())
            progress = logs / f"run{run:02d}_progress.txt"
            ffmpeg = subprocess.Popen(
                ffmpeg_segment_command(self.session_dir, start_number, progress), stdin=streamlink.stdout,
                stdout=subprocess.DEVNULL, stderr=ff_log, **_popen_group_kwargs())
            self._progress = segment_manager.ProgressLogger(self.session_dir, run, progress)
            self._sl_log = StreamlinkLog(logs / f"run{run:02d}_streamlink.log")
            streamlink.stdout.close()   # ffmpeg owns the read end now
            self._procs = (streamlink, ffmpeg)
            self.state = "recording"
            outcome = self._watch(run, start_number, streamlink, ffmpeg)
            _stop_process(streamlink, 10)
            try:
                ffmpeg.wait(30)          # EOF on stdin: ffmpeg finishes the last segment
            except subprocess.TimeoutExpired:
                _stop_process(ffmpeg, 15)
            self._progress.poll()

        # Everything this run wrote is finished now; the newest file is the run's last.
        remaining = [f for f in list_segment_files(self.session_dir)
                     if segment_index(f) >= start_number and not f.with_suffix(".json").exists()]
        for f in remaining:
            if f.stat().st_size == 0:
                f.unlink()
                continue
            self._complete(f, run, final=(f == remaining[-1]))
        return outcome

    def _watch(self, run, start_number, streamlink, ffmpeg):
        last_size, last_growth = -1, time.monotonic()
        last_disk_check = 0.0
        while True:
            if self._stop_event.wait(0.5):
                return "stopped"
            self._progress.poll()
            self._sl_log.poll()
            files = [f for f in list_segment_files(self.session_dir) if segment_index(f) >= start_number]
            # Every file except the newest one is closed and complete.
            for f in files[:-1]:
                if not f.with_suffix(".json").exists():
                    self._complete(f, run, final=False)
            if files:
                try:
                    size = files[-1].stat().st_size
                except FileNotFoundError:
                    size = last_size
                if size != last_size:
                    last_size, last_growth = size, time.monotonic()
                    if size > 0:
                        self._run_had_data = True
            now = time.monotonic()
            if self._sl_log.paused:
                # Ad break: Streamlink holds output on purpose. Not a stall,
                # unless it lasts far longer than any real ad break.
                self.state = "recording (ad break)"
                self.ad_breaks = self._sl_log.ad_breaks
                if now - self._sl_log.paused_since > config.MAX_AD_BREAK_SECONDS:
                    return f"ad break longer than {config.MAX_AD_BREAK_SECONDS}s"
                last_growth = now
            elif self.state != "recording":
                self.state = "recording"
            if now - last_growth > config.STALL_RESTART_SECONDS:
                return "stream stalled (no data)"
            if now - last_disk_check > 30:
                last_disk_check = now
                free = free_gb(config.RECORDINGS_DIR)
                if free < config.MIN_FREE_DISK_GB / 2:
                    return f"disk almost full ({free:.1f} GB free)"
            if streamlink.poll() is not None:
                return f"streamlink exited (code {streamlink.returncode})"
            if ffmpeg.poll() is not None:
                return f"ffmpeg exited (code {ffmpeg.returncode})"

    def _complete(self, path, run, final):
        side = segment_manager.complete_segment(self.session_dir, path, run, final)
        self.segments += 1
        if self.on_segment:
            try:
                self.on_segment(self.session_dir, side)
            except Exception:
                log.exception("segment callback failed")


class RecordingManager:
    def __init__(self, on_segment=None, chat_manager=None):
        self.on_segment = on_segment
        self.chat_manager = chat_manager
        self.recorders = {}
        self.messages = {}          # key -> last reason a recording could not start
        self.cooldown = {}          # key -> (until monotonic time, reason) after a recorder gave up
        self.support = platform_support()
        self._lock = threading.Lock()

    def active(self):
        return [r for r in self.recorders.values() if r.is_alive() and not r.stopping]

    def ensure_recording(self, target):
        with self._lock:
            rec = self.recorders.get(target.key)
            if rec and rec.is_alive():
                return "finishing previous session" if rec.stopping else "recording"
            ok, reason = self.support.get(target.platform, (False, "platform not enabled for clipping"))
            if not ok:
                return self._refuse(target, reason)
            until, why = self.cooldown.get(target.key, (0, ""))
            if time.monotonic() < until:
                return self._refuse(target, f"{why}; retrying in {int((until - time.monotonic()) / 60) + 1} min")
            free = free_gb(config.RECORDINGS_DIR)
            if free < config.MIN_FREE_DISK_GB:
                return self._refuse(target, f"only {free:.1f} GB free (MIN_FREE_DISK_GB={config.MIN_FREE_DISK_GB})")
            if len(self.active()) >= config.MAX_CONCURRENT_RECORDINGS:
                return self._refuse(target, f"MAX_CONCURRENT_RECORDINGS ({config.MAX_CONCURRENT_RECORDINGS}) reached")
            if not can_handle_url(target.url):
                return self._refuse(target, f"Streamlink cannot open {target.url}")
            rec = Recorder(target, on_segment=self.on_segment)
            self.recorders[target.key] = rec
            self.messages.pop(target.key, None)
            rec.start()
            log.info("recording started: %s -> %s", target.url, rec.session_dir)
        if self.chat_manager:
            self.chat_manager.start(target, rec.session_dir)
        return "started"

    def _refuse(self, target, reason):
        if self.messages.get(target.key) != reason:
            log.warning("not recording %s/%s: %s", target.platform, target.name, reason)
        self.messages[target.key] = reason
        return reason

    def stop_recording(self, platform, name):
        key = (platform, name.lower())
        rec = self.recorders.get(key)
        if rec and rec.is_alive() and not rec.stopping:
            rec.stop()
            log.info("recording stopping: %s/%s", platform, name)
            if self.chat_manager:
                self.chat_manager.stop(rec.target)
            return True
        return False

    def reap(self):
        """Forget recorders that have fully stopped (and cool down ones that gave up)."""
        with self._lock:
            for key in [k for k, r in self.recorders.items() if not r.is_alive()]:
                rec = self.recorders.pop(key)
                if rec.state == "gave up":
                    self.cooldown[key] = (time.monotonic() + config.RECORDING_RETRY_COOLDOWN_MINUTES * 60,
                                          rec.last_error)
                if self.chat_manager:
                    self.chat_manager.stop(rec.target)

    def stop_unmonitored(self, monitored_keys):
        """Stop recordings of streamers the monitor no longer reports (removed from its config)."""
        stopped = []
        for key, rec in list(self.recorders.items()):
            if key not in monitored_keys and rec.is_alive() and not rec.stopping:
                self.stop_recording(*key)
                stopped.append(key)
        return stopped

    def stop_all(self, timeout=60):
        for rec in list(self.recorders.values()):
            if rec.is_alive():
                rec.stop()
                if self.chat_manager:
                    self.chat_manager.stop(rec.target)
        deadline = time.monotonic() + timeout
        for rec in list(self.recorders.values()):
            rec.join(max(0.1, deadline - time.monotonic()))

    def kill_all(self):
        for rec in list(self.recorders.values()):
            rec.kill()

    def status_lines(self):
        lines = []
        for rec in self.recorders.values():
            if rec.is_alive():
                state = "stopping" if rec.stopping else rec.state
                err = f" ({rec.last_error})" if rec.state == "reconnecting" and rec.last_error else ""
                ads = f", {rec.ad_breaks} ad break(s) skipped" if rec.ad_breaks else ""
                lines.append(f"  ● {rec.target.platform}/{rec.target.name}: {state}{err}, "
                             f"{rec.session_dir.name}, {rec.segments} segment(s){ads}")
        for (platform, name), reason in self.messages.items():
            if (platform, name) not in self.recorders:
                lines.append(f"  ○ {platform}/{name}: not recording - {reason}")
        return lines
