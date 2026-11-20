"""
Connects the clipping engine to the existing stream-monitor.

The monitor stays the single source of truth for WHO is monitored
(stream-monitor/config.py STREAMERS) and WHO is live. Its run_monitor()
loop calls our hook after every cycle; we turn the displayed rows into
recording decisions:

  fresh LIVE     -> start recording (no-op if already recording)
  fresh OFFLINE  -> stop recording
  UNKNOWN / a stale row kept from memory after an API error -> no change
"""

import sys

from clipping_engine import config

MONITOR_DIR = config.ENGINE_DIR.parent / "stream-monitor"


def load_monitor():
    """Import stream-monitor's modules (it is a folder of scripts, not a package)."""
    if str(MONITOR_DIR) not in sys.path:
        sys.path.insert(0, str(MONITOR_DIR))
    import main as monitor_main            # noqa: E402  (also loads stream-monitor/.env)
    import config as monitor_config        # noqa: E402
    from utils.tracker import is_stale     # noqa: E402
    return monitor_main, monitor_config, is_stale


def selected(platform, name):
    if platform not in config.CLIP_PLATFORMS:
        return False
    if config.CLIP_ONLY is None:
        return True
    return name.lower() in {n.lower() for n in config.CLIP_ONLY.get(platform, [])}


def targets_from_rows(sections, is_stale):
    """(to_start [StreamTarget], to_stop [(platform, name)]) from the monitor's rows."""
    from clipping_engine.capture.recording_manager import StreamTarget
    start, stop = [], []
    for _, _, rows in sections:
        for row in rows:
            if not selected(row.platform, row.name) or is_stale(row):
                continue
            if row.status == "LIVE" and row.url:
                start.append(StreamTarget(row.platform, row.name, row.url, display_name=row.label,
                                          title=row.title, extra=dict(row.extra or {})))
            elif row.status == "OFFLINE":
                stop.append((row.platform, row.name))
    return start, stop


def make_hook(recorder, chat_manager, pipeline, is_stale):
    def on_cycle(sections, changes):
        to_start, to_stop = targets_from_rows(sections, is_stale)
        for platform, name in to_stop:
            recorder.stop_recording(platform, name)
        for target in to_start:
            recorder.ensure_recording(target)
        recorder.reap()
        lines = ["", "=" * 60, "CLIPPING ENGINE"]
        rec = recorder.status_lines()
        lines += rec or ["  (no live streamers being recorded)"]
        lines += chat_manager.status_lines() if chat_manager else []
        lines += pipeline.status_lines()
        return lines
    return on_cycle
