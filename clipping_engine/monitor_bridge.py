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

import re
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
        monitored = {(row.platform, row.name.lower()) for _, _, rows in sections for row in rows
                     if selected(row.platform, row.name)}
        recorder.stop_unmonitored(monitored)            # removed from stream-monitor/config.py
        for platform, name in to_stop:
            recorder.stop_recording(platform, name)
        for target in to_start:
            recorder.ensure_recording(target)
        recorder.reap()
        rec = recorder.status_lines()
        lines = rec or ["  (no live streamers being recorded)"]
        lines += chat_manager.status_lines() if chat_manager else []
        lines += pipeline.status_lines()
        return engine_section(lines)
    return on_cycle


# ---------------------------------------------------------------- display

_REC = re.compile(r"^(\s*)● ([^:]+): ([^,]+)(.*)$")
_IDLE = re.compile(r"^(\s*)○ ([^:]+): (.*)$")
_EVENT = re.compile(r"^(\s*)(\[\d\d:\d\d:\d\d\])(.*)$")


def _ui():
    load_monitor()                         # makes stream-monitor's utils importable
    from utils import terminal             # noqa: E402
    return terminal


def engine_section(lines):
    """Heading + status lines, colored with the monitor's palette (plain text when color is off)."""
    ui = _ui()
    return ["", ui.rule(heavy=True, label="CLIPPING ENGINE")] + [style_line(line, ui) for line in lines]


def style_line(line, ui):
    p = ui.paint
    m = _REC.match(line)
    if m:                                                    # a recording
        indent, who, state, rest = m.groups()
        role = ("warn" if state.startswith("reconnecting") else
                "info" if "ad break" in state or state.startswith("starting") else
                "dim" if state.startswith("stopping") else "live")
        return f"{indent}{p('●', 'rec')} {p(who, bold=True)}: {p(state, role)}{p(rest, 'dim')}"
    m = _IDLE.match(line)
    if m:                                                    # not recording, with the reason
        indent, who, reason = m.groups()
        return f"{indent}{p('○', 'offline')} {who}: {p(reason, 'warn')}"
    stripped = line.strip()
    if stripped.startswith("💬"):
        return p(line, "unknown") if "no chat" in stripped else line
    if stripped.startswith("⚙"):                              # analysis queue + clips sent
        line = re.sub(r"(\d+ clip\(s\) sent)", lambda x: p(x.group(1), "title", bold=True), line)
        line = re.sub(r"(\d+ running|\d+ waiting \([\d.]+ GB\))", lambda x: p(x.group(1), "info"), line)
        return line.replace("⚙ analysis", p("⚙ analysis", "heading", bold=True), 1)
    if stripped.startswith("⚠"):
        return p(line, "warn")
    m = _EVENT.match(line)
    if m:                                                    # recent processing results
        indent, stamp, text = m.groups()
        if "FAILED" in text:
            return f"{indent}{p(stamp, 'dim')}{p(text, 'error')}"
        if "error, retrying" in text:
            return f"{indent}{p(stamp, 'dim')}{p(text, 'warn')}"
        text, _, problem = text.partition(" ⚠ ")
        text = re.sub(r"\b([1-9]\d* clip\(s\))", lambda x: p(x.group(1), "title", bold=True), text)
        return f"{indent}{p(stamp, 'dim')}{text}" + (" " + p("⚠ " + problem, "warn") if problem else "")
    if stripped.startswith("("):
        return p(line, "dim")
    return line
