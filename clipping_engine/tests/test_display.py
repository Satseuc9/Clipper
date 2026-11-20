"""Terminal display: color depth detection, plain-text fallback, styled engine section, worker count."""

import io
import re

import pytest

from clipping_engine import pipeline
from clipping_engine.monitor_bridge import engine_section, load_monitor, style_line

load_monitor()
from platforms.base import LIVE, OFFLINE, UNKNOWN, StreamResult  # noqa: E402
from utils import terminal  # noqa: E402

ANSI = re.compile(r"\x1b\[[0-9;]*m")


class Tty(io.StringIO):
    def isatty(self):
        return True


@pytest.mark.parametrize("env,tty,expected", [
    ({"TERM": "xterm-256color"}, True, 256),
    ({"TERM": "xterm-256color", "COLORTERM": "truecolor"}, True, 24),
    ({"TERM_PROGRAM": "Apple_Terminal", "TERM": "xterm"}, True, 256),
    ({"TERM": "xterm"}, True, 16),
    ({"TERM": "xterm-256color"}, False, 0),                       # redirected / piped
    ({"TERM": "xterm-256color", "NO_COLOR": "1"}, True, 0),
    ({"TERM": "dumb"}, True, 0),
    ({"TERM": "xterm-256color", "FORCE_COLOR": "1"}, False, 256),
    ({"TERM": "xterm-256color", "FORCE_COLOR": "1", "NO_COLOR": "1"}, False, 0),   # NO_COLOR wins
])
def test_color_depth(env, tty, expected):
    stream = Tty() if tty else io.StringIO()
    assert terminal.color_depth(stream, env) == expected


ENGINE_LINES = [
    "  ● twitch/ishowspeed: recording, session_011, 3 segment(s), 2 ad break(s) skipped",
    "  ● twitch/caseoh_: recording (ad break), session_011, 4 segment(s)",
    "  ● twitch/xqc: reconnecting (streamlink exited (code 1)), session_010, 6 segment(s)",
    "  ○ twitch/kaicenat: not recording - MAX_CONCURRENT_RECORDINGS (3) reached",
    "  💬 no chat for xqc: Twitch chat not authorized",
    "  ⚙ analysis (2 workers): 1 running, 2 waiting (0.9 GB), 3 clip(s) sent",
    "  ⚠ analysis is falling behind (2 worker(s))",
    "    [22:41:57] twitch/caseoh_/session_011/segment_0002: 6 moments, best 81, 1 clip(s) [audio+chat]",
    "    [22:45:05] twitch/x/session_1/segment_0004: 5 moments, best 58, 0 clip(s) [audio] ⚠ transcription unavailable",
    "    [22:46:20] twitch/x/session_1/segment_0006: FAILED after 3 attempts (boom)",
    "  (no live streamers being recorded)",
]


def rows():
    return [StreamResult("twitch", "IShowSpeed", LIVE, viewers=17515, title="GTA 6"),
            StreamResult("twitch", "N3on", OFFLINE),
            StreamResult("twitch", "ghost", UNKNOWN, error="user not found")]


def test_plain_mode_has_no_escape_codes(monkeypatch, capsys):
    monkeypatch.setattr(terminal, "DEPTH", 0)
    totals = terminal.print_report([("TWITCH", "", rows())], [], 0)
    terminal.print_cycle_footer(totals, 60)
    for line in engine_section(list(ENGINE_LINES)):
        print(line)
    out = capsys.readouterr().out
    assert "\x1b" not in out
    assert "[ LIVE ]    IShowSpeed | 17,515 viewers | GTA 6" in out
    assert "TOTAL: 3 | LIVE: 1 | OFFLINE: 1 | UNKNOWN: 1" in out
    assert "3 clip(s) sent" in out and "CLIPPING ENGINE" in out
    assert "━" not in out, "plain mode uses ASCII rules"


@pytest.mark.parametrize("depth", [16, 256, 24])
def test_colors_never_change_the_text(monkeypatch, depth):
    monkeypatch.setattr(terminal, "DEPTH", depth)
    for line in ENGINE_LINES:
        styled = style_line(line, terminal)
        assert ANSI.sub("", styled) == line
    assert "\x1b" in style_line(ENGINE_LINES[0], terminal)
    for row in rows():
        assert ANSI.search(terminal.format_row(row))
    if depth == 24:
        assert "38;2;" in terminal.paint("x", "heading")
    elif depth == 256:
        assert "38;5;" in terminal.paint("x", "heading")


def test_states_use_distinct_roles(monkeypatch):
    monkeypatch.setattr(terminal, "DEPTH", 256)
    live, offline, unknown = (terminal.format_row(r) for r in rows())
    warn = style_line(ENGINE_LINES[2], terminal)
    codes = [ANSI.findall(x)[0] for x in (live, offline, unknown)]
    assert len(set(codes)) == 3
    assert terminal._code("warn") in warn, "reconnecting is shown as a warning"


def test_auto_workers(monkeypatch):
    assert pipeline.resolve_workers(4) == 4
    monkeypatch.setattr(pipeline, "_performance_cores", lambda: 4)       # M1/M2/M3/M4
    assert pipeline.resolve_workers("auto") == 2
    monkeypatch.setattr(pipeline, "_performance_cores", lambda: 12)      # Max chips
    assert pipeline.resolve_workers("auto") == 3
    monkeypatch.setattr(pipeline, "_performance_cores", lambda: None)
    monkeypatch.setattr(pipeline.os, "cpu_count", lambda: 4)
    assert pipeline.resolve_workers("auto") == 1

