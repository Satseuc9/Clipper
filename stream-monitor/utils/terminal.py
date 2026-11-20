"""
Terminal output: colors, the status report and change notifications.

"Neon night" palette: cyan section headings, magenta titles, violet rules and
restrained sunset-orange accents. State colors keep their meaning:
LIVE = neon green, OFFLINE = muted slate, UNKNOWN = gold, warnings =
sunset orange, errors = hot pink-red.

Color depth follows the terminal: 24-bit when COLORTERM says truecolor
(iTerm2, VS Code, Ghostty...), 256 colors for TERM=*-256color (macOS
Terminal), else the basic 16. Plain text, with no escape codes at all, when
output is not a terminal (redirected/piped), NO_COLOR is set or TERM=dumb.
FORCE_COLOR=1 forces color (e.g. for `| less -R`); NO_COLOR always wins.
"""

import os
import sys
import time
from datetime import datetime

from platforms.base import LIVE, OFFLINE, UNKNOWN, format_duration
from utils.tracker import is_stale

try:
    import colorama

    colorama.just_fix_windows_console()
except (ImportError, AttributeError):
    pass

# role: (24-bit RGB, 256-color index, basic 16-color SGR code)
PALETTE = {
    "title":   ((255, 70, 200), 205, "95"),    # neon magenta
    "heading": ((0, 220, 255), 45, "96"),      # neon cyan
    "rule":    ((130, 80, 200), 97, "35"),     # deep violet
    "accent":  ((255, 140, 70), 209, "33"),    # sunset orange (used sparingly)
    "live":    ((70, 255, 170), 85, "92"),     # neon green
    "offline": ((125, 125, 155), 103, "90"),   # muted slate
    "unknown": ((255, 210, 90), 221, "93"),    # gold (distinct from the orange warnings)
    "warn":    ((255, 140, 70), 209, "33"),    # sunset orange
    "error":   ((255, 70, 110), 204, "91"),    # hot pink-red
    "info":    ((0, 220, 255), 45, "96"),      # cyan values (viewers, counts)
    "rec":     ((255, 70, 200), 205, "95"),    # magenta recording dot
    "dim":     ((150, 150, 175), 246, "90"),
}
STATE_ROLE = {LIVE: "live", OFFLINE: "offline", UNKNOWN: "unknown"}
TAG_WIDTH = len("[ UNKNOWN ]")
WIDTH = 64
RESET = "\033[0m"


def color_depth(stream=None, env=None):
    """0 (plain), 16, 256 or 24 (truecolor) for this output stream."""
    env = os.environ if env is None else env
    stream = sys.stdout if stream is None else stream
    if env.get("NO_COLOR"):
        return 0
    forced = env.get("FORCE_COLOR", "") not in ("", "0")
    try:
        tty = stream.isatty()
    except (AttributeError, ValueError):
        tty = False
    if not (tty or forced) or env.get("TERM") == "dumb":
        return 0
    if env.get("COLORTERM", "").lower() in ("truecolor", "24bit"):
        return 24
    if "256" in env.get("TERM", "") or env.get("TERM_PROGRAM") == "Apple_Terminal":
        return 256
    return 16


DEPTH = color_depth()


def set_depth(depth):
    """Override the detected color depth (0 = plain text)."""
    global DEPTH
    DEPTH = depth


def _code(role, bold=False):
    rgb, idx, basic = PALETTE[role]
    if DEPTH == 24:
        code = f"38;2;{rgb[0]};{rgb[1]};{rgb[2]}"
    elif DEPTH == 256:
        code = f"38;5;{idx}"
    else:
        code = basic
    return f"\033[{'1;' if bold else ''}{code}m"


def paint(text, role=None, bold=False):
    """Color `text` with a palette role; returns it unchanged in plain mode."""
    if not DEPTH or (role is None and not bold):
        return text
    if role is None:
        return f"\033[1m{text}{RESET}"
    return f"{_code(role, bold)}{text}{RESET}"


def rule(heavy=False, label=None):
    """A horizontal rule, optionally with a heading label (ASCII in plain mode)."""
    char = ("━" if heavy else "─") if DEPTH else ("=" if heavy else "-")
    if label is None:
        return paint(char * WIDTH, "rule")
    lead = char * 2
    tail = char * max(3, WIDTH - len(label) - 4)
    return f"{paint(lead, 'rule')} {paint(label, 'heading', bold=True)} {paint(tail, 'rule')}"


# Never crash on emoji or non-Latin stream titles in limited consoles.
try:
    sys.stdout.reconfigure(errors="replace")
except (AttributeError, ValueError):
    pass


def clean(text, limit=80):
    """Make API-provided text safe to print: no control characters, bounded length."""
    text = " ".join("".join(ch if ch.isprintable() else " " for ch in str(text)).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def clear_screen():
    if sys.stdout.isatty():
        print("\033[2J\033[H", end="", flush=True)


def timestamp(epoch=None, fmt="%H:%M:%S"):
    return datetime.fromtimestamp(epoch or time.time()).strftime(fmt)


def platform_name(key):
    return {"twitch": "Twitch", "kick": "Kick", "youtube": "YouTube", "tiktok": "TikTok"}.get(key, key)


def format_row(row):
    role = STATE_ROLE.get(row.status, "unknown")
    tag = paint(f"[ {row.status} ]".ljust(TAG_WIDTH), role, bold=row.status == LIVE)
    sep = paint(" | ", "dim")
    name = clean(row.label, 40)
    if row.status == LIVE:
        parts = [paint(name, bold=True)]
        if row.viewers is not None:
            parts.append(paint(f"{row.viewers:,} viewers", "info"))
        if row.title:
            parts.append(clean(row.title, 60))
    elif row.status == OFFLINE:
        parts = [paint(name, "offline")]
    else:
        parts = [name]
        if row.error:
            parts.append(paint(clean(row.error, 100), "dim"))
    line = f"{tag} {sep.join(parts)}"
    if is_stale(row):
        age = format_duration(time.time() - row.checked_at)
        line += " " + paint(f"(last confirmed {age} ago; check failed: {clean(row.error, 70)})", "warn")
    return line


def count(rows_by_platform):
    totals = {LIVE: 0, OFFLINE: 0, UNKNOWN: 0}
    for rows in rows_by_platform.values():
        for row in rows:
            totals[row.status] = totals.get(row.status, 0) + 1
    return totals


def totals_line(totals):
    sep = paint(" | ", "dim")
    return sep.join([
        f"TOTAL: {sum(totals.values())}",
        paint(f"LIVE: {totals[LIVE]}", "live", bold=True),
        paint(f"OFFLINE: {totals[OFFLINE]}", "offline"),
        paint(f"UNKNOWN: {totals[UNKNOWN]}", "unknown"),
    ])


def print_report(sections, history, recent_changes_to_show, warnings=(), show_totals=True):
    """
    sections: list of (title, note, rows), one per platform.
    history:  list of StatusChange, newest last.
    show_totals: False in continuous mode, where the cycle footer shows them.
    """
    print(rule(heavy=True))
    stamp = timestamp(fmt="%H:%M:%S")
    title = "STREAMER LIVE MONITOR"
    print(f"{paint('◆', 'accent') if DEPTH else '*'} {paint(title, 'title', bold=True)}"
          f"{' ' * max(1, WIDTH - len(title) - len(stamp) - 2)}{paint(stamp, 'dim')}")
    print(rule(heavy=True))
    for warning in warnings:
        print(paint(f"WARNING: {warning}", "warn"))

    for title, note, rows in sections:
        print()
        print(paint(title, "heading", bold=True) + (" " + paint(f"({note})", "dim") if note else ""))
        if not rows:
            print(paint("  (no streamers configured)", "dim"))
        for row in rows:
            print(format_row(row))

    totals = count({title: rows for title, _, rows in sections})
    if show_totals:
        print()
        print(rule())
        print(totals_line(totals))
        print(paint(f"Last checked: {timestamp(fmt='%Y-%m-%d %H:%M:%S')}", "dim"))

    if recent_changes_to_show and history:
        print()
        print(paint("RECENT CHANGES", "heading", bold=True))
        for change in history[-recent_changes_to_show:]:
            row = change.new
            print(f"  {paint('[' + timestamp(change.time) + ']', 'dim')} {platform_name(row.platform)} "
                  f"{clean(row.label, 40)}: {change.old} -> "
                  f"{paint(row.status, STATE_ROLE[row.status], bold=row.status == LIVE)}")
    return totals


def print_change(change):
    row = change.new
    print()
    if row.status == LIVE:
        print(paint(f"[{timestamp(change.time)}] 🔴 STREAMER IS NOW LIVE", "live", bold=True))
    else:
        print(paint(f"[{timestamp(change.time)}] ⚫ STREAMER WENT OFFLINE", "offline", bold=True))
    label = lambda text: paint(text, "dim")      # noqa: E731
    print(f"{label('Platform:')} {platform_name(row.platform)}")
    print(f"{label('Username:')} {clean(row.label, 60)}")
    if row.status == LIVE:
        if row.viewers is not None:
            print(f"{label('Viewers:')} {paint(f'{row.viewers:,}', 'info')}")
        if row.title:
            print(f"{label('Title:')} {clean(row.title, 100)}")
        if row.url:
            print(f"{label('URL:')} {paint(row.url, 'heading')}")


def print_cycle_footer(totals, interval):
    print()
    print(rule())
    print(totals_line(totals))
    print(paint(f"Last checked: {timestamp(fmt='%Y-%m-%d %H:%M:%S')}", "dim") + "  "
          + paint(f"Next check in {interval} seconds...", "accent") + paint(" (CTRL+C to stop)", "dim"),
          flush=True)
