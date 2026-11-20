"""
Terminal output: colors, the status report and change notifications.

GREEN = LIVE, GRAY = OFFLINE, YELLOW = UNKNOWN / API error.
Colors are turned off when output is not a terminal or NO_COLOR is set.
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

USE_COLOR = sys.stdout.isatty() and not os.getenv("NO_COLOR")
GREEN = "\033[92m" if USE_COLOR else ""
GRAY = "\033[90m" if USE_COLOR else ""
YELLOW = "\033[93m" if USE_COLOR else ""
BOLD = "\033[1m" if USE_COLOR else ""
RESET = "\033[0m" if USE_COLOR else ""

STATUS_COLOR = {LIVE: GREEN, OFFLINE: GRAY, UNKNOWN: YELLOW}
TAG_WIDTH = len("[ UNKNOWN ]")
LINE = "=" * 60
THIN = "-" * 60

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
    color = STATUS_COLOR.get(row.status, YELLOW)
    tag = f"[ {row.status} ]".ljust(TAG_WIDTH)
    parts = [clean(row.label, 40)]
    if row.status == LIVE:
        if row.viewers is not None:
            parts.append(f"{row.viewers:,} viewers")
        if row.title:
            parts.append(clean(row.title, 60))
    elif row.status == UNKNOWN and row.error:
        parts.append(clean(row.error, 100))

    line = f"{color}{tag}{RESET} {' | '.join(parts)}"
    if is_stale(row):
        age = format_duration(time.time() - row.checked_at)
        line += f" {YELLOW}(last confirmed {age} ago; check failed: {clean(row.error, 70)}){RESET}"
    return line


def count(rows_by_platform):
    totals = {LIVE: 0, OFFLINE: 0, UNKNOWN: 0}
    for rows in rows_by_platform.values():
        for row in rows:
            totals[row.status] = totals.get(row.status, 0) + 1
    return totals


def print_report(sections, history, recent_changes_to_show, warnings=()):
    """
    sections: list of (title, note, rows), one per platform.
    history:  list of StatusChange, newest last.
    """
    print(LINE)
    print(f"{BOLD}STREAMER LIVE MONITOR{RESET}")
    print(LINE)
    for warning in warnings:
        print(f"{YELLOW}WARNING: {warning}{RESET}")

    for title, note, rows in sections:
        print()
        print(f"{BOLD}{title}{RESET}" + (f" {GRAY}({note}){RESET}" if note else ""))
        if not rows:
            print(f"{GRAY}  (no streamers configured){RESET}")
        for row in rows:
            print(format_row(row))

    totals = count({title: rows for title, _, rows in sections})
    print()
    print(THIN)
    print(f"TOTAL: {sum(totals.values())}")
    print(f"{GREEN}LIVE: {totals[LIVE]}{RESET}")
    print(f"{GRAY}OFFLINE: {totals[OFFLINE]}{RESET}")
    print(f"{YELLOW}UNKNOWN: {totals[UNKNOWN]}{RESET}")
    print()
    print(f"Last checked: {timestamp(fmt='%Y-%m-%d %H:%M:%S')}")

    if recent_changes_to_show and history:
        print()
        print(f"{BOLD}RECENT CHANGES{RESET}")
        for change in history[-recent_changes_to_show:]:
            row = change.new
            color = STATUS_COLOR[row.status]
            print(f"  [{timestamp(change.time)}] {platform_name(row.platform)} "
                  f"{clean(row.label, 40)}: {change.old} -> {color}{row.status}{RESET}")
    return totals


def print_change(change):
    row = change.new
    print()
    if row.status == LIVE:
        print(f"{GREEN}{BOLD}[{timestamp(change.time)}] 🔴 STREAMER IS NOW LIVE{RESET}")
    else:
        print(f"{GRAY}{BOLD}[{timestamp(change.time)}] ⚫ STREAMER WENT OFFLINE{RESET}")
    print(f"Platform: {platform_name(row.platform)}")
    print(f"Username: {clean(row.label, 60)}")
    if row.status == LIVE:
        if row.viewers is not None:
            print(f"Viewers: {row.viewers:,}")
        if row.title:
            print(f"Title: {clean(row.title, 100)}")
        if row.url:
            print(f"URL: {row.url}")


def print_cycle_footer(totals, interval):
    print()
    print(THIN)
    print(f"{GREEN}LIVE: {totals[LIVE]}{RESET} | {GRAY}OFFLINE: {totals[OFFLINE]}{RESET} | "
          f"{YELLOW}UNKNOWN: {totals[UNKNOWN]}{RESET}")
    print()
    print(f"Next check in {interval} seconds... (CTRL+C to stop)", flush=True)
