#!/usr/bin/env python3
"""
Multi-platform streamer live status checker.

    python main.py                 continuous monitoring (default, CTRL+C to stop)
    python main.py --once          check once and exit
    python main.py --interval 30   continuous monitoring every 30 seconds

Streamers are listed in config.py; credentials are read from .env.
"""

import argparse
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))  # so the script also works when run from another folder

from dotenv import load_dotenv  # noqa: E402

load_dotenv(BASE_DIR / ".env")

import config  # noqa: E402
from platforms.base import dedupe, format_duration  # noqa: E402
from platforms.kick import KickMonitor  # noqa: E402
from platforms.tiktok import TikTokMonitor  # noqa: E402
from platforms.twitch import TwitchMonitor  # noqa: E402
from platforms.youtube import YouTubeMonitor  # noqa: E402
from utils import terminal  # noqa: E402
from utils.tracker import StatusTracker  # noqa: E402

MIN_INTERVAL = 10


def parse_args():
    parser = argparse.ArgumentParser(description="Check whether streamers are live on Twitch, Kick, YouTube and TikTok.")
    parser.add_argument("--once", action="store_true", help="run a single check and exit")
    parser.add_argument(
        "--interval", type=int, metavar="SECONDS",
        help=f"seconds between checks in continuous mode (default {config.CHECK_INTERVAL_SECONDS}, "
             f"minimum {MIN_INTERVAL}); platforms with stricter API limits refresh less often",
    )
    args = parser.parse_args()
    if args.interval is not None and args.interval < MIN_INTERVAL:
        parser.error(f"--interval must be at least {MIN_INTERVAL} seconds")
    return args


def build_monitors():
    timeout = config.REQUEST_TIMEOUT_SECONDS
    minimum = config.PLATFORM_MIN_INTERVAL_SECONDS
    return [
        TwitchMonitor(timeout, minimum.get("twitch", 10)),
        KickMonitor(timeout, minimum.get("kick", 30)),
        YouTubeMonitor(
            timeout, minimum.get("youtube", 60),
            daily_quota=config.YOUTUBE_DAILY_QUOTA,
            quota_share=config.YOUTUBE_QUOTA_SHARE,
            videos_to_scan=config.YOUTUBE_RECENT_VIDEOS_TO_SCAN,
        ),
        TikTokMonitor(timeout),
    ]


def streamers_for(key):
    names = config.STREAMERS.get(key) or []
    return [names] if isinstance(names, str) else list(names)


def run_cycle(monitors, tracker, interval):
    """Check every platform once. One platform failing never stops the others."""
    sections, changes = [], []
    for monitor in monitors:
        names = streamers_for(monitor.key)
        poll_interval = monitor.poll_interval(names, interval)
        try:
            results = monitor.check(names, interval)
        except Exception as exc:  # check() handles its own errors; this is a last safety net
            results = [monitor.unknown(n, f"unexpected error ({type(exc).__name__})") for n in dedupe(names)]

        rows, new_changes = tracker.update(monitor.key, results, poll_interval)
        note = ""
        if monitor.available and names and poll_interval > interval:
            note = f"refreshes every {format_duration(poll_interval)} to stay within API limits"
        sections.append((monitor.title, note, rows))
        changes.extend(new_changes)
    return sections, changes


def config_warnings(monitors):
    known = {m.key for m in monitors}
    return [f"config.py: unknown platform '{key}' in STREAMERS (ignored)"
            for key in config.STREAMERS if key not in known]


def _call_hook(on_cycle, sections, changes):
    """Run the optional per-cycle hook; a failing hook never stops monitoring."""
    if on_cycle is None:
        return []
    try:
        return list(on_cycle(sections, changes) or [])
    except Exception as exc:
        return [f"WARNING: cycle hook failed ({type(exc).__name__}: {exc})"]


def run_monitor(interval, once=False, on_cycle=None):
    """
    Run the monitoring loop (or a single check with once=True).

    on_cycle(sections, changes) is called after every cycle with the rows
    that were just displayed (sections = [(title, note, rows)]) and the
    status changes of that cycle. It may return extra lines to print.
    This is how other tools (e.g. the clipping engine) reuse the monitor.
    """
    interval = max(MIN_INTERVAL, int(interval))
    monitors = build_monitors()
    tracker = StatusTracker(config.KEEP_LAST_STATUS_SECONDS)
    warnings = config_warnings(monitors)

    try:
        if once:
            sections, changes = run_cycle(monitors, tracker, interval)
            terminal.print_report(sections, [], 0, warnings)
            for line in _call_hook(on_cycle, sections, changes):
                print(line)
            return 0

        print("Checking streamers...", flush=True)
        while True:
            started = time.monotonic()
            sections, changes = run_cycle(monitors, tracker, interval)

            terminal.clear_screen()
            earlier = tracker.history[: len(tracker.history) - len(changes)]
            totals = terminal.print_report(sections, earlier, config.RECENT_CHANGES_TO_SHOW, warnings)
            for change in changes:
                terminal.print_change(change)
            for line in _call_hook(on_cycle, sections, changes):
                print(line)

            wait = max(1, round(interval - (time.monotonic() - started)))
            terminal.print_cycle_footer(totals, wait)
            time.sleep(wait)
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")
        return 0


def main():
    args = parse_args()
    return run_monitor(args.interval or config.CHECK_INTERVAL_SECONDS, once=args.once)


if __name__ == "__main__":
    sys.exit(main())
