"""
src/live_stage4_monitor.py
============================
Live Real-Time Stage 4 Feature Extraction Terminal Monitor

Monitors real filesystem operations on disk in real time. Whenever YOU create,
edit, copy, or save a file in the watched folder, this script processes it
through Stages 1-4 and prints the live extracted feature vector in your terminal!

Default Watched Path:
  - `monitored_test_dir/` (created automatically in project root for safe testing)
  - `~/Downloads` (optional)

Usage:
  python src/live_stage4_monitor.py
  python src/live_stage4_monitor.py --path C:/Users/YourName/Downloads
"""

import sys
import os
import time
import argparse
import queue
import logging
from pathlib import Path

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Windows UTF-8 stdout fix
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import threading
from collector.file_monitor import FileMonitor
from collector.process_monitor import ProcessMonitor
from collector.queue_joiner import QueueJoiner
from src.stage_3_dbrg.dbrg_manager import DBRGManager
from src.stage_4_features.feature_extractor import FeatureExtractor

# Suppress verbose debug logs in console
logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")


def print_banner(watch_paths):
    print("=" * 82)
    print(" 🛡️  REAL-TIME STAGE 4 MONITOR — WATCHING YOUR ACTUAL FILE OPERATIONS")
    print("=" * 82)
    print(" Active Watched Directories:")
    for p in watch_paths:
        print(f"   📂 {p}")
    print("\n INSTRUCTIONS:")
    print("   1. Keep this terminal window OPEN.")
    print("   2. Open your file explorer or editor (e.g. Notepad, VS Code).")
    print("   3. Create, edit, or copy any file inside the watched folder.")
    print("   4. Watch Stage 4 compute and print the live feature scores below!")
    print("=" * 82 + "\n")

    header = (
        f"{'#':<4} | {'Time':<8} | {'Operation':<11} | {'File Name':<20} | "
        f"{'S_ent':<6} | {'S_ETD':<6} | {'S_dist':<6} | {'S_dev':<6} | "
        f"{'λ_norm':<6} | {'State':<12}"
    )
    print("-" * len(header))
    print(header)
    print("-" * len(header))


def format_row(idx: int, fv: dict) -> str:
    file_basename = os.path.basename(fv.get("file_path", ""))
    if len(file_basename) > 20:
        file_basename = file_basename[:17] + "..."
        
    t_str = time.strftime("%H:%M:%S", time.localtime(fv["timestamp"] / 1000.0 if fv["timestamp"] > 1e12 else fv["timestamp"]))

    state_str = "🟢 NORMAL"
    if fv["is_superheated"]:
        state_str = "🔥 SUPERHEATED"
    elif fv["S_entropy"] > 0.85 and fv["S_dist"] > 0.2:
        state_str = "🚨 CRITICAL"
    elif fv["S_entropy"] > 0.85:
        state_str = "⚠️ HIGH ENTROPY"

    return (
        f"{idx:<4} | {t_str:<8} | {fv['operation']:<11} | {file_basename:<20} | "
        f"{fv['S_entropy']:<6.3f} | {fv['S_ETD']:<6.3f} | {fv['S_dist']:<6.3f} | "
        f"{fv['S_dev']:<6.3f} | {fv['lambda_norm']:<6.3f} | {state_str:<12}"
    )


def main():
    parser = argparse.ArgumentParser(description="Live Stage 4 Feature Extraction Monitor")
    parser.add_argument("--path", type=str, help="Custom folder path to watch in real-time")
    args = parser.parse_args()

    # Determine paths to watch (defaulting to user's Downloads directory)
    downloads_dir = Path.home() / "Downloads"
    if not downloads_dir.exists():
        downloads_dir.mkdir(parents=True, exist_ok=True)

    watch_paths = [str(downloads_dir.resolve())]
    if args.path:
        custom_p = Path(args.path).expanduser().resolve()
        if custom_p.exists() and str(custom_p) not in watch_paths:
            watch_paths.append(str(custom_p))
        else:
            print(f"⚠️ Custom path does not exist or already added: {custom_p}")

    print_banner(watch_paths)

    raw_queue = queue.Queue(maxsize=10000)
    file_monitor = FileMonitor(paths=watch_paths, recursive=True)

    # Setup ProcessMonitor for PID resolution
    process_monitor = ProcessMonitor(poll_interval_sec=1.0, hash_timeout_sec=2.0)
    process_monitor.start()

    dbrg = DBRGManager(decay_lambda=0.01)
    extractor = FeatureExtractor(dbrg_manager=dbrg)

    event_count = 0
    lock = threading.Lock()

    def process_ecar_event(ecar_event: dict):
        nonlocal event_count
        dbrg.process_event(ecar_event)
        fv = extractor.extract_features(ecar_event)
        if fv:
            with lock:
                event_count += 1
                print(format_row(event_count, fv), flush=True)

    queue_joiner = QueueJoiner(
        raw_queue=raw_queue,
        process_monitor=process_monitor,
        db_writer=None,  # Not saving to DB in live monitor
        on_event_callback=process_ecar_event,
    )

    # Start Stage 1 components
    file_monitor.start(raw_queue)
    queue_joiner.start()

    try:
        while True:
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n\n" + "=" * 82)
        print(" 🛑 Stopping Live Monitor... Cleaning up.")
        print("=" * 82)
        file_monitor.stop()
        queue_joiner.stop()
        process_monitor.stop()
        print(" Done. Bye!")


if __name__ == "__main__":
    main()
