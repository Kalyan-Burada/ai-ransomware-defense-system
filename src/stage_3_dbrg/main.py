import time
import os
import sys
from pathlib import Path

# Ensure project root is in sys.path for direct script execution
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from src.stage_3_dbrg import DBRGManager, DBRGGarbageCollector

# Initialize Stage 3 DBRG Manager & Garbage Collector
mgr = DBRGManager(decay_lambda=0.05)
gc = DBRGGarbageCollector(mgr, prune_threshold=0.01, prune_interval=5.0)
gc.start()

class LiveFileHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if not event.is_directory:
            self._process(event.src_path, 'FILE_MODIFY')
    def on_created(self, event):
        if not event.is_directory:
            self._process(event.src_path, 'FILE_CREATE')
            
    def _process(self, path, op):
        # Ingest real OS file event into Stage 3 DBRG Graph
        mgr.process_event({
            'actorID': f'pid:{os.getpid()}',
            'objectID': f'file:{path}',
            'pid': os.getpid(),
            'operation': op,
            'context': {'exe_path': 'local_bash_or_editor', 'ppid': 1, 'parent_exe': 'bash'}
        })
        print(f'\n[LIVE EVENT DETECTED] {op} -> {os.path.basename(path)}')
        print(f' -> Total Nodes in Graph: {mgr.get_node_count()} | Total Edges: {mgr.get_edge_count()}')
        edge = mgr.get_edge_data(f'pid:{os.getpid()}', f'file:{path}')
        if edge:
            print(f' -> Edge Weight (TDEW): {edge["weight"]:.4f} | Event Count: {edge["event_count"]}')


watch_dir = os.path.expanduser('~/Downloads')
observer = Observer()
observer.schedule(LiveFileHandler(), path=watch_dir, recursive=True)
observer.start()

print('=' * 65)
print(f'  STAGE 3 LIVE MONITOR RUNNING ON: {watch_dir}')
print('  Open another terminal or text editor and modify files in ~/Downloads!')
print('  Press Ctrl+C to stop.')
print('=' * 65)

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    observer.stop()
    gc.stop()