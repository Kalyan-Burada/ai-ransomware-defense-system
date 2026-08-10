"""
src/stage_4_features/demo_stage4_features.py
================================================
Stage 4 — Live Feature Extraction Terminal Monitor

Runs an interactive live demonstration comparing:
  [1] Benign Document Editing Scenario (Notepad/Text editor)
  [2] High-Entropy Compressed Archiving Scenario (7-Zip / ZIP)
  [3] Rapid Encrypted Ransomware Burst Attack Scenario

Outputs the full per-event Feature Vector in real time:
  - Shannon Content Entropy (S_entropy & raw H)
  - Entropy-Topology Divergence (S_ETD)
  - Graph Topology & Velocity (S_dist, S_dev, S_stab)
  - Hawkes Temporal Burst Intensity (lambda_norm, branching n)
  - Superheated Attack State Flag
"""

import sys
import os
import time
import random
import tempfile
import zlib

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Force stdout utf-8 encoding on Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from src.stage_3_dbrg.dbrg_manager import DBRGManager
from src.stage_4_features.feature_extractor import FeatureExtractor


def print_banner():
    print("=" * 76)
    print(" 🛡️  AI Ransomware Defense System — Stage 4 Feature Extraction Monitor")
    print("=" * 76)
    print(" Stage 4 extracts live content-level & temporal threat signals per write:")
    print("   • S_entropy   : Normalized 4KB Shannon Content Entropy [0.0 - 1.0]")
    print("   • S_ETD       : Entropy-Topology 2D KL-Divergence score [0.0 - 1.0]")
    print("   • S_dist/dev  : Graph Fan-out Distance & Target Access Velocity")
    print("   • S_stab      : Operational Stability (Repeated vs New targets)")
    print("   • lambda_norm : Hawkes Point Process Burst Intensity [0.0 - 1.0]")
    print("   • Branching n : Hawkes Self-Excitation Ratio (n >= 1.0 = SUPERHEATED!)")
    print("=" * 76 + "\n")


def print_feature_vector_header():
    header = (
        f"{'Event':<6} | {'PID':<5} | {'Operation':<11} | {'S_ent':<6} | "
        f"{'S_ETD':<6} | {'S_dist':<6} | {'S_dev':<6} | {'S_stab':<6} | "
        f"{'λ_norm':<6} | {'Hawkes n':<8} | {'State':<12}"
    )
    print("-" * len(header))
    print(header)
    print("-" * len(header))


def print_feature_row(idx: int, fv: dict):
    if not fv:
        return
    
    state_str = "🔥 SUPERHEATED" if fv["is_superheated"] else "🟢 NORMAL"
    if fv["S_entropy"] > 0.85 and fv["fan_out_degree"] > 5:
        state_str = "⚠️ SUSPICIOUS"
    if fv["is_superheated"] or (fv["S_entropy"] > 0.9 and fv["S_dist"] > 0.2):
        state_str = "🚨 CRITICAL"

    row = (
        f"{idx:<6} | {fv['pid']:<5} | {fv['operation']:<11} | "
        f"{fv['S_entropy']:<6.3f} | {fv['S_ETD']:<6.3f} | "
        f"{fv['S_dist']:<6.3f} | {fv['S_dev']:<6.3f} | "
        f"{fv['S_stab']:<6.3f} | {fv['lambda_norm']:<6.3f} | "
        f"{fv['branching_n']:<8.3f} | {state_str:<12}"
    )
    print(row)


def run_benign_editing_demo(extractor: FeatureExtractor, dbrg: DBRGManager):
    print("\n" + "─" * 76)
    print(" 📖 SCENARIO 1: Benign Document Editing (Notepad saving text files)")
    print("─" * 76)
    print_feature_vector_header()

    pid = 1042
    temp_dir = tempfile.mkdtemp(prefix="stage4_demo_benign_")
    
    try:
        # Simulate text editing on 2 files with repeated saves
        sample_texts = [
            b"Project Report v1.0\nThis is a benign text document edited by user.\n" * 50,
            b"Meeting Notes 2026-08-10\n- Discussed Stage 4 pipeline.\n- Verified ETD engine.\n" * 50,
        ]

        event_idx = 1
        for cycle in range(3):
            for file_i, content in enumerate(sample_texts):
                file_path = os.path.join(temp_dir, f"doc_{file_i+1}.txt")
                with open(file_path, "wb") as f:
                    f.write(content)

                ecar = {
                    "actorID": f"pid:{pid}",
                    "objectID": f"file:{file_path}",
                    "pid": pid,
                    "operation": "FILE_MODIFY",
                    "timestamp": int(time.time() * 1000),
                    "context": {"exe_path": "notepad.exe"},
                }

                dbrg.process_event(ecar)
                fv = extractor.extract_features(ecar)
                print_feature_row(event_idx, fv)
                event_idx += 1
                time.sleep(0.3)  # Gentle human editing speed

    finally:
        # Cleanup
        for root, dirs, files in os.walk(temp_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            os.rmdir(root)


def run_compressed_archive_demo(extractor: FeatureExtractor, dbrg: DBRGManager):
    print("\n" + "─" * 76)
    print(" 📦 SCENARIO 2: Benign Compressed Archiving (7-Zip creating single archive)")
    print("─" * 76)
    print_feature_vector_header()

    pid = 2088
    temp_dir = tempfile.mkdtemp(prefix="stage4_demo_archive_")

    try:
        # High entropy compressed data, but single output target (low d_out)
        random.seed(101)
        raw_source = bytes(random.getrandbits(8) for _ in range(32768))
        compressed_data = zlib.compress(raw_source, level=9)[:4096]

        archive_path = os.path.join(temp_dir, "backup_dataset.zip")
        
        event_idx = 1
        for chunk_i in range(4):
            with open(archive_path, "ab") as f:
                f.write(compressed_data)

            ecar = {
                "actorID": f"pid:{pid}",
                "objectID": f"file:{archive_path}",
                "pid": pid,
                "operation": "FILE_MODIFY",
                "timestamp": int(time.time() * 1000),
                "context": {"exe_path": "7z.exe"},
            }

            dbrg.process_event(ecar)
            fv = extractor.extract_features(ecar)
            print_feature_row(event_idx, fv)
            event_idx += 1
            time.sleep(0.2)

    finally:
        for root, dirs, files in os.walk(temp_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            os.rmdir(root)


def run_ransomware_burst_demo(extractor: FeatureExtractor, dbrg: DBRGManager):
    print("\n" + "─" * 76)
    print(" 🚨 SCENARIO 3: Ransomware Rapid Encryption Burst (High entropy + Rapid fan-out)")
    print("─" * 76)
    print_feature_vector_header()

    pid = 6660
    temp_dir = tempfile.mkdtemp(prefix="stage4_demo_ransomware_")

    try:
        random.seed(999)
        event_idx = 1
        
        # Rapidly target 15 unique files with high-entropy ciphertext
        for file_i in range(15):
            ciphertext = bytes(random.getrandbits(8) for _ in range(4096))
            file_path = os.path.join(temp_dir, f"important_doc_{file_i+1}.docx.locked")
            
            with open(file_path, "wb") as f:
                f.write(ciphertext)

            ecar = {
                "actorID": f"pid:{pid}",
                "objectID": f"file:{file_path}",
                "pid": pid,
                "operation": "FILE_MODIFY",
                "timestamp": int(time.time() * 1000),
                "context": {"exe_path": "encryptor.exe"},
            }

            dbrg.process_event(ecar)
            fv = extractor.extract_features(ecar)
            print_feature_row(event_idx, fv)
            event_idx += 1
            time.sleep(0.03)  # Rapid 30ms burst per file!

    finally:
        for root, dirs, files in os.walk(temp_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            os.rmdir(root)


def main():
    print_banner()

    dbrg = DBRGManager(decay_lambda=0.01)
    extractor = FeatureExtractor(
        dbrg_manager=dbrg,
        hawkes_mu=0.1,
        hawkes_alpha=0.5,
        hawkes_beta=0.8,
    )

    run_benign_editing_demo(extractor, dbrg)
    run_compressed_archive_demo(extractor, dbrg)
    run_ransomware_burst_demo(extractor, dbrg)

    print("\n" + "=" * 76)
    print(" 💡 STAGE 4 OBSERVATIONS:")
    print("   1. Benign Editing  : Low S_entropy (~0.4-0.5), Low Hawkes intensity, Low S_ETD.")
    print("   2. 7-Zip Archiving : High S_entropy (~0.99), BUT Low d_out (1 archive) -> S_ETD stays low!")
    print("   3. Ransomware Burst: High S_entropy + High Target Velocity + Rapid Hawkes Accumulation -> 🚨 CRITICAL!")
    print("=" * 76 + "\n")


if __name__ == "__main__":
    main()
