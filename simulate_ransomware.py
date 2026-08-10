"""
simulate_ransomware.py
======================
Safely simulates ransomware behavior inside `monitored_test_dir/`
to trigger Stage 4 real-time detection escalations in the live monitor.

Actions performed:
  1. Creates 20 dummy files inside monitored_test_dir/
  2. Overwrites each file rapidly with high-entropy random bytes (AES simulation)
  3. Pauses briefly so you can watch the live terminal state escalate:
     🟢 NORMAL  →  ⚠️ HIGH ENTROPY  →  🚨 CRITICAL
"""

import sys
import os
import time
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
TEST_DIR = PROJECT_ROOT / "monitored_test_dir"
TEST_DIR.mkdir(exist_ok=True)

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

print("=" * 65)
print(" RANSOMWARE ATTACK SIMULATOR (SAFE & NON-DESTRUCTIVE)")
print("=" * 65)
print(f" Target Directory: {TEST_DIR}")
print(" Simulating rapid encrypted file writes (20 files)...")
print("=" * 65 + "\n")

random.seed(42)

# Step 1: Create 20 files rapidly with encrypted high-entropy bytes
for i in range(1, 21):
    file_path = TEST_DIR / f"user_document_{i}.docx.locked"
    
    # Generate 4KB of high-entropy random ciphertext bytes
    ciphertext = bytes(random.getrandbits(8) for _ in range(4096))
    
    with open(file_path, "wb") as f:
        f.write(ciphertext)

    print(f"  [+] Encrypted target file {i}/20: {file_path.name}")
    time.sleep(0.08)  # Rapid 80ms write speed to trigger Hawkes burst!

print("\n" + "=" * 65)
print(" 🛑 Simulation Complete! Look at your Monitor Terminal to see")
print("    how Stage 4 detected the attack and escalated the state.")
print("=" * 65 + "\n")
