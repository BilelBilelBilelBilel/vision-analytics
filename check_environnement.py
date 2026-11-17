
import sys
import os
import importlib
import subprocess
import traceback

print("=" * 65)
print("VISION ANALYTICS - ENVIRONMENT DIAGNOSTICS")
print("=" * 65)

print(f"\nPython version : {sys.version}")
print(f"Python path    : {sys.executable}")
print(f"Virtual env    : {sys.prefix != sys.base_prefix}")

# Packages required by the project
packages = [
    "numpy",
    "cv2",
    "torch",
    "ultralytics",
    "supervision",
    "PIL",
    "scipy",
    "fastapi",
    "uvicorn",
    "requests",
    "onnx",
    "onnxruntime",
    "matplotlib",
]

print("\n[1] PACKAGE IMPORT TESTS")
print("-" * 65)

failed = []

for package in packages:
    try:
        module = importlib.import_module(package)
        version = getattr(module, "__version__", "version unavailable")
        print(f"[OK]   {package:<18} {version}")
    except Exception as e:
        print(f"[FAIL] {package:<18} {type(e).__name__}: {e}")
        failed.append(package)

print("\n[2] PIP DEPENDENCY CONSISTENCY")
print("-" * 65)

result = subprocess.run(
    [sys.executable, "-m", "pip", "check"],
    capture_output=True,
    text=True
)

print(result.stdout or "No dependency conflicts.")
if result.stderr:
    print(result.stderr)

print("\n[3] PROJECT FILES")
print("-" * 65)

files = [
    "requirements.txt",
    "step1_tracking.py",
    "step2_telemetry.py",
    "step3_benchmark.py",
    "step4_alerts.py",
    "data",
    "output",
]

for file in files:
    print(f"[{'OK' if os.path.exists(file) else 'MISSING'}] {file}")

print("\n[4] VIDEO INPUT")
print("-" * 65)

video = "data/youtube_test.mp4"

if os.path.isfile(video):
    print(f"[OK] Video found: {video}")
    print(f"Size: {os.path.getsize(video) / (1024*1024):.2f} MB")
else:
    print(f"[MISSING] {video}")

print("\n" + "=" * 65)
print("DIAGNOSTIC SUMMARY")
print("=" * 65)

if failed or result.returncode != 0:
    print(f"ISSUES FOUND: {len(failed)} package import failure(s).")
    print("Review the errors above.")
else:
    print("ALL PACKAGE IMPORTS PASSED.")
    print("No pip dependency conflicts detected.")

print("=" * 65)
