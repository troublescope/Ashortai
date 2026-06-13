"""
colab_utils — Google Colab Optimizations & Helpers

Provides utilities for:
  • Mounting Google Drive for persistent model caching
  • VRAM/RAM monitoring and aggressive cache clearing
  • Safe disk cleanup between pipeline stages
  • Zipping outputs for easy download
  • Auto-configuring optimal Colab T4 defaults

Usage (in notebook):
    from clipping.colab_utils import setup_colab, free_gpu, zip_outputs
    setup_colab(mount_drive=True)
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

# =============================================================================
# GPU / RAM UTILITIES
# =============================================================================

def _in_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def get_gpu_info():
    """Print GPU name and memory status."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=False,
        )
        if result.returncode == 0:
            name, used, total = result.stdout.strip().split(", ")
            print(f"🎮 GPU: {name} | VRAM {used:.0f}/{total:.0f} MB")
        else:
            print("⚠️ nvidia-smi tidak tersedia atau tidak ada GPU.")
    except Exception as e:
        print(f"⚠️ Gagal membaca info GPU: {e}")


def get_ram_info():
    """Print system RAM usage."""
    try:
        mem = dict((i.split()[0].rstrip(":"), int(i.split()[1])) for i in open("/proc/meminfo").read().splitlines())
        total = mem.get("MemTotal", 0) / 1024 / 1024  # GB
        free = mem.get("MemAvailable", mem.get("MemFree", 0)) / 1024 / 1024
        print(f"🧠 RAM: {free:.1f} GB free / {total:.1f} GB total")
    except Exception as e:
        print(f"⚠️ Gagal membaca RAM: {e}")


def free_gpu():
    """Aggressively free GPU memory between heavy stages."""
    print("🧹 Freeing GPU memory...")
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print("   ✅ torch.cuda.empty_cache() + synchronize() executed.")
        else:
            print("   ℹ️ CUDA tidak tersedia, skip.")
    except Exception as e:
        print(f"   ⚠️ Error freeing GPU: {e}")
    get_gpu_info()


# =============================================================================
# DISK & CLEANUP
# =============================================================================

def get_disk_info(path: str = "/content"):
    """Print disk usage for the given path."""
    try:
        stat = shutil.disk_usage(path)
        free_gb = stat.free / (1024 ** 3)
        total_gb = stat.total / (1024 ** 3)
        print(f"💾 Disk ({path}): {free_gb:.1f} GB free / {total_gb:.1f} GB total")
    except Exception as e:
        print(f"⚠️ Gagal membaca disk: {e}")


def cleanup_temp_files(base_dir: str | None = None, patterns=None):
    """
    Remove temporary files matching common patterns to reclaim disk space.
    Safe — only deletes known temp patterns.
    """
    if base_dir is None:
        base_dir = os.getcwd()
    if patterns is None:
        patterns = [
            "*.ts", "*_silent_*.mp4", "*_ass", "temp_broll_*",
            "h_*.ts", "m_*.ts", "ah_*.ass", "am_*.ass",
            "bgm_*.mp3", "video_asli.mp4", "*.wav",
        ]
    removed = 0
    for pat in patterns:
        for fp in Path(base_dir).glob(pat):
            try:
                fp.unlink()
                removed += 1
            except Exception:
                pass
    if removed:
        print(f"🗑️  Cleanup: removed {removed} temp files from {base_dir}")
    else:
        print("🗑️  Cleanup: no temp files matched.")
    get_disk_info(base_dir)


def cleanup_outputs(base_dir: str | None = None, keep_manifest: bool = True):
    """
    Remove everything inside outputs/ except final MP4/JPG and optionally manifest.
    """
    if base_dir is None:
        base_dir = os.getcwd()
    out_dir = Path(base_dir) / "outputs"
    if not out_dir.exists():
        return
    removed = 0
    for fp in out_dir.rglob("*"):
        if fp.is_dir():
            continue
        name = fp.name.lower()
        if name.endswith(("_ready.mp4", ".jpg")):
            continue
        if keep_manifest and name == "render_manifest.json":
            continue
        try:
            fp.unlink()
            removed += 1
        except Exception:
            pass
    print(f"🗑️  Outputs cleanup: removed {removed} intermediate files. Kept finals & manifest.")


# =============================================================================
# GOOGLE DRIVE
# =============================================================================

def mount_drive(force_remount: bool = False) -> str:
    """Mount Google Drive and return the mounted path."""
    if not _in_colab():
        print("⚠️ Not running in Colab, skipping Drive mount.")
        return ""
    from google.colab import drive
    path = "/content/drive"
    if not os.path.ismount(path) or force_remount:
        drive.mount(path, force_remount=force_remount)
    print(f"✅ Drive mounted at {path}")
    return path


def ensure_drive_cache_dir(drive_path: str = "/content/drive/MyDrive/osc_cache") -> str:
    """Create and return a persistent cache directory on Drive."""
    os.makedirs(drive_path, exist_ok=True)
    print(f"📂 Drive cache dir ready: {drive_path}")
    return drive_path


def copy_models_to_drive_cache(drive_cache: str, base_dir: str | None = None):
    """Copy downloaded models to Drive cache for reuse across sessions."""
    if base_dir is None:
        base_dir = os.getcwd()
    models = ["blaze_face_full_range.tflite", "face_yolov8m.pt"]
    copied = 0
    for m in models:
        src = Path(base_dir) / m
        dst = Path(drive_cache) / m
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
            copied += 1
            print(f"   💾 Cached → Drive: {m}")
    if copied == 0:
        print("   ℹ️ No new models to cache.")


def restore_models_from_drive_cache(drive_cache: str, base_dir: str | None = None):
    """Restore models from Drive cache to working directory."""
    if base_dir is None:
        base_dir = os.getcwd()
    restored = 0
    for m in ["blaze_face_full_range.tflite", "face_yolov8m.pt"]:
        src = Path(drive_cache) / m
        dst = Path(base_dir) / m
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
            restored += 1
            print(f"   📥 Restored from Drive: {m}")
    if restored == 0:
        print("   ℹ️ No cached models found on Drive.")


# =============================================================================
# OUTPUT PACKAGING
# =============================================================================

def zip_outputs(zip_name: str = "outputs.zip", base_dir: str | None = None) -> str:
    """Zip the outputs/ folder and return the zip path."""
    if base_dir is None:
        base_dir = os.getcwd()
    out_dir = Path(base_dir) / "outputs"
    zip_path = Path(base_dir) / zip_name
    if not out_dir.exists():
        print("⚠️ outputs/ directory not found, nothing to zip.")
        return ""
    shutil.make_archive(str(zip_path.with_suffix("")), "zip", root_dir=str(out_dir))
    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"🤐 Zipped outputs → {zip_path} ({size_mb:.1f} MB)")
    return str(zip_path)


def download_outputs(zip_path: str):
    """Trigger Colab file download."""
    if not _in_colab():
        print("⚠️ Not in Colab, skipping browser download.")
        return
    from google.colab import files
    files.download(zip_path)
    print(f"📥 Browser download triggered for {zip_path}")


# =============================================================================
# ONE-STEP SETUP
# =============================================================================

def setup_colab(should_mount_drive: bool = True, drive_cache: bool = True):
    """
    Run all recommended Colab setup steps.
    Call this once at the top of your notebook.
    """
    print("=" * 60)
    print("🚀 OpenSource Clipping — Colab Optimized Setup")
    print("=" * 60)

    get_ram_info()
    get_gpu_info()
    get_disk_info()

    drive_path = ""
    if should_mount_drive:
        drive_path = mount_drive(force_remount=False)
        if drive_cache and drive_path:
            cache = ensure_drive_cache_dir(f"{drive_path}/MyDrive/osc_cache")
            restore_models_from_drive_cache(cache)

    print("=" * 60)
    print("✅ Setup complete. Ready to render!")
    print("=" * 60)
    return drive_path


# =============================================================================
# AUTO-CONFIG
# =============================================================================

def get_colab_defaults():
    """
    Return a dict of CLI-arg overrides optimized for Colab T4.
    Usage:
        args = ["--url", url] + clipping.colab_utils.get_colab_defaults()
        cfg = build_config(args)
    """
    return [
        "--whisper-compute-type", "float16",
        "--whisper-device", "cuda",
        "--face-detector", "mediapipe",        # lighter than YOLO on T4 VRAM
        "--video-preset", "ultrafast",         # fastest CPU fallback preset
        "--video-scale-algo", "bicubic",      # slightly faster than lanczos
        "--words-per-sub", "5",
        "--hook-duration", "3",
    ]
