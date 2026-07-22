"""
PixelRevive - Person 3: Super-Resolution Upscaling Module
============================================================
This script takes a photo (already cleaned + colorised by Person 1 & 2)
and upscales it to HD/4K quality using the Real-ESRGAN AI model.

MODIFICATIONS (full memory-resilient rewrite):
-----------------------------------------------

Root causes of failure in the original script:
  A. torch.OutOfMemoryError propagated uncaught → subprocess exit code 1.
  B. "CPU fallback" still caused CUDA OOM because HAMI (Hardware Accelerator
     Manager Interface) intercepts ALL large memory allocations at the OS level
     once a CUDA context has been created in the process — even allocations
     that PyTorch nominally makes on CPU RAM via cudaMallocManaged.
     The upsampler.enhance() call returned "CPU error: CUDA out of memory"
     precisely because HAMI was counting the CPU-path tensor allocations
     against the GPU budget of the already-initialised CUDA context.

  Solution to (B): spawn a grandchild subprocess with CUDA_VISIBLE_DEVICES=""
  BEFORE any torch/CUDA initialisation.  In that child process torch.cuda.
  is_available() returns False, no CUDA context is ever created, HAMI never
  intercepts, and inference runs truly on CPU RAM.

Changes:

1. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
   Set at module level before torch is imported, reducing CUDA memory
   fragmentation for the GPU retry loop.

2. GPU_TILE_SIZES = [64, 32, 16]
   Starts at the smallest practical GPU tile sizes (512/256/128 all OOM in
   the 2 GiB HAMI environment so skipping them saves time).  TILE_PAD=0
   to maximise available tile budget.

3. _is_oom(exc) helper
   Detects OOM from both torch.OutOfMemoryError (torch ≥ 2.0) and the
   legacy RuntimeError("out of memory") from older torch builds.

4. upscale_with_retry(img, scale, input_path, output_path, fmt)
   GPU retry loop with automatic VRAM flush between attempts.
   After all GPU tiles are exhausted, calls _run_cpu_subprocess() instead
   of attempting in-process CPU inference (which HAMI would still intercept).

5. _run_cpu_subprocess(input_path, output_path, scale, fmt)
   Spawns a grandchild subprocess of THIS script with:
     - CUDA_VISIBLE_DEVICES=""  → torch sees no GPU, HAMI never engages
     - --cpu-only flag          → grandchild skips GPU path entirely
   The grandchild performs true CPU inference, writes output_path, exits 0.

6. --cpu-only CLI flag + _cpu_only_inference()
   When this flag is present (set by the grandchild invocation), the script
   sets CUDA_VISIBLE_DEVICES="" at the very start of main() — before any
   function that imports torch — so the CUDA runtime is never initialised.
   _cpu_only_inference() then loads the model to CPU and runs enhance().

7. upscale_image() signature preserved (upsampler arg accepted but unused).
   Raises exceptions instead of returning None so main() can detect failure.

8. main() wrapped in try/except:
   - sys.exit(0) on success (services/upscaler.py checks returncode == 0)
   - sys.exit(1) on failure (error sent to stderr)
   sys.exit() is called ONLY here, never inside reusable functions.
"""

import os
import sys
import argparse
import subprocess
import time
import urllib.request

# MODIFICATION 1: set PYTORCH_CUDA_ALLOC_CONF before torch is imported
# so PyTorch's CUDA allocator uses expandable segments, reducing the
# fragmentation that causes "reserved but unallocated" OOM.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import cv2
import numpy as np


# ─── Constants ────────────────────────────────────────────────────────────────

MODEL_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/"
    "v0.1.0/RealESRGAN_x4plus.pth"
)
# Absolute path: resolves to PixelRevive/models/ regardless of cwd
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "models",
    "RealESRGAN_x4plus.pth"
)
DEFAULT_SCALE = 4

# MODIFICATION 2: GPU tile sizes to try in descending order.
# Sizes 512/256/128 all OOM against the 2 GiB HAMI budget once CUDA context
# overhead (~300 MB) + model weights (~64 MB) are counted.  Starting at 64
# avoids wasting time on sizes that are guaranteed to fail in this environment.
# After all GPU tiles are exhausted the script spawns a CPU-only subprocess.
GPU_TILE_SIZES = [256, 128, 64]
TILE_PAD = 10


# ─── Model download ───────────────────────────────────────────────────────────

def download_model():
    """Download Real-ESRGAN weights if not already present."""
    if os.path.exists(MODEL_PATH):
        print(f"[OK] Model already downloaded: {MODEL_PATH}")
        return
    print("[..] Downloading Real-ESRGAN model (~64 MB)...")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print(f"[OK] Model saved to: {MODEL_PATH}")


# ─── Model loader ─────────────────────────────────────────────────────────────

def load_model(scale=4, device="cuda", tile=64, tile_pad=TILE_PAD):
    """
    Load RealESRGANer onto the specified device.

    MODIFICATION: added `device`, `tile`, and `tile_pad` parameters so that:
      - GPU retry attempts pass progressively smaller tile sizes.
      - CPU fallback uses device="cpu", half=False (fp16 is CUDA-only),
        gpu_id=None (RealESRGANer's signal to use CPU mode).

    gpu_id=None → RealESRGANer sets self.device = torch.device('cpu').
    half=True is only valid on CUDA; passing it on CPU raises RuntimeError.
    """
    from realesrgan import RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            f"Model not found at: {MODEL_PATH}. "
            "Copy RealESRGAN_x4plus.pth to the models/ folder."
        )

    use_half = (device == "cuda")   # fp16 is CUDA-only; CPU must use fp32
    gpu_id   = 0 if device == "cuda" else None  # None → CPU mode

    tile_label = str(tile) if tile else "none (full image)"
    print(
        f"[..] Loading Real-ESRGAN on {device.upper()} "
        f"(half={use_half}, tile={tile_label})..."
    )

    model = RRDBNet(
        num_in_ch=3, num_out_ch=3, num_feat=64,
        num_block=23, num_grow_ch=32, scale=scale
    )
    upsampler = RealESRGANer(
        scale=scale,
        model_path=MODEL_PATH,
        model=model,
        tile=tile,
        tile_pad=tile_pad,
        pre_pad=0,
        half=use_half,
        gpu_id=gpu_id,
    )
    print(f"[OK] Model loaded on {device.upper()}! Ready to upscale at {scale}x")
    return upsampler


# ─── Image quality helpers (unchanged) ───────────────────────────────────────

def sharpness_score(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return round(cv2.Laplacian(gray, cv2.CV_64F).var(), 2)


def print_quality_report(original, upscaled, scale):
    orig_h, orig_w = original.shape[:2]
    up_h,   up_w   = upscaled.shape[:2]
    orig_sharp  = sharpness_score(original)
    up_sharp    = sharpness_score(upscaled)
    improvement = ((up_sharp - orig_sharp) / max(orig_sharp, 1)) * 100

    print("\n" + "-" * 45)
    print("  QUALITY REPORT")
    print("-" * 45)
    print(f"  Original  -> {orig_w}x{orig_h} px  |  Sharpness: {orig_sharp}")
    print(f"  Upscaled  -> {up_w}x{up_h} px  |  Sharpness: {up_sharp}")
    print(f"  Scale     -> {scale}x upscale")
    print(f"  Improvement -> +{improvement:.1f}% sharper")
    print("-" * 45 + "\n")


def save_image(image_bgr, output_path, fmt="PNG"):
    """Save the upscaled array and print file size.  Returns the final path."""
    fmt = fmt.upper()
    if fmt in ("JPEG", "JPG"):
        output_path = output_path.rsplit(".", 1)[0] + ".jpg"
        cv2.imwrite(output_path, image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    else:
        output_path = output_path.rsplit(".", 1)[0] + ".png"
        cv2.imwrite(output_path, image_bgr)
    size_kb = os.path.getsize(output_path) / 1024
    print(f"  [OK] Saved -> {output_path}  ({size_kb:.1f} KB)")
    return output_path


# ─── OOM detection ────────────────────────────────────────────────────────────

def _is_oom(exc):
    """
    MODIFICATION 3: Return True if exc is a CUDA out-of-memory error.

    Handles both:
    - torch.OutOfMemoryError  (torch >= 2.0)
    - RuntimeError("out of memory")  (older torch builds)

    torch is imported lazily here so the helper works even if torch is not
    installed (fallback to string check).
    """
    try:
        import torch
        if isinstance(exc, torch.OutOfMemoryError):
            return True
    except ImportError:
        pass
    return "out of memory" in str(exc).lower()


# ─── CPU-only inference (grandchild subprocess) ───────────────────────────────

def _cpu_only_inference(input_path, output_path, scale, fmt):
    """
    MODIFICATION 6: True CPU inference.

    Called from main() when the --cpu-only flag is set.  By this point
    CUDA_VISIBLE_DEVICES="" has already been placed in the environment
    (by the parent _run_cpu_subprocess call) and re-enforced at the start
    of main(), so torch.cuda.is_available() returns False and no CUDA
    context is ever created.  HAMI therefore never intercepts any
    allocation made here, and the inference runs on pure system RAM.

    Design choices:
    - tile=128: process the image in 128px tiles to keep peak RAM usage
      within Render's container memory limits (~512 MB – 2 GB).  Using
      tile=0 (whole image) on a 1024x1024 input requires several GB of
      intermediate RRDB activations and OOM-kills the subprocess.
    - tile_pad=10: small overlap between tiles to avoid visible seams.
    - half=False: fp16 has no hardware benefit on CPU and can degrade
      results; fp32 is used for correctness and compatibility.
    """
    import torch

    # Defensive assertion — this code path must only execute when CUDA is
    # genuinely hidden.  If this fires, the parent did not set
    # CUDA_VISIBLE_DEVICES="" correctly before spawning.
    if torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is visible in --cpu-only mode.  "
            "CUDA_VISIBLE_DEVICES should be empty string."
        )

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input not found: {input_path}")

    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Could not read image: {input_path}")

    print(f"\n  [..] CPU inference: {os.path.basename(input_path)}")

    # Load model to CPU with tiling to prevent OOM on memory-limited containers.
    # tile=128 keeps peak RAM well under 2 GB even for 1024x1024 inputs.
    upsampler = load_model(scale=scale, device="cpu", tile=128, tile_pad=10)

    start = time.time()
    with torch.inference_mode():
        output, _ = upsampler.enhance(img, outscale=scale)
    elapsed = time.time() - start

    print(f"  [OK] CPU inference done in {elapsed:.1f}s")
    print_quality_report(img, output, scale)
    save_image(output, output_path, fmt)


def _run_cpu_subprocess(input_path, output_path, scale, fmt):
    """
    MODIFICATION 5: Spawn a grandchild subprocess with CUDA_VISIBLE_DEVICES="".

    Why a subprocess and not an in-process call:
      Once a CUDA context has been initialised in a process (which happens as
      soon as any GPU tile attempt calls load_model), HAMI hooks into every
      subsequent large memory allocation — even ones PyTorch routes to CPU RAM.
      This causes the bogus "CPU error: CUDA out of memory" errors seen in logs.

      The only way to perform truly CUDA-free inference is to start a fresh
      process where CUDA_VISIBLE_DEVICES="" prevents the CUDA runtime from
      ever loading.  In that process:
        - torch.cuda.is_available() → False
        - No CUDA context is created
        - HAMI has nothing to intercept
        - All tensor operations run on CPU RAM with no GPU budget consumed

    The grandchild runs THIS SAME SCRIPT with --cpu-only, so it benefits from
    all the same model-loading and quality-report logic.  Its stdout/stderr
    are streamed back and printed to the parent's terminal.

    Returns True on success.  Raises RuntimeError on failure.
    Never calls sys.exit().
    """
    script_path = os.path.abspath(__file__)
    cwd         = os.path.dirname(script_path)

    # Build grandchild environment: completely hide all GPUs
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""  # CUDA_VISIBLE_DEVICES="" → no GPUs

    cmd = [
        sys.executable,    # same conda env Python that launched this script
        script_path,
        "--input",   input_path,
        "--output",  output_path,
        "--scale",   str(scale),
        "--format",  fmt,
        "--cpu-only",          # tell grandchild to skip GPU entirely
    ]

    print("[..] Spawning CPU-only subprocess (CUDA_VISIBLE_DEVICES='')...")

    result = subprocess.run(
        cmd,
        env=env,
        capture_output=True,
        text=True,
        cwd=cwd,
    )

    # Echo grandchild output to parent terminal for full visibility
    if result.stdout.strip():
        for line in result.stdout.strip().splitlines():
            print(f"  [CPU] {line}")
    if result.stderr.strip():
        for line in result.stderr.strip().splitlines():
            print(f"  [CPU STDERR] {line}", file=sys.stderr)

    if result.returncode == 0:
        print("[OK] CPU subprocess completed successfully.")
        return True

    raise RuntimeError(
        f"CPU subprocess failed (exit code {result.returncode}). "
        "See [CPU STDERR] lines above for the error."
    )


# ─── Core: GPU retry + CPU subprocess fallback ────────────────────────────────

def upscale_with_retry(img, scale, input_path, output_path, fmt):
    """
    MODIFICATION 4: Memory-resilient upscaling.

    Algorithm:
    1. If CUDA is available, iterate GPU_TILE_SIZES from largest → smallest.
       Before each attempt: flush VRAM with torch.cuda.empty_cache().
       On OOM: delete upsampler, flush VRAM again, try next smaller tile.
       On non-OOM GPU error: log and break (smaller tile won't help).
    2. After all GPU tiles are exhausted (for-else) or a non-OOM GPU error
       breaks the loop: call _run_cpu_subprocess() which spawns a grandchild
       process with CUDA_VISIBLE_DEVICES="" for true CPU inference.
    3. On CPU subprocess success: return None to signal that output_path has
       already been written by the grandchild.
    4. On CPU subprocess failure: raise RuntimeError.

    This function NEVER calls sys.exit().
    """
    import torch

    # ── GPU retry loop ────────────────────────────────────────────────────
    if torch.cuda.is_available():
        for tile_size in GPU_TILE_SIZES:
            print(f"[..] GPU attempt: tile_size={tile_size}px ...")

            # Flush any VRAM left by previous attempt or DDColor subprocess
            torch.cuda.empty_cache()

            upsampler = None
            try:
                upsampler = load_model(
                    scale=scale, device="cuda",
                    tile=tile_size, tile_pad=TILE_PAD
                )
                output, _ = upsampler.enhance(img, outscale=scale)
                # ── GPU SUCCESS ───────────────────────────────────────────
                print(f"[OK] GPU inference succeeded (tile={tile_size}px)")
                return output   # caller will save this numpy array

            except Exception as exc:
                if _is_oom(exc):
                    print(
                        f"[!!] OOM at tile={tile_size}px — "
                        "freeing VRAM, retrying with smaller tile..."
                    )
                    # continue to next tile size
                else:
                    # Non-OOM error: a smaller tile won't fix this
                    print(
                        f"[!!] Non-OOM GPU error at tile={tile_size}px: "
                        f"{exc}\n     Skipping remaining GPU sizes."
                    )
                    break   # exit loop → go directly to CPU subprocess

            finally:
                # MODIFICATION: explicitly delete upsampler and call
                # empty_cache() in the finally block so the GPU tensors
                # are freed before the next iteration starts, regardless
                # of whether enhance() succeeded or raised.
                if upsampler is not None:
                    del upsampler
                torch.cuda.empty_cache()

        else:
            # for-else: loop ran to completion without 'break'
            # → every GPU tile size was tried and all OOM'd
            print("[!!] All GPU tile sizes exhausted.")

    else:
        print("[..] CUDA not available in this process.")

    # ── CPU subprocess fallback ───────────────────────────────────────────
    # MODIFICATION 5: spawn a grandchild with CUDA_VISIBLE_DEVICES="" so
    # HAMI never intercepts its memory allocations.
    print("[..] Falling back to CPU subprocess inference...")
    _run_cpu_subprocess(input_path, output_path, scale, fmt)

    # Signal to caller that the grandchild already wrote output_path
    return None


# ─── Public API (signature unchanged) ─────────────────────────────────────────

def upscale_image(upsampler, input_path, output_path, scale=4, fmt="PNG"):
    """
    Upscale a single image and save the result.

    MODIFICATION: `upsampler` parameter is accepted for API compatibility
    but is no longer used — model loading and retries happen inside
    upscale_with_retry().

    MODIFICATION: Raises FileNotFoundError / RuntimeError instead of
    returning None so callers can detect failures reliably.

    Return value:
      str — path to the saved upscaled image.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"\n  [..] Processing: {os.path.basename(input_path)}")
    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Could not read image: {input_path}")

    start = time.time()

    # MODIFICATION 4: delegate to upscale_with_retry
    output = upscale_with_retry(img, scale, input_path, output_path, fmt)

    elapsed = time.time() - start
    print(f"  [OK] Pipeline completed in {elapsed:.1f}s")

    if output is None:
        # MODIFICATION 5: CPU subprocess already wrote to output_path.
        # Compute the final path using the same extension logic as save_image().
        fmt_upper = fmt.upper()
        if fmt_upper in ("JPEG", "JPG"):
            final_path = output_path.rsplit(".", 1)[0] + ".jpg"
        else:
            final_path = output_path.rsplit(".", 1)[0] + ".png"
        return final_path

    # GPU path: save the numpy result the normal way
    print_quality_report(img, output, scale)
    return save_image(output, output_path, fmt)


def run_pipeline(input_path, output_path, scale=4, fmt="PNG"):
    """
    Convenience function for direct Python callers.
    Returns output path on success; raises on failure.
    """
    download_model()
    return upscale_image(None, input_path, output_path, scale, fmt)


# ─── CLI entry point ──────────────────────────────────────────────────────────

def main():
    """
    MODIFICATION 8: Wrapped in try/except for clean exit codes.

    MODIFICATION 6: --cpu-only flag.
    When present (set by _run_cpu_subprocess() when spawning the grandchild),
    CUDA_VISIBLE_DEVICES="" is enforced at the very top of main() — before
    any function that imports torch — so the CUDA runtime is never initialised
    in this process.  All subsequent operations are purely on CPU RAM and HAMI
    does not intercept any allocations.

    Exit codes:
      0 — success (output file written)
      1 — failure (error printed to stderr)

    sys.exit() is called ONLY here, never inside upscale_image() or any
    other reusable function (so run_pipeline() is safe to call in-process).
    """
    parser = argparse.ArgumentParser(description="PixelRevive Person 3 Upscaler")
    parser.add_argument("--input",    type=str, required=True,
                        help="Path to the input image")
    parser.add_argument("--output",   type=str, required=True,
                        help="Destination path for the upscaled image")
    parser.add_argument("--scale",    type=int, default=4, choices=[2, 4],
                        help="Upscale factor (default: 4)")
    parser.add_argument("--format",   type=str, default="PNG",
                        choices=["PNG", "JPEG"],
                        help="Output format (default: PNG)")
    # MODIFICATION 6: --cpu-only flag consumed by grandchild subprocesses
    parser.add_argument("--cpu-only", action="store_true",
                        help="Force CPU inference — CUDA_VISIBLE_DEVICES must "
                             "be empty before torch is imported")
    args = parser.parse_args()

    # ── CPU-only branch ───────────────────────────────────────────────────
    # MODIFICATION 6: handle --cpu-only BEFORE any function that imports torch.
    # The parent _run_cpu_subprocess already set CUDA_VISIBLE_DEVICES="" in the
    # environment before exec'ing this process.  We enforce it again here for
    # safety and to make it self-documenting.
    if args.cpu_only:
        # Re-enforce: CUDA must be hidden before torch ever initialises.
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        try:
            download_model()   # no torch import — just filesystem checks
            _cpu_only_inference(
                args.input, args.output, args.scale, args.format
            )
            print("[OK] Done! (CPU mode)")
            sys.exit(0)
        except Exception as err:
            print(f"[FATAL] CPU upscaler error: {err}", file=sys.stderr)
            sys.exit(1)

    # ── Normal GPU-with-retry branch ──────────────────────────────────────
    try:
        download_model()
        # upsampler=None: upscale_image ignores it and loads internally
        upscale_image(None, args.input, args.output, args.scale, args.format)
        print("[OK] Done!")
        sys.exit(0)   # explicit exit(0) — services/upscaler.py checks returncode

    except Exception as err:
        # Print to stderr so services/upscaler.py captures it in result.stderr
        print(f"[FATAL] Upscaler error: {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
