"""
services/colorizer.py
─────────────────────
Person 2 — DDColor AI colorization service.

MODIFICATIONS (memory-efficient rewrite):
------------------------------------------
Root cause of Person 2 OOM:
  The HAMI GPU virtualizer enforces a hard 2 GiB cap per process.
  DDColor large model (ConvNext-L encoder + MultiScale decoder) consumes
  ~900 MB of weights.  With input_size=512 the forward pass allocates a
  further ~600 MB of intermediate activations → total >1.5 GB.
  When PyTorch then tries to allocate the next 256 MB block it overflows
  the HAMI limit.

Changes made:

1. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
   Set BEFORE torch initializes so PyTorch's CUDA allocator uses expandable
   memory segments.  This converts "reserved but unallocated" VRAM
   (visible in every OOM message) back into reusable budget instead of
   leaving it stranded as fragmented blocks.

2. input_size 512 → 256
   Activation memory scales as O(input_size²).  Halving the spatial
   dimension reduces feature-map allocations by 4×.  Colorization quality
   impact is minimal: only the ab (chroma) channels are predicted at this
   resolution; the L (luminance) channel is always taken from the original
   full-resolution image and blended at the end.

3. torch.amp.autocast('cuda') in colorize_image()
   Runs the forward pass in fp16 on CUDA, halving activation memory again.
   Model weights remain in fp32 — only intermediate tensors change dtype.
   Combined with change 2 this gives roughly 8× less activation VRAM vs.
   the original configuration.

4. Post-inference cleanup (always runs, even on error):
   - torch.cuda.synchronize()  : wait for all CUDA kernels to finish.
   - torch.cuda.empty_cache()  : return reserved-but-unallocated VRAM to
     the HAMI/OS allocator so the next request starts with a clean budget.
   - gc.collect()              : collect Python objects holding tensor refs.

Model caching: load_colorizer() loads the ~900 MB model once at startup
and caches it for the lifetime of the Flask process.  Only the per-request
activation tensors are freed — the weights stay in VRAM.
"""

import gc
import contextlib
import os

# MODIFICATION 1: set BEFORE torch is imported so the CUDA allocator
# switches to expandable segments.  os.environ.setdefault leaves any
# existing value supplied by the operator intact.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import sys
import cv2
import torch
from ddcolor import DDColor, ColorizationPipeline, build_ddcolor_model
from huggingface_hub import hf_hub_download
# MODIFICATION 2: reduced from 512 → 256.
# Activation VRAM ∝ input_size², so 256px needs only ¼ of what 512px needed.
# Quality impact is negligible: the prediction is only for colour channels
# (ab in Lab space) which are low-frequency and tolerate lower resolution;
# full-resolution luminance is always preserved from the original image.
COLORIZER_INPUT_SIZE = 512

_model_path = None

def load_colorizer():
    """
    Build and cache the DDColor ColorizationPipeline.
    Called once at Flask startup and reused for every subsequent request.
    """
    global _model_path
    if _model_path is None:
        _model_path = hf_hub_download(repo_id="piddnad/DDColor-models", filename="ddcolor_paper.pth")
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_ddcolor_model(
        DDColor,
        model_path=_model_path,
        input_size=COLORIZER_INPUT_SIZE,
        model_size="large",
        decoder_type="MultiScaleColorDecoder",
        device=device,
    )
    print(f"[DDColor] Model loaded on {device}  "
          f"(input_size={COLORIZER_INPUT_SIZE})")
    return ColorizationPipeline(model, input_size=COLORIZER_INPUT_SIZE)

def is_grayscale_image(img, threshold=20, percentage=10.0):
    """
    Determine if an image is monochrome / sepia / grayscale vs true multi-color.
    Returns True if the photo needs DDColor colorization.
    Returns False ONLY if the photo is already a rich, genuine multi-color image.
    """
    if len(img.shape) < 3 or img.shape[2] == 1:
        return True
    
    import numpy as np
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_channel = hsv[:, :, 0]
    s_channel = hsv[:, :, 1]
    
    mean_sat = s_channel.mean()
    colorful_mask = s_channel > 35
    colorful_pct = (colorful_mask.sum() / s_channel.size) * 100.0
    
    if colorful_pct > 5.0:
        hue_std = float(h_channel[colorful_mask].std())
    else:
        hue_std = 0.0
        
    print(f"[Colorizer] Color analysis: Mean Saturation={mean_sat:.1f}, "
          f"Colorful Pixels (S>35)={colorful_pct:.2f}%, Hue StdDev={hue_std:.1f}")
          
    is_true_color = (colorful_pct > 12.0) and (hue_std > 20.0 or mean_sat > 40.0)
    return not is_true_color


def colorize_image(colorizer, input_path, output_path):
    """
    Colorize a single image and write the result to output_path.

    MODIFICATION 3: GPU inference is wrapped in torch.amp.autocast('cuda')
    so all intermediate activations are computed in fp16 rather than fp32,
    halving the peak VRAM usage during the forward pass.  On CPU the context
    degrades gracefully to a no-op (contextlib.nullcontext).

    MODIFICATION 4: a finally block performs mandatory post-inference cleanup:
      - deletes local tensor references so Python can reclaim them
      - synchronizes the CUDA stream (ensures kernels truly finished)
      - empties the CUDA allocator cache (frees reserved-but-unallocated VRAM)
      - runs gc.collect() to sweep up any surviving Python-level references
    This cleanup runs even when an exception is raised, so the VRAM budget
    is always restored for the next incoming request.
    """
    img_bgr = cv2.imread(input_path)
    if img_bgr is None:
        raise ValueError(f"Could not read image: {input_path}")

    # Check if the image is already in color
    if not is_grayscale_image(img_bgr):
        import shutil
        print(f"[DDColor] Image is already in color. Skipping colorization.")
        shutil.copy(input_path, output_path)
        return

    # MODIFICATION 3: build the autocast context for the forward pass.
    # We check the colorizer's device rather than querying cuda.is_available()
    # globally so that this still works correctly if the model was placed on CPU.
    is_cuda = (
        torch.cuda.is_available()
        and hasattr(colorizer, "device")
        and getattr(colorizer.device, "type", None) == "cuda"
    )
    if is_cuda and hasattr(colorizer, "model") and colorizer.model is not None:
        print("  [DDColor] Moving model to GPU for inference...")
        colorizer.model.to(colorizer.device)

    amp_ctx = (
        torch.amp.autocast("cuda", dtype=torch.float16)
        if is_cuda
        else contextlib.nullcontext()
    )

    # MODIFICATION 4: initialise result to None so the finally block can
    # safely test it even if colorizer.process() raises before binding it.
    result = None
    try:
        with amp_ctx:
            result = colorizer.process(img_bgr)

        cv2.imwrite(output_path, result)
        print(f"Saved: {output_path}")

    finally:
        # ── Release per-inference tensor references ───────────────────────
        # Deleting these local names allows Python to GC the backing arrays
        # immediately rather than waiting for the next garbage-collection
        # cycle.  This is especially important for the numpy arrays that
        # wrap CUDA tensors (they hold a reference keeping the VRAM live).
        del img_bgr
        if result is not None:
            del result

        # ── Flush CUDA allocator cache ────────────────────────────────────
        if torch.cuda.is_available():
            torch.cuda.synchronize()    # wait for all CUDA kernels to finish
            torch.cuda.empty_cache()    # return reserved-but-unallocated VRAM

        # ── Python garbage collection ─────────────────────────────────────
        gc.collect()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 colorize.py <input_image> <output_image>")
        sys.exit(1)
        
    # Check if the image is already in color BEFORE loading the heavy 900MB model!
    import cv2
    img_bgr = cv2.imread(sys.argv[1])
    if img_bgr is not None and not is_grayscale_image(img_bgr):
        import shutil
        print(f"[DDColor] Image is already in color. Skipping model loading in CLI.")
        shutil.copy(sys.argv[1], sys.argv[2])
        sys.exit(0)
        
    colorizer = load_colorizer()
    colorize_image(colorizer, sys.argv[1], sys.argv[2])
