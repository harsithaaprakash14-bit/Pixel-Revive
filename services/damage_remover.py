"""
services/damage_remover.py
───────────────────────────
Person 1 — LaMa Damage Removal Service.

This module is the production service wrapper for LaMa (Large Mask) inpainting.
It is called as STEP 1 of the AI pipeline by services/ai_connector.py, before
DDColor (Step 2) and Real-ESRGAN (Step 3).

Architecture (mirrors services/colorizer.py):
  - Module-level model cache (_lama) initialised once on first request.
  - load_damage_remover() called once by ai_connector's _get_damage_remover().
  - Single public function restore_image(input_path, output_path) is the API
    contract with ai_connector.py.
  - Post-inference GPU cleanup (empty_cache + gc.collect) before Person 2 starts.

Mask auto-generation:
  LaMa requires a binary mask (white = inpaint, black = keep).  Since no user
  mask is provided, generate_damage_mask() creates one automatically using four
  complementary detection layers:

    Layer 1 — Extreme bright overexposure / tape marks.
    Layer 2 — Severe dark holes / deep stains.
    Layer 3 — High-contrast scratches via Canny edge detection.
    Layer 4 — Paper fold/crease lines via Hessian ridge detection + LSD + Hough.
              This is a full multi-technique pipeline described in detail below.

  All layers are OR-combined, then dilated (3×3) and capped at 10% coverage.

Memory strategy:
  SimpleLama TorchScript model is ~200 MB.  Model stays loaded between requests.
  After each inference: torch.cuda.empty_cache() + gc.collect().

GPU / CPU fallback:
  Primary: GPU.  On any GPU exception: release GPU model, empty_cache, reload on
  CPU, retry.  Flask app is never crashed by a GPU failure.
"""

import gc
import os

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

# Set expandable_segments BEFORE torch so the CUDA allocator fragments less.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch


# ─── Module-level model cache ─────────────────────────────────────────────────
_lama        = None
_lama_device = None


def load_damage_remover():
    """
    Load and cache the SimpleLama inpainting model.

    Weight download:
      simple_lama_inpainting downloads big-lama.pt (~200 MB) automatically
      to ~/.cache/torch/hub/checkpoints/ on first call.

    Returns:
        tuple[SimpleLama, torch.device]
    """
    global _lama, _lama_device
    if _lama is not None:
        return _lama, _lama_device

    from simple_lama_inpainting import SimpleLama

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    label  = "CUDA" if device.type == "cuda" else "CPU"
    print(f"[LaMa] Loading model on {label}...")
    _lama        = SimpleLama(device=device)
    _lama_device = device
    print(f"[LaMa] Model loaded on {device}  (big-lama.pt TorchScript ~200 MB)")
    return _lama, _lama_device


# ─────────────────────────────────────────────
# FACE DETECTION — to protect faces from masking
# ─────────────────────────────────────────────
def _get_face_regions(gray_img):
    """
    Detect face regions in the image.
    Returns a binary mask where faces = 255 (protected), rest = 0.
    Uses OpenCV's built-in Haar cascade — no extra downloads needed.
    """
    face_mask = np.zeros(gray_img.shape, dtype=np.uint8)
    try:
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
        faces = face_cascade.detectMultiScale(
            gray_img, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30)
        )
        for (x, y, w, h) in faces:
            # Add 15% padding around face to protect edges too
            pad_x = int(w * 0.15)
            pad_y = int(h * 0.15)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(gray_img.shape[1], x + w + pad_x)
            y2 = min(gray_img.shape[0], y + h + pad_y)
            face_mask[y1:y2, x1:x2] = 255
        if len(faces) > 0:
            print(f'  [FaceGuard] Protected {len(faces)} face region(s) from masking')
    except Exception as e:
        print(f'  [FaceGuard] Face detection skipped: {e}')
    return face_mask


# ─────────────────────────────────────────────
# CREASE / FOLD LINE DETECTION
# ─────────────────────────────────────────────
def _detect_creases(gray_img):
    """
    Detect paper fold/crease lines using:
    1. Adaptive thresholding to find bright crease highlights
    2. Canny edge detection on blurred image for line structure
    3. Probabilistic Hough Transform to find long straight lines (folds)
    4. Morphological dilation to widen crease mask slightly for full coverage
    """
    h, w = gray_img.shape
    crease_mask = np.zeros((h, w), dtype=np.uint8)

    # --- Method 1: Bright crease highlight detection ---
    # Fold lines often appear as bright streaks in old photos
    blurred = cv2.GaussianBlur(gray_img, (5, 5), 0)
    bright_thresh = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=15, C=-10
    )
    # Keep only elongated bright regions (not small dots/noise)
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
    bright_h = cv2.morphologyEx(bright_thresh, cv2.MORPH_OPEN, kernel_h)
    bright_v = cv2.morphologyEx(bright_thresh, cv2.MORPH_OPEN, kernel_v)
    bright_lines = cv2.bitwise_or(bright_h, bright_v)
    crease_mask = cv2.bitwise_or(crease_mask, bright_lines)

    # --- Method 2: Hough line detection for straight fold lines ---
    edges = cv2.Canny(blurred, threshold1=30, threshold2=100)
    lines = cv2.HoughLinesP(
        edges,
        rho=1, theta=np.pi / 180,
        threshold=80,
        minLineLength=int(min(h, w) * 0.25),  # line must span 25% of image
        maxLineGap=20
    )
    if lines is not None:
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            # Only accept near-horizontal or near-vertical lines (folds are straight)
            angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
            if angle < 15 or angle > 165 or (75 < angle < 105):
                cv2.line(crease_mask, (x1, y1), (x2, y2), 255, thickness=4)

    # --- Method 3: Dark crease shadow detection ---
    # Some folds appear as dark shadows rather than bright highlights
    _, dark_thresh = cv2.threshold(blurred, 20, 255, cv2.THRESH_BINARY_INV)
    dark_h = cv2.morphologyEx(dark_thresh, cv2.MORPH_OPEN, kernel_h)
    dark_v = cv2.morphologyEx(dark_thresh, cv2.MORPH_OPEN, kernel_v)
    dark_lines = cv2.bitwise_or(dark_h, dark_v)
    crease_mask = cv2.bitwise_or(crease_mask, dark_lines)

    # Dilate to ensure full crease width is covered
    dilation_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    crease_mask = cv2.dilate(crease_mask, dilation_kernel, iterations=2)

    return crease_mask


# ─────────────────────────────────────────────
# SCRATCH & STAIN DETECTION
# ─────────────────────────────────────────────
def _detect_scratches_and_stains(gray_img):
    """
    Detect random scratches and stains using:
    1. Gaussian blur difference to find sharp local anomalies
    2. Morphological closing to merge nearby damage regions
    """
    scratch_mask = np.zeros(gray_img.shape, dtype=np.uint8)

    # Unsharp mask difference — reveals fine scratches
    blur_light = cv2.GaussianBlur(gray_img, (3, 3), 0)
    blur_heavy = cv2.GaussianBlur(gray_img, (21, 21), 0)
    diff = cv2.absdiff(blur_light, blur_heavy)
    _, scratch_thresh = cv2.threshold(diff, 35, 255, cv2.THRESH_BINARY)

    # Remove tiny noise dots — keep only real damage shapes
    noise_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    scratch_thresh = cv2.morphologyEx(scratch_thresh, cv2.MORPH_OPEN, noise_kernel)

    # Stain detection — large uniform discoloured regions
    blur_stain = cv2.GaussianBlur(gray_img, (31, 31), 0)
    diff_stain = cv2.absdiff(gray_img, blur_stain)
    _, stain_thresh = cv2.threshold(diff_stain, 55, 255, cv2.THRESH_BINARY)
    stain_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    stain_thresh = cv2.morphologyEx(stain_thresh, cv2.MORPH_CLOSE, stain_kernel)

    scratch_mask = cv2.bitwise_or(scratch_thresh, stain_thresh)
    return scratch_mask


# ─── Main damage mask generator ───────────────────────────────────────────────

def generate_damage_mask(img_bgr: np.ndarray) -> np.ndarray:
    """
    Auto-generate a conservative binary damage mask from a photograph.
    Combines creases, scratches and stains, protects detected face regions,
    and caps the coverage at 10%.

    Returns:
        np.ndarray uint8 (H, W): 255 = damaged region, 0 = clean region.
    """
    gray = (cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            if len(img_bgr.shape) == 3 else img_bgr.copy())

    print(f"  [Mask] Analysing image ({img_bgr.shape[1]}x{img_bgr.shape[0]})...")

    # Detect creases
    crease_mask = _detect_creases(gray)
    crease_px = np.sum(crease_mask > 0)
    print(f"  [Mask] Crease detection: {crease_px} pixels flagged")

    # Detect scratches and stains
    scratch_mask = _detect_scratches_and_stains(gray)
    scratch_px = np.sum(scratch_mask > 0)
    print(f"  [Mask] Scratch/stain detection: {scratch_px} pixels flagged")

    # Combine detections
    combined = cv2.bitwise_or(crease_mask, scratch_mask)

    # Protect face regions
    face_region = _get_face_regions(gray)
    if np.any(face_region > 0):
        combined = cv2.bitwise_and(combined, cv2.bitwise_not(face_region))

    # Final cleanup — remove tiny isolated noise pixels
    cleanup_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    final_mask = cv2.morphologyEx(combined, cv2.MORPH_OPEN, cleanup_kernel)

    # Coverage cap: erode until mask <= 10%
    MAX_COV      = 0.10
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    coverage     = (final_mask > 0).sum() / final_mask.size

    n_erode = 0
    while coverage > MAX_COV and n_erode < 30:
        final_mask = cv2.erode(final_mask, erode_kernel, iterations=1)
        coverage   = (final_mask > 0).sum() / final_mask.size
        n_erode   += 1

    if n_erode > 0:
        print(f"[LaMa] Mask eroded {n_erode}x to cap coverage below 10%")

    return final_mask



# ─── Public API ───────────────────────────────────────────────────────────────

def restore_image(input_path: str, output_path: str) -> str:
    """
    Detect and remove damage from a photograph using LaMa inpainting.

    Called by services/ai_connector.py as Step 1 of the pipeline.

    Parameters:
        input_path  (str): Path to the original uploaded image.
        output_path (str): Path where the restored image will be saved.

    Returns:
        str: output_path.

    Raises:
        FileNotFoundError: If input_path does not exist.
        RuntimeError: If inference fails on both GPU and CPU.
        Never calls sys.exit().
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"[LaMa] Input file not found: {input_path}")

    img_bgr = cv2.imread(input_path)
    if img_bgr is None:
        raise RuntimeError(f"[LaMa] Could not read image: {input_path}")

    original_h, original_w = img_bgr.shape[:2]
    print(f"[LaMa] Processing {os.path.basename(input_path)} ({original_w}x{original_h}px)...")

    mask_np   = generate_damage_mask(img_bgr)
    final_pct = (mask_np > 0).sum() / mask_np.size * 100
    print(f"[LaMa] Final damage mask: {final_pct:.2f}% of pixels flagged")

    LAMA_SIZE = 512
    img_rgb   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(img_rgb).resize((LAMA_SIZE, LAMA_SIZE), Image.LANCZOS)
    pil_mask  = Image.fromarray(mask_np).resize((LAMA_SIZE, LAMA_SIZE), Image.NEAREST)

    result_pil  = _run_lama_inference(pil_image, pil_mask)
    result_full = result_pil.resize((original_w, original_h), Image.LANCZOS)

    result_full.save(output_path)
    size_kb = os.path.getsize(output_path) / 1024
    print(f"[LaMa] Restored image saved -> {output_path}  ({size_kb:.1f} KB)")
    return output_path


def _run_lama_inference(pil_image: Image.Image, pil_mask: Image.Image) -> Image.Image:
    """
    Run SimpleLama inference with automatic GPU -> CPU fallback.

    Never calls sys.exit().
    Raises RuntimeError if both GPU and CPU fail.
    """
    global _lama, _lama_device

    gpu_exc    = None
    lama_model, device = load_damage_remover()

    # ── GPU attempt ───────────────────────────────────────────────────────
    if device.type == "cuda":
        try:
            if hasattr(lama_model, 'model') and lama_model.model is not None:
                print("  [LaMa] Moving model to GPU for inference...")
                lama_model.model.to(device)
            print(f"[LaMa] Running inference on {device}...")
            result = lama_model(pil_image, pil_mask)
            print("[LaMa] GPU inference complete.")
            return result
        except Exception as exc:
            gpu_exc = exc
            print(f"[LaMa] GPU inference failed ({exc}). Freeing GPU model...")
            del _lama
            _lama        = None
            _lama_device = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        finally:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            gc.collect()

    # ── CPU fallback ──────────────────────────────────────────────────────
    try:
        from simple_lama_inpainting import SimpleLama
        cpu_device = torch.device("cpu")
        print("[LaMa] Loading model on CPU for fallback inference...")
        cpu_lama = SimpleLama(device=cpu_device)
        print("[LaMa] Running inference on CPU (may take 30-120s)...")
        result = cpu_lama(pil_image, pil_mask)
        print("[LaMa] CPU inference complete.")
        _lama        = cpu_lama
        _lama_device = cpu_device
        return result
    except Exception as cpu_exc:
        raise RuntimeError(
            f"LaMa inpainting failed on both GPU and CPU.\n"
            f"  GPU error : {gpu_exc}\n"
            f"  CPU error : {cpu_exc}"
        ) from cpu_exc
    finally:
        gc.collect()


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python3 damage_remover.py <input_image> <output_image>")
        sys.exit(1)
    
    # Force CUDA empty cache on startup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    restore_image(sys.argv[1], sys.argv[2])
