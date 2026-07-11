"""
PixelRevive — Module 1: Scratch & Damage Removal (v2.0 — Production)
=====================================================================
Member: A (Blaze Instance)
Model:  LaMa (Large Mask Inpainting) via simple-lama-inpainting

WHAT'S NEW in v2.0:
  - Auto fold/crease line detection using OpenCV (no manual mask needed)
  - Face-aware masking — detected faces are excluded from the mask
  - Scratch and stain detection added alongside crease detection
  - Manual mask still supported as optional override
  - Two modes:
      1. Auto mode  — detects damage automatically from the photo
      2. Manual mode — uses a provided mask (same as v1.0)

Usage:
    from damage_removal import restore_photo, generate_damage_mask

    # AUTO mode (recommended) — detects creases + scratches automatically
    result = restore_photo('old_photo.jpg', output_path='restored.png')

    # MANUAL mode — provide your own mask
    result = restore_photo('old_photo.jpg', mask_path='mask.png', output_path='restored.png')

    # Just generate the mask to inspect it
    mask = generate_damage_mask('old_photo.jpg', save_mask_path='mask_preview.png')
"""

import cv2
import numpy as np
from PIL import Image
from simple_lama_inpainting import SimpleLama

# Load LaMa model once at import time
lama = SimpleLama()


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


# ─────────────────────────────────────────────
# MAIN MASK GENERATOR
# ─────────────────────────────────────────────
def generate_damage_mask(input_path, save_mask_path=None,
                          detect_creases=True,
                          detect_scratches=True,
                          crease_sensitivity=1.0):
    """
    Automatically generate a damage mask from a photo.
    Detects fold lines, creases, scratches and stains.
    Protects detected face regions from being masked.

    Parameters:
        input_path         (str):   Path to input photo
        save_mask_path     (str):   Optional — save mask preview to this path
        detect_creases     (bool):  Enable fold/crease detection (default True)
        detect_scratches   (bool):  Enable scratch/stain detection (default True)
        crease_sensitivity (float): 0.5=less sensitive, 1.0=normal, 2.0=more sensitive

    Returns:
        PIL.Image: Mask image (white=damage, black=good)
    """
    # Load image
    img_bgr = cv2.imread(input_path)
    if img_bgr is None:
        raise ValueError(f'Could not load image: {input_path}')

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    combined_mask = np.zeros((h, w), dtype=np.uint8)

    print(f'  [Mask] Analysing image ({w}x{h})...')

    # Detect creases/folds
    if detect_creases:
        crease_mask = _detect_creases(gray)
        combined_mask = cv2.bitwise_or(combined_mask, crease_mask)
        crease_px = np.sum(crease_mask > 0)
        print(f'  [Mask] Crease detection: {crease_px} pixels flagged')

    # Detect scratches and stains
    if detect_scratches:
        scratch_mask = _detect_scratches_and_stains(gray)
        combined_mask = cv2.bitwise_or(combined_mask, scratch_mask)
        scratch_px = np.sum(scratch_mask > 0)
        print(f'  [Mask] Scratch/stain detection: {scratch_px} pixels flagged')

    # Apply sensitivity scaling
    if crease_sensitivity != 1.0:
        iters = max(1, int(crease_sensitivity * 2))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        if crease_sensitivity > 1.0:
            combined_mask = cv2.dilate(combined_mask, kernel, iterations=iters)
        else:
            combined_mask = cv2.erode(combined_mask, kernel, iterations=iters)

    # Protect face regions — remove face areas from mask
    face_region = _get_face_regions(gray)
    if np.any(face_region > 0):
        combined_mask = cv2.bitwise_and(combined_mask, cv2.bitwise_not(face_region))

    # Final cleanup — remove tiny isolated noise pixels
    cleanup_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, cleanup_kernel)

    total_damage = np.sum(combined_mask > 0)
    total_pixels = h * w
    pct = (total_damage / total_pixels) * 100
    print(f'  [Mask] Total damage area: {total_damage} px ({pct:.1f}% of image)')

    # Warn if mask seems too large (likely false positives)
    if pct > 40:
        print(f'  [Mask] WARNING: Mask covers {pct:.1f}% of image — consider lowering crease_sensitivity')

    mask_pil = Image.fromarray(combined_mask)

    if save_mask_path:
        mask_pil.save(save_mask_path)
        print(f'  [Mask] Mask preview saved to {save_mask_path}')

    return mask_pil


# ─────────────────────────────────────────────
# MAIN RESTORE FUNCTION
# ─────────────────────────────────────────────
def restore_photo(input_path, output_path,
                  mask_path=None,
                  detect_creases=True,
                  detect_scratches=True,
                  crease_sensitivity=1.0,
                  save_mask_preview=None):
    """
    Restore a damaged photo using LaMa AI inpainting.
    Automatically detects fold lines, creases and scratches.
    Protects faces from being masked.

    Parameters:
        input_path          (str):   Path to damaged input photo (JPEG or PNG)
        output_path         (str):   Path to save the restored photo (PNG)
        mask_path           (str):   Optional — provide your own mask instead of auto-detection
        detect_creases      (bool):  Auto-detect fold/crease lines (default True)
        detect_scratches    (bool):  Auto-detect scratches and stains (default True)
        crease_sensitivity  (float): 0.5=less, 1.0=normal, 2.0=more sensitive
        save_mask_preview   (str):   Optional — save mask to this path for inspection

    Returns:
        str: output_path — path to the restored photo
    """
    print(f'\n[Module 1] Restoring: {input_path}')

    # Load original image
    image = Image.open(input_path).convert('RGB')
    original_size = image.size
    print(f'  [Load] Image size: {original_size[0]}x{original_size[1]}')

    # Get mask — manual or auto
    if mask_path:
        print(f'  [Mask] Using manual mask: {mask_path}')
        mask = Image.open(mask_path).convert('L')
    else:
        print(f'  [Mask] Running auto damage detection...')
        mask = generate_damage_mask(
            input_path,
            save_mask_path=save_mask_preview,
            detect_creases=detect_creases,
            detect_scratches=detect_scratches,
            crease_sensitivity=crease_sensitivity
        )

    # Resize both to 512x512 for LaMa
    image_resized = image.resize((512, 512))
    mask_resized  = mask.resize((512, 512))

    # Run LaMa inpainting on GPU
    print(f'  [LaMa] Running inpainting on GPU...')
    result = lama(image_resized, mask_resized)

    # Resize back to original dimensions
    result = result.resize(original_size)
    result.save(output_path)

    print(f'  [Done] Restored photo saved to {output_path}\n')
    return output_path


# ─────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────
if __name__ == '__main__':
    import os

    test_input  = os.path.join(os.path.dirname(__file__), 'test_input.png')
    test_output = os.path.join(os.path.dirname(__file__), 'output_restored.png')
    mask_preview = os.path.join(os.path.dirname(__file__), 'mask_preview.png')

    if not os.path.exists(test_input):
        print('ERROR: test_input.png not found. Please add a test photo.')
    else:
        restore_photo(
            input_path         = test_input,
            output_path        = test_output,
            detect_creases     = True,
            detect_scratches   = True,
            crease_sensitivity = 1.0,
            save_mask_preview  = mask_preview
        )
        print('Module 1 v2.0 is working perfectly!')
        print(f'Check mask_preview.png to see what was detected and masked.')
