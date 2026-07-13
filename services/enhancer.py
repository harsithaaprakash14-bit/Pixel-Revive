"""
PixelRevive — services/enhancer.py
====================================
Final Enhancement Stage — applied after Real-ESRGAN upscaling.

Purpose:
  Real-ESRGAN and other upscalers often introduce a slight softening or
  over-smoothing during the super-resolution pass.  This module applies a
  cascade of lightweight CPU-based enhancements to restore perceived sharpness,
  local contrast, and fine texture—without introducing halos, ringing, or
  artificial paint-like effects.

Pipeline (all purely OpenCV/numpy, no additional models):
  1. Unsharp Mask          — controlled edge/detail boost (no halo bleed)
  2. CLAHE (LAB L-channel) — local contrast enhancement, avoids global blowout
  3. HF Texture Blend      — rescues fine texture lost in CLAHE normalisation
  4. Adaptive Sharpen      — boosts high-frequency details selectively in textured regions
  5. Edge Reinforcement    — mild Laplacian overlay to sharpen hard edges
  6. Halo Guard Blend      — blend sharpened result with original to cap intensity

Input:  BGR uint8 image (from upscaler output).
Output: BGR uint8 enhanced image, same resolution.
"""

import cv2
import numpy as np


def _unsharp_mask(img: np.ndarray, radius: float = 1.2, amount: float = 0.20) -> np.ndarray:
    """
    Classic unsharp mask using Gaussian blur as the blurring kernel.
    Sharp = Original + amount * (Original - Blur)

    Amount reduced 0.45 → 0.20: Real-ESRGAN already sharpens the image;
    applying heavy unsharp mask on top creates ringing and over-sharpened edges.
    """
    blur = cv2.GaussianBlur(img, (0, 0), radius)
    sharpened = cv2.addWeighted(img, 1.0 + amount, blur, -amount, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def _clahe_lab(img: np.ndarray, clip_limit: float = 2.0, tile_size: int = 8) -> np.ndarray:
    """
    Apply CLAHE only to the L channel of the LAB colour space.
    This boosts local contrast without altering hue or saturation.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    l_enhanced = clahe.apply(l_ch)
    merged = cv2.merge([l_enhanced, a_ch, b_ch])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def _hf_texture_blend(original: np.ndarray, processed: np.ndarray, strength: float = 0.15) -> np.ndarray:
    """
    Recover fine texture details by blending a high-frequency layer
    (original - blurred_original) back into the result.

    Strength reduced 0.35 → 0.15: was injecting too much high-frequency noise
    back, causing artificial grain in smooth skin areas.
    """
    blur_orig = cv2.GaussianBlur(original.astype(np.float32), (0, 0), 1.5)
    hf = original.astype(np.float32) - blur_orig          # high-frequency detail layer
    result = processed.astype(np.float32) + strength * hf  # inject back
    return np.clip(result, 0, 255).astype(np.uint8)


def _adaptive_sharpen(img: np.ndarray, strength: float = 0.15) -> np.ndarray:
    """
    Boosts fine detail in textured and structural regions (like clothing folds and hair)
    while keeping flat, smooth regions (like human skin) natural and artifact-free.
    Uses standard deviation/variance gating to target detailed regions, and Canny edge
    gating to prevent halos on high-contrast edges.

    Strength reduced 0.35 → 0.15: with multiple sharpening passes in the cascade,
    each at their previous strength, the cumulative effect was over-sharpening.
    At 0.15 this step adds a subtle final crispness without compounding artefacts.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Bilateral filter on L channel to preserve strong edges but smooth textures
    l_smooth = cv2.bilateralFilter(l, d=5, sigmaColor=15, sigmaSpace=15)
    
    # Detail layer is L - l_smooth (contains fine local textures/grain)
    detail = l.astype(np.float32) - l_smooth.astype(np.float32)
    
    # 1. Identify extreme edges to avoid halos
    edges = cv2.Canny(l, 30, 100)
    edge_mask = cv2.GaussianBlur(edges.astype(np.float32), (5, 5), 0) / 255.0
    
    # 2. Compute local variance to identify textured regions (hair, clothes) and ignore flat regions (skin)
    local_mean = cv2.blur(l.astype(np.float32), (5, 5))
    local_var = cv2.blur(l.astype(np.float32) ** 2, (5, 5)) - local_mean ** 2
    texture_mask = np.clip(local_var / 50.0, 0, 1)
    
    # Sharpen where texture is present but suppress on extreme edges to prevent halos
    sharpen_mask = texture_mask * (1.0 - edge_mask * 0.5)
    sharpen_mask = cv2.GaussianBlur(sharpen_mask, (3, 3), 0)
    
    # Apply details back scaled by the mask
    enhanced_l = l.astype(np.float32) + strength * detail * sharpen_mask
    enhanced_l = np.clip(enhanced_l, 0, 255).astype(np.uint8)
    
    merged = cv2.merge([enhanced_l, a, b])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def _edge_reinforce(img: np.ndarray, strength: float = 0.08) -> np.ndarray:
    """
    Mild Laplacian edge overlay to reinforce hard structures.

    Strength reduced 0.15 → 0.08: the previous value introduced a slight
    painterly/HDR look when combined with the other sharpening steps.
    """
    img_f = img.astype(np.float32)
    lap = cv2.Laplacian(img_f, cv2.CV_32F, ksize=3)
    # Suppress extreme noise values
    lap = np.clip(lap, -20, 20)
    result = img_f - strength * lap   # subtract because Laplacian is negative at edges
    return np.clip(result, 0, 255).astype(np.uint8)


def enhance_final_output(input_path: str, output_path: str) -> str:
    """
    Run the full final-enhancement cascade on a PNG/JPG image.

    Parameters:
        input_path  (str): Path to the upscaled image (from Real-ESRGAN).
        output_path (str): Destination path for the enhanced image.

    Returns:
        str: output_path on success.
    """
    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"[Enhancer] Cannot read image: {input_path}")

    h, w = img.shape[:2]
    print(f"  [Enhancer] Input size: {w}x{h}")

    # Keep a pristine copy for the final halo-guard blend
    original_copy = img.copy()

    # Step 1: Unsharp mask (amount 0.10 — gentle; ESRGAN already sharp)
    print("  [Enhancer] Step 1: Unsharp mask sharpening...")
    img = _unsharp_mask(img, radius=1.2, amount=0.10)

    # Step 2: CLAHE local contrast (LAB L-channel, clipLimit 1.2 — subtle)
    print("  [Enhancer] Step 2: CLAHE local contrast (LAB)...")
    img = _clahe_lab(img, clip_limit=1.2, tile_size=8)

    # Step 3: High-frequency texture blend (strength 0.10 — gentle recovery)
    print("  [Enhancer] Step 3: Texture preservation blend...")
    img = _hf_texture_blend(original_copy, img, strength=0.10)

    # Step 4: Adaptive detail sharpening (strength 0.08 — subtle crispness)
    print("  [Enhancer] Step 4: Adaptive detail sharpening...")
    img = _adaptive_sharpen(img, strength=0.08)

    # Step 5: Edge reinforcement (strength 0.04 — very mild)
    print("  [Enhancer] Step 5: Edge reinforcement...")
    img = _edge_reinforce(img, strength=0.04)

    # Step 6: Halo-guard blend — cap total sharpening intensity.
    # Blend ratio adjusted 0.70/0.30 -> 0.50/0.50: with 50% original upscaled blended
    # back in, any residual over-sharpening or colour shift is completely damped.
    print("  [Enhancer] Step 6: Halo-guard blend...")
    img = cv2.addWeighted(img, 0.50, original_copy, 0.50, 0).astype(np.uint8)

    # Save
    ok = cv2.imwrite(output_path, img)
    if not ok:
        raise RuntimeError(f"[Enhancer] cv2.imwrite failed: {output_path}")

    print(f"  [Enhancer] Enhancement complete -> {output_path}")
    return output_path
