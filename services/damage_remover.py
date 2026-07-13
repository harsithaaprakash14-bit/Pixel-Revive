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
# CREASE / FOLD LINE DETECTION HELPERS
# ─────────────────────────────────────────────
def _get_oriented_kernel(angle_rad, length):
    size = int(length) * 2 + 1
    if size < 3:
        size = 3
    kernel = np.zeros((size, size), dtype=np.uint8)
    cx, cy = size // 2, size // 2
    dx = int(length * np.cos(angle_rad))
    dy = int(length * np.sin(angle_rad))
    cv2.line(kernel, (cx - dx, cy - dy), (cx + dx, cy + dy), 1, thickness=1)
    return kernel

def _merge_hough_segments(segments, max_gap):
    if not segments:
        return []
    
    merged_segments = []
    used = set()
    
    angle_thresh = 10.0
    perp_thresh = 8.0
    gap_thresh = max_gap * 1.5
    
    for i in range(len(segments)):
        if i in used:
            continue
        
        curr = segments[i]
        j = 0
        while j < len(segments):
            if j == i or j in used:
                j += 1
                continue
            
            other = segments[j]
            if _can_merge_segments(curr, other, angle_thresh, gap_thresh, perp_thresh):
                curr = _merge_two_segments(curr, other)
                used.add(j)
                j = 0
                continue
            j += 1
        
        merged_segments.append(curr)
        used.add(i)
        
    return merged_segments

def _can_merge_segments(S1, S2, angle_thresh, gap_thresh, perp_thresh):
    p1, p2 = S1
    p3, p4 = S2
    
    v1 = np.array(p2) - np.array(p1)
    v2 = np.array(p4) - np.array(p3)
    L1 = np.linalg.norm(v1)
    L2 = np.linalg.norm(v2)
    if L1 < 1e-5 or L2 < 1e-5:
        return False
    
    u1 = v1 / L1
    u2 = v2 / L2
    
    cos_theta = abs(np.dot(u1, u2))
    cos_theta = np.clip(cos_theta, 0.0, 1.0)
    angle_diff = np.arccos(cos_theta) * 180.0 / np.pi
    if angle_diff > angle_thresh:
        return False
        
    n = np.array([-u1[1], u1[0]])
    d3 = abs(np.dot(np.array(p3) - np.array(p1), n))
    d4 = abs(np.dot(np.array(p4) - np.array(p1), n))
    if d3 > perp_thresh or d4 > perp_thresh:
        return False
        
    pts = [p1, p2, p3, p4]
    dists = []
    for i in range(2):
        for j in range(2, 4):
            dists.append(np.linalg.norm(np.array(pts[i]) - np.array(pts[j])))
    min_gap = min(dists)
    if min_gap > gap_thresh:
        return False
        
    return True

def _merge_two_segments(S1, S2):
    p1, p2 = S1
    p3, p4 = S2
    pts = [p1, p2, p3, p4]
    max_d = -1
    best_pair = (p1, p2)
    for i in range(4):
        for j in range(i+1, 4):
            d = np.linalg.norm(np.array(pts[i]) - np.array(pts[j]))
            if d > max_d:
                max_d = d
                best_pair = (pts[i], pts[j])
    return best_pair

# ─────────────────────────────────────────────
# CREASE / FOLD LINE DETECTION
# ─────────────────────────────────────────────
def _detect_creases(gray_img):
    """
    Detect paper fold/crease lines using a multi-scale Hessian-based ridge filter
    (Frangi filter), adaptive dual-percentile thresholding gated by a morphological
    top-hat dark-stripe detector, connected component structural filtering,
    and Hough Line detection with adaptive segment gap bridging.

    Enhancements:
      - Black-Hat Wide Fold Layer: captures wider dark structures
      - Local Darkness Ridge Map: identifies strong local depressions
      - Extended Sigmas: added sigma=5.0 for wide creases
      - Fold Severity Classification: Light, Medium, Deep folds
      - Directional Dilation: perpendicular-oriented custom line kernels
      - Intelligent Fold Segment Merging: groups nearby Hough lines by angle/proximity
      - False Positive Suppression Mask: edge-density/entropy guard
      - Detail Preservation Guard: scales back dilation in high-detail areas
    """
    from skimage.filters import frangi

    h, w = gray_img.shape

    # ── 1. Resize large images for faster, scale-robust detection ──────────────
    max_dim = 800
    scale = 1.0
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        gray_resized = cv2.resize(
            gray_img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
        )
    else:
        gray_resized = gray_img.copy()

    hr, wr = gray_resized.shape

    # ── 2. Local contrast enhancement via CLAHE ────────────────────────────────
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_enhanced = clahe.apply(gray_resized)

    # ── 3. Black-Hat Wide Fold Detection ─────────────────────────────────────
    # Threshold raised 12→20 to avoid false positives on normal shadow gradients
    # in skin, clothing and hair.
    bh_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    blackhat = cv2.morphologyEx(gray_enhanced, cv2.MORPH_BLACKHAT, bh_kernel)
    _, blackhat_bin = cv2.threshold(blackhat, 20, 255, cv2.THRESH_BINARY)

    # ── 4. Local Darkness Ridge Detection ────────────────────────────────────
    # Threshold raised 15→25: shallow shadow gradients (hair, eyebrows, clothing
    # folds) no longer trigger as crease candidates.
    local_mean = cv2.GaussianBlur(gray_enhanced, (25, 25), 0)
    local_darkness = cv2.subtract(local_mean, gray_enhanced)
    _, local_darkness_bin = cv2.threshold(local_darkness, 25, 255, cv2.THRESH_BINARY)

    # ── 5. Morphological top-hat dark-stripe detector ─────────────────────────
    dark_inv = cv2.bitwise_not(gray_enhanced)
    kh = max(9,  int(hr * 0.025))
    kv = max(9,  int(wr * 0.025))
    kd = max(7,  int(min(hr, wr) * 0.018))

    tophat_h = cv2.morphologyEx(
        dark_inv,
        cv2.MORPH_TOPHAT,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, kh))
    )
    tophat_v = cv2.morphologyEx(
        dark_inv,
        cv2.MORPH_TOPHAT,
        cv2.getStructuringElement(cv2.MORPH_RECT, (kv, 1))
    )
    tophat_d = cv2.morphologyEx(
        dark_inv,
        cv2.MORPH_TOPHAT,
        cv2.getStructuringElement(cv2.MORPH_RECT, (kd, kd))
    )
    tophat = np.maximum(np.maximum(tophat_h, tophat_v), tophat_d)

    tophat_f = tophat.astype(np.float32)
    # Require a minimum absolute top-hat value (e.g. 40.0) to prevent flagging minor noise on healthy images.
    tophat_thresh = max(40.0, np.percentile(tophat_f[tophat_f > 0], 90)) if np.any(tophat_f > 0) else 40.0
    tophat_binary = (tophat_f >= tophat_thresh).astype(np.uint8) * 255

    # ── 6. Multi-scale Frangi ridge filter ────────────────────────────────────
    # Sigmas match thin (1.0), medium (2.0), thick (3.5), and very thick (5.0) fold structures
    sigmas = [1.0, 2.0, 3.5, 5.0]
    frangi_dark  = frangi(gray_enhanced, sigmas=sigmas, black_ridges=True)
    frangi_light = frangi(gray_enhanced, sigmas=sigmas, black_ridges=False)
    frangi_comb  = np.maximum(frangi_dark, frangi_light)

    # ── 7. Dual-threshold Frangi binarisation ─────────────────────────────────
    # Percentiles raised 94→96 (hi) and 90→93 (lo) to ensure only the strongest
    # ridge responses are accepted, suppressing weak responses from natural
    # textures (fine hair, fabric weave, leaf veins).
    thresh_hi = max(0.08, np.percentile(frangi_comb, 96.0))
    binary_hi  = (frangi_comb > thresh_hi).astype(np.uint8) * 255

    # Gated low threshold raised absolute floor from 0.015 to 0.04
    thresh_lo = max(0.04, np.percentile(frangi_comb, 93.0))
    binary_lo_candidate = (frangi_comb > thresh_lo).astype(np.uint8) * 255
    dark_clues = cv2.bitwise_or(tophat_binary, cv2.bitwise_or(blackhat_bin, local_darkness_bin))
    binary_lo_gated = cv2.bitwise_and(binary_lo_candidate, dark_clues)

    binary_frangi = cv2.bitwise_or(binary_hi, binary_lo_gated)

    # ── 8. False Positive Suppression ────────────────────────────────────────
    # Threshold lowered 45→25: a lower edge-density threshold means more
    # natural-texture regions (hair, feathers, fabric, foliage) are correctly
    # excluded from the crease mask, reducing false crease detections.
    edges = cv2.Canny(gray_resized, 30, 100)
    edge_density = cv2.blur(edges.astype(np.float32), (15, 15))
    texture_mask = (edge_density > 25.0).astype(np.uint8) * 255

    binary_frangi = cv2.bitwise_and(binary_frangi, cv2.bitwise_not(texture_mask))

    # ── 9. Detail Preservation Guard (NEW) ────────────────────────────────────
    laplacian = cv2.Laplacian(gray_resized, cv2.CV_32F)
    laplacian_sq = cv2.multiply(laplacian, laplacian)
    local_detail_var = cv2.blur(laplacian_sq, (9, 9))
    detail_mask = (local_detail_var > 100.0).astype(np.uint8) * 255

    # ── 10. Connected component structural filtering ───────────────────────────
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_frangi)
    cc_mask = np.zeros_like(binary_frangi)
    
    severity_counts = {"light": 0, "medium": 0, "deep": 0}
    crease_accum = np.zeros_like(binary_frangi)

    for i in range(1, num_labels):
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        diag = np.sqrt(bw ** 2 + bh ** 2)
        aspect_ratio = max(bw, bh) / max(1, min(bw, bh))
        if diag > 25 or (diag > 10 and aspect_ratio > 1.8):
            cc_mask[labels == i] = 255
            comp_mask = (labels == i).astype(np.uint8) * 255
            
            # Orientation via Moments
            m = cv2.moments(comp_mask)
            if abs(m['mu20'] - m['mu02']) > 1e-5:
                theta = 0.5 * np.arctan2(2 * m['mu11'], m['mu20'] - m['mu02'])
            else:
                theta = 0.0
            
            perp_theta = theta + np.pi/2
            
            # Severity classification for components
            comp_frangi = cv2.mean(frangi_comb, mask=comp_mask)[0]
            comp_dark = cv2.mean(local_darkness, mask=comp_mask)[0]
            comp_detail = cv2.mean(detail_mask, mask=comp_mask)[0]
            
            severity = "light"
            if comp_dark > 25.0 or comp_frangi > 0.08 or (diag > min(hr, wr) * 0.20 and comp_dark > 15.0):
                severity = "deep"
            elif comp_dark > 12.0 or comp_frangi > 0.04:
                severity = "medium"
                
            severity_counts[severity] += 1
            
            # Set directional dilation width parameters
            if severity == "deep":
                d_perp = 4.0
                d_parallel = 1.5
            elif severity == "medium":
                d_perp = 2.5
                d_parallel = 0.8
            else:
                d_perp = 1.2
                d_parallel = 0.4
                
            if comp_detail > 50.0:
                d_perp = max(0.8, d_perp * 0.5)
                d_parallel = max(0.4, d_parallel * 0.5)
                
            # Perform anisotropic directional dilation
            dilated_comp = cv2.dilate(comp_mask, _get_oriented_kernel(perp_theta, d_perp))
            dilated_comp = cv2.dilate(dilated_comp, _get_oriented_kernel(theta, d_parallel))
            
            crease_accum = cv2.bitwise_or(crease_accum, dilated_comp)

    # ── 11. Hough Line Detection with adaptive gap bridging ────────────────────
    min_line_len = int(min(hr, wr) * 0.10)
    max_line_gap = max(20, int(min(hr, wr) * 0.04))

    combined_candidates = cv2.bitwise_or(cc_mask, cv2.bitwise_and(dark_clues, cv2.bitwise_not(texture_mask)))
    lines = cv2.HoughLinesP(
        combined_candidates,
        rho=1, theta=np.pi / 180,
        threshold=30,
        minLineLength=min_line_len,
        maxLineGap=max_line_gap,
    )

    raw_segments = []
    if lines is not None:
        raw_segments = [((x1, y1), (x2, y2)) for x1, y1, x2, y2 in lines.reshape(-1, 4)]

    # Intelligent segment merging (NEW)
    merged_segments = _merge_hough_segments(raw_segments, max_line_gap)
    print(f"  [Crease] Hough lines: {len(raw_segments)} segments -> {len(merged_segments)} merged folds")

    for p1, p2 in merged_segments:
        theta = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
        perp_theta = theta + np.pi/2
        
        line_mask = np.zeros_like(gray_resized)
        cv2.line(line_mask, p1, p2, 255, thickness=1)
        
        seg_frangi = cv2.mean(frangi_comb, mask=line_mask)[0]
        seg_dark = cv2.mean(local_darkness, mask=line_mask)[0]
        seg_detail = cv2.mean(detail_mask, mask=line_mask)[0]
        length = np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
        
        severity = "light"
        if seg_dark > 25.0 or seg_frangi > 0.08 or (length > min(hr, wr) * 0.20 and seg_dark > 15.0):
            severity = "deep"
        elif seg_dark > 12.0 or seg_frangi > 0.04:
            severity = "medium"
            
        severity_counts[severity] += 1
        
        if severity == "deep":
            d_perp = 5.0
            d_parallel = 2.0
        elif severity == "medium":
            d_perp = 3.0
            d_parallel = 1.0
        else:
            d_perp = 1.5
            d_parallel = 0.5
            
        if seg_detail > 50.0:
            d_perp = max(1.0, d_perp * 0.5)
            d_parallel = max(0.5, d_parallel * 0.5)
            
        # Draw the anisotropic dilated polygon
        v1 = np.array(p2) - np.array(p1)
        L = np.linalg.norm(v1)
        if L > 1e-5:
            u = v1 / L
            v = np.array([-u[1], u[0]])
            
            pt1 = np.array(p1) - d_parallel * u + d_perp * v
            pt2 = np.array(p1) - d_parallel * u - d_perp * v
            pt3 = np.array(p2) + d_parallel * u - d_perp * v
            pt4 = np.array(p2) + d_parallel * u + d_perp * v
            
            poly = np.array([pt1, pt2, pt3, pt4], dtype=np.int32)
            cv2.fillPoly(crease_accum, [poly], 255)

    print(f"  [Crease] Severity counts: {severity_counts}")

    # ── 12. Resize mask back to original resolution ───────────────────────────
    crease_mask = cv2.resize(crease_accum, (w, h), interpolation=cv2.INTER_NEAREST)

    return crease_mask




# ─────────────────────────────────────────────
# SCRATCH & STAIN DETECTION
# ─────────────────────────────────────────────
def _detect_scratches_and_stains(gray_img):
    """
    Detect random scratches and stains using:
    1. Gaussian blur difference to find sharp local anomalies
    2. Morphological closing to merge nearby damage regions
    3. Bright hairline scratch Canny layer (catches fine bright surface marks)

    Threshold tuning (v2):
      - Unsharp-mask diff threshold raised 22→30: previously too sensitive,
        firing on normal hair/fabric edge transitions.
      - Min component area raised 6→25px: removes residual pepper noise.
      - Stain diff threshold raised 55→70: natural shading gradients no longer
        flagged as stains.
      - Added edge-density texture suppression (matching crease detector) so
        high-texture regions (hair, feathers, foliage) are excluded.
    """
    scratch_mask = np.zeros(gray_img.shape, dtype=np.uint8)

    # Build a texture-suppression mask first: exclude high-edge-density zones
    # (hair, feathers, grass, fabric) from ALL scratch detection.
    edges_sup = cv2.Canny(gray_img, 30, 100)
    edge_density_sup = cv2.blur(edges_sup.astype(np.float32), (15, 15))
    texture_suppress = (edge_density_sup > 25.0).astype(np.uint8) * 255

    # Unsharp mask difference — reveals fine scratches.
    # Threshold raised 22→30 to stop firing on normal texture edges.
    blur_light = cv2.GaussianBlur(gray_img, (3, 3), 0)
    blur_heavy = cv2.GaussianBlur(gray_img, (21, 21), 0)
    diff = cv2.absdiff(blur_light, blur_heavy)
    _, scratch_thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    # Suppress high-texture regions immediately after thresholding
    scratch_thresh = cv2.bitwise_and(scratch_thresh, cv2.bitwise_not(texture_suppress))

    # Remove tiny noise — min area raised 6→25px for a stronger noise filter.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(scratch_thresh)
    clean_scratch = np.zeros_like(scratch_thresh)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= 25:
            clean_scratch[labels == i] = 255
    scratch_thresh = clean_scratch

    # Stain detection — large uniform discoloured regions.
    # Threshold raised 55→70: natural shading gradients no longer trigger.
    blur_stain = cv2.GaussianBlur(gray_img, (31, 31), 0)
    diff_stain = cv2.absdiff(gray_img, blur_stain)
    _, stain_thresh = cv2.threshold(diff_stain, 70, 255, cv2.THRESH_BINARY)
    stain_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    stain_thresh = cv2.morphologyEx(stain_thresh, cv2.MORPH_CLOSE, stain_kernel)
    stain_thresh = cv2.bitwise_and(stain_thresh, cv2.bitwise_not(texture_suppress))

    # ── Bright hairline scratch layer ──────────────────────────────────────────
    # Catches fine, bright surface marks the difference method misses.
    # Only applied to the top-bright region so sky/walls are not erroneously hit.
    bright_region = (gray_img > 200).astype(np.uint8) * 255
    # Canny on original — fine edges in the bright area are cracks
    canny_fine = cv2.Canny(gray_img, 60, 140)
    bright_canny = cv2.bitwise_and(canny_fine, bright_region)
    bright_canny = cv2.bitwise_and(bright_canny, cv2.bitwise_not(texture_suppress))
    # Dilate 1px to make hairlines visible to LaMa
    bc_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    bright_canny = cv2.dilate(bright_canny, bc_kernel, iterations=1)

    scratch_mask = cv2.bitwise_or(cv2.bitwise_or(scratch_thresh, stain_thresh), bright_canny)
    return scratch_mask


# ─────────────────────────────────────────────
# DENSE BRIGHT CRACK NETWORK DETECTION
# ─────────────────────────────────────────────
def _detect_bright_crack_network(gray_img):
    """
    Detect dense white/bright surface crack networks (e.g., shattered-glass
    crack patterns on old photos).

    Strategy:
      1. Adaptive Otsu threshold on the bright top-30% of the image to
         isolate crack pixels from the background.
      2. Morphological thinning (skeleton) to reduce blobs to crack skeletons.
      3. Hough line detection to extract crack directions.
      4. Directional dilation (perpendicular to each crack line) to restore
         the full crack width for LaMa inpainting.
      5. Connected-component filtering: discard blob-shaped regions (uniform
         bright patches like walls/sky) and keep only high-aspect-ratio
         crack-shaped components.

    Returns:
        np.ndarray: uint8 (H, W) binary mask — 255 = crack, 0 = clean.
    """
    h, w = gray_img.shape

    # ── 1. CLAHE to boost local contrast of crack edges ───────────────────────
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray_img)

    # ── 2. Adaptive threshold to isolate bright crack regions ─────────────────
    # Global Otsu on the full image to find the crack brightness level.
    # Otsu shift raised 1.05→1.15: requires pixels to be clearly brighter than
    # the scene average before being classified as cracks — suppresses false
    # positives on bright fabric, paper grain and overexposed skin.
    otsu_thresh, _ = cv2.threshold(
        enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    crack_thresh = int(min(255, otsu_thresh * 1.15))
    _, bright_bin = cv2.threshold(enhanced, crack_thresh, 255, cv2.THRESH_BINARY)

    # Also run per-tile adaptive threshold to catch locally varying cracks.
    # C raised -12→-20: more conservative — only accepts strongly above-local-
    # mean pixels, preventing foliage/texture patterns from being detected.
    adaptive_bin = cv2.adaptiveThreshold(
        enhanced, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=21,
        C=-20
    )
    # Union global and adaptive thresholds
    combined_bright = cv2.bitwise_or(bright_bin, adaptive_bin)

    # ── 3. Connected-component filtering: remove blob-shaped bright areas ──────
    #   Keep only high-aspect-ratio components (cracks are thin/elongated).
    #   Aspect minimum raised 1.5→2.0: blobs with low elongation (e.g. bright
    #   skin patches, sky areas, paper texture) are now excluded.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        combined_bright, connectivity=8
    )
    crack_bin = np.zeros_like(combined_bright)
    for i in range(1, num_labels):
        bw   = stats[i, cv2.CC_STAT_WIDTH]
        bh   = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        diag = np.sqrt(bw ** 2 + bh ** 2)
        # Aspect ratio check: minimum raised 1.5→2.0 — cracks must be more
        # elongated to be accepted; reduces blob false positives.
        aspect = max(bw, bh) / max(1, min(bw, bh))
        # Fill-ratio check: cracks have low fill (thin line in bounding box)
        fill_ratio = area / max(1, bw * bh)
        # Accept if elongated OR if it is a small spot in a large crack context
        if (aspect >= 2.0 and fill_ratio < 0.65) or (diag > min(h, w) * 0.05 and fill_ratio < 0.45):
            crack_bin[labels == i] = 255

    # ── 4. Morphological skeleton — thin cracks to single-pixel paths ──────────
    # Use repeated erosion + hit-or-miss to approximate Zhang-Suen thinning
    # (cv2 does not have a built-in skeleton, so we use iterative erosion)
    skeleton = np.zeros_like(crack_bin)
    temp    = crack_bin.copy()
    skel_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    for _ in range(12):  # max 12 thinning iterations
        eroded   = cv2.erode(temp, skel_kernel)
        opened   = cv2.dilate(eroded, skel_kernel)
        diff     = cv2.subtract(temp, opened)
        skeleton = cv2.bitwise_or(skeleton, diff)
        temp     = eroded
        if cv2.countNonZero(temp) == 0:
            break

    # ── 5. Hough line detection on skeleton for directional info ───────────────
    min_len = 12   # cracks can be short, do not ignore them
    max_gap = max(10, int(min(h, w) * 0.025))  # allow small gaps between fragments
    lines = cv2.HoughLinesP(
        skeleton, rho=1, theta=np.pi / 180,
        threshold=15,
        minLineLength=min_len,
        maxLineGap=max_gap
    )

    # ── 6. Directional dilation: restore crack width ───────────────────────────
    # Draw each line dilated perpendicular to its orientation
    crack_mask_out = skeleton.copy()  # include skeleton itself
    if lines is not None:
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            # Draw thick line (covers natural crack width ~3–7px)
            cv2.line(crack_mask_out, (x1, y1), (x2, y2), 255, thickness=4)

    # Also re-include the filtered crack blobs (fills interior crack areas)
    crack_mask_out = cv2.bitwise_or(crack_mask_out, crack_bin)

    # ── 7. Final morphological closing to bridge tiny gaps and boundary dilation ─
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    crack_mask_out = cv2.morphologyEx(crack_mask_out, cv2.MORPH_CLOSE, close_kernel)
    
    # Dilate 1px to cover crack boundaries completely (critical for LaMa inpainting)
    dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    crack_mask_out = cv2.dilate(crack_mask_out, dilation_kernel, iterations=1)

    crack_px = int(np.sum(crack_mask_out > 0))
    pct = round(crack_px / crack_mask_out.size * 100, 2)
    print(f"  [CrackNet] Bright crack network detected: {crack_px} px ({pct:.2f}%)")
    return crack_mask_out


# ─── Main damage mask generator ───────────────────────────────────────────────

def generate_damage_mask(img_bgr: np.ndarray):
    """
    Auto-generate a conservative binary damage mask from a photograph.
    Combines creases, scratches and stains, protects detected face regions,
    and caps the coverage at 10%.

    Returns:
        tuple[np.ndarray, dict]:
            - mask: uint8 (H, W) — 255 = damaged region, 0 = clean region.
            - meta: dict with restoration diagnostics for the Score Card:
                {
                  "crease_px":        int,   # crease pixels before cap
                  "scratch_px":       int,   # scratch/stain pixels before cap
                  "face_regions":     int,   # number of detected faces protected
                  "total_damage_pct": float, # combined damage as % of image
                  "mask_coverage_pct":float, # final mask coverage after cap
                }
    """
    gray = (cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            if len(img_bgr.shape) == 3 else img_bgr.copy())

    print(f"  [Mask] Analysing image ({img_bgr.shape[1]}x{img_bgr.shape[0]})...")

    # ── Detect creases ────────────────────────────────────────────────────────
    crease_mask = _detect_creases(gray)
    crease_px   = int(np.sum(crease_mask > 0))
    print(f"  [Mask] Crease detection: {crease_px} pixels flagged")

    # ── Detect scratches and stains ───────────────────────────────────────────
    scratch_mask = _detect_scratches_and_stains(gray)
    scratch_px   = int(np.sum(scratch_mask > 0))
    print(f"  [Mask] Scratch/stain detection: {scratch_px} pixels flagged")

    # ── Detect dense bright crack networks (NEW) ──────────────────────────────
    crack_network_mask = _detect_bright_crack_network(gray)
    crack_network_px   = int(np.sum(crack_network_mask > 0))

    # ── Auto-detect heavy crack damage mode ───────────────────────────────────
    # A crack network covering >5% of pixels means this is a heavily damaged
    # photo (like the shattered-glass crack pattern). In that case we must use
    # a much higher coverage cap so the mask is not eroded away before LaMa.
    total_pixels = gray.size
    crack_coverage_pct = crack_network_px / total_pixels
    is_crack_mode = crack_coverage_pct > 0.05
    if is_crack_mode:
        print(f"  [Mask] CRACK MODE active — dense crack network ({crack_coverage_pct*100:.1f}% coverage). Cap raised to 40%.")
    else:
        print(f"  [Mask] Normal mode — crack network: {crack_coverage_pct*100:.1f}% (below 5% threshold).")

    # ── Combine all detections ────────────────────────────────────────────────
    combined = cv2.bitwise_or(crease_mask, scratch_mask)
    combined = cv2.bitwise_or(combined, crack_network_mask)
    total_damage_pct = round(float(np.sum(combined > 0) / combined.size) * 100, 2)

    # ── Protect face regions ──────────────────────────────────────────────────
    face_region  = _get_face_regions(gray)
    face_regions = 0
    if np.any(face_region > 0):
        # Count distinct face blobs that were protected
        n_lab, _ = cv2.connectedComponents(face_region)
        face_regions = max(0, n_lab - 1)   # subtract background label
        combined = cv2.bitwise_and(combined, cv2.bitwise_not(face_region))

    # ── Final cleanup — remove tiny isolated noise pixels (line-preserving) ───
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(combined)
    final_mask = np.zeros_like(combined)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= 5:
            final_mask[labels == i] = 255

    # ── Coverage cap ──────────────────────────────────────────────────────────
    # Normal photos: cap tightened 25%→20% to be more conservative — fewer
    # undamaged pixels will be sent to LaMa.
    # Crack-mode photos: cap at 45% (unchanged — heavy damage needs coverage).
    MAX_COV = 0.45 if is_crack_mode else 0.20
    coverage = (final_mask > 0).sum() / final_mask.size

    if coverage > MAX_COV:
        # Sort components by area (excluding background label 0) to keep largest damage features
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(final_mask)
        component_indices = sorted(range(1, num_labels), key=lambda idx: stats[idx, cv2.CC_STAT_AREA], reverse=True)
        
        capped_mask = np.zeros_like(final_mask)
        current_pixels = 0
        max_pixels = int(MAX_COV * final_mask.size)
        
        for idx in component_indices:
            comp_area = stats[idx, cv2.CC_STAT_AREA]
            if current_pixels + comp_area <= max_pixels:
                capped_mask[labels == idx] = 255
                current_pixels += comp_area
            else:
                continue
        
        dropped_pct = ((final_mask > 0).sum() - (capped_mask > 0).sum()) / final_mask.size * 100
        print(f"[LaMa] Mask capped at {MAX_COV*100}% by dropping minor components. Dropped {dropped_pct:.2f}% of image area.")
        final_mask = capped_mask

    # Always save debug mask for inspection
    cv2.imwrite("debug_mask.png", final_mask)
    print("[LaMa] Debug mask saved as debug_mask.png")

    mask_coverage_pct = round(float((final_mask > 0).sum() / final_mask.size) * 100, 2)

    meta = {
        "crease_px":         crease_px,
        "scratch_px":        scratch_px,
        "crack_network_px":  crack_network_px,
        "face_regions":      face_regions,
        "total_damage_pct":  total_damage_pct,
        "mask_coverage_pct": mask_coverage_pct,
        "crack_mode":        is_crack_mode,
    }

    return final_mask, meta



# ─── Public API ───────────────────────────────────────────────────────────────

def restore_image(input_path: str, output_path: str):
    """
    Detect and remove damage from a photograph using LaMa inpainting.

    Called by services/ai_connector.py as Step 1 of the pipeline.

    Parameters:
        input_path  (str): Path to the original uploaded image.
        output_path (str): Path where the restored image will be saved.

    Returns:
        tuple[str, dict]: (output_path, restoration_meta)
            restoration_meta keys: crease_px, scratch_px, face_regions,
                                   total_damage_pct, mask_coverage_pct.

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

    mask_np, meta = generate_damage_mask(img_bgr)
    final_pct = (mask_np > 0).sum() / mask_np.size * 100
    print(f"[LaMa] Final damage mask: {final_pct:.2f}% of pixels flagged")

    # ── Inpainting skip gate ───────────────────────────────────────────────────
    # If the damage mask covers less than 0.5% of the image, the photo is
    # effectively healthy. Skip the LaMa inpainting entirely and return the
    # original image unchanged to prevent unnecessary reconstruction.
    SKIP_THRESHOLD_PCT = 0.5
    if final_pct < SKIP_THRESHOLD_PCT:
        import shutil
        print(f"[LaMa] Mask coverage {final_pct:.3f}% is below skip threshold "
              f"({SKIP_THRESHOLD_PCT}%). Image appears healthy — skipping inpainting.")
        shutil.copy(input_path, output_path)
        size_kb = os.path.getsize(output_path) / 1024
        print(f"[LaMa] Original image copied -> {output_path}  ({size_kb:.1f} KB)")
        return output_path, meta

    LAMA_SIZE = 512
    img_rgb   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(img_rgb).resize((LAMA_SIZE, LAMA_SIZE), Image.LANCZOS)
    pil_mask  = Image.fromarray(mask_np).resize((LAMA_SIZE, LAMA_SIZE), Image.NEAREST)

    result_pil  = _run_lama_inference(pil_image, pil_mask)
    
    # ─── Enhancement 5: Residual Fold Verification ─────────────────────────
    # Run fold detection on the restored 512x512 image
    restored_np = np.array(result_pil)
    restored_gray = cv2.cvtColor(restored_np, cv2.COLOR_RGB2GRAY)
    orig_mask_512 = np.array(pil_mask)
    
    restored_creases = _detect_creases(restored_gray)
    residual_mask = cv2.bitwise_and(restored_creases, orig_mask_512)
    
    residual_px = np.sum(residual_mask > 0)
    residual_pct = (residual_px / residual_mask.size) * 100
    
    if residual_pct > 0.05:
        print(f"[LaMa] Residual folds detected: {residual_px} px ({residual_pct:.2f}%). Running second pass...")
        pil_residual_mask = Image.fromarray(residual_mask)
        result_pil = _run_lama_inference(result_pil, pil_residual_mask)
    else:
        print(f"[LaMa] Residual folds check: clean ({residual_pct:.2f}% remaining). Skipping second pass.")

    result_full = result_pil.resize((original_w, original_h), Image.LANCZOS)

    result_full.save(output_path)
    size_kb = os.path.getsize(output_path) / 1024
    print(f"[LaMa] Restored image saved -> {output_path}  ({size_kb:.1f} KB)")
    return output_path, meta


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
