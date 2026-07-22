"""
Create 2 additional test images covering:
  09 - Color portrait with heavy stains + torn corners
  10 - Multi-damage: creases + scratches + faded/missing regions
"""
import cv2
import numpy as np
from PIL import Image, ImageDraw
import os

OUT = "/home/ubuntu/Projects/PixelRevive/test_images"
os.makedirs(OUT, exist_ok=True)
np.random.seed(42)

# ── 09: Color portrait with heavy stains and torn corners ──────────────────────
portrait = np.zeros((512, 512, 3), dtype=np.uint8)
portrait[:] = [60, 55, 70]  # dark background

# Skin face oval
cv2.ellipse(portrait, (256, 256), (120, 150), 0, 0, 360, (160, 190, 220), -1)
# Hair
cv2.ellipse(portrait, (256, 160), (135, 110), 0, 180, 360, (35, 25, 15), -1)
# Eyes
cv2.ellipse(portrait, (210, 210), (22, 14), 0, 0, 360, (45, 35, 25), -1)
cv2.ellipse(portrait, (302, 210), (22, 14), 0, 0, 360, (45, 35, 25), -1)
cv2.circle(portrait, (210, 210), 8, (10, 8, 8), -1)
cv2.circle(portrait, (302, 210), 8, (10, 8, 8), -1)
# Nose
cv2.ellipse(portrait, (256, 268), (13, 16), 0, 0, 360, (130, 155, 185), -1)
# Mouth
pts = np.array([[225, 315], [256, 325], [287, 315]], np.int32)
cv2.polylines(portrait, [pts], False, (90, 70, 110), 2)
# Shoulders/collar
cv2.ellipse(portrait, (256, 450), (180, 80), 0, 0, 180, (50, 45, 65), -1)

# Heavy stains (dark circular blotches)
for _ in range(8):
    cx = np.random.randint(40, 470)
    cy = np.random.randint(40, 470)
    r  = np.random.randint(15, 50)
    alpha = np.random.uniform(0.3, 0.7)
    overlay = portrait.copy()
    cv2.circle(overlay, (cx, cy), r, (20, 15, 10), -1)
    portrait = cv2.addWeighted(overlay, alpha, portrait, 1 - alpha, 0)

# Yellowish water-damage stain
cv2.ellipse(portrait, (380, 100), (70, 50), 30, 0, 360, (120, 130, 60), -1)
portrait = cv2.addWeighted(portrait, 0.55, np.full_like(portrait, (120, 130, 60)), 0.45, 0)

# Torn corners (fill with black)
torn_pts_tl = np.array([[0, 0], [80, 0], [0, 80]], np.int32)
torn_pts_br = np.array([[512, 512], [512, 420], [420, 512]], np.int32)
cv2.fillPoly(portrait, [torn_pts_tl], (0, 0, 0))
cv2.fillPoly(portrait, [torn_pts_br], (0, 0, 0))

cv2.imwrite(f"{OUT}/09_color_portrait_stained_torn.jpg", portrait, [cv2.IMWRITE_JPEG_QUALITY, 90])
print("Created 09_color_portrait_stained_torn.jpg")

# ── 10: Multi-damage BW — creases + scratches + faded missing regions ──────────
multi = np.ones((512, 640), dtype=np.uint8) * 190

# Add some subject matter: building silhouette
cv2.rectangle(multi, (100, 200), (540, 500), 160, -1)      # building body
cv2.rectangle(multi, (200, 100), (440, 200), 155, -1)      # upper section
for x in range(130, 520, 50):                               # windows
    cv2.rectangle(multi, (x, 230), (x+30, 270), 100, -1)
    cv2.rectangle(multi, (x, 310), (x+30, 350), 100, -1)
    cv2.rectangle(multi, (x, 390), (x+30, 430), 100, -1)
# Sky
multi[:200, :] = 220
# Ground
multi[500:, :] = 140

# Deep diagonal crease (bright + shadow pair)
cv2.line(multi, (0, 150), (640, 380), 240, 4)
cv2.line(multi, (2, 152), (642, 382), 130, 2)
# Second crease
cv2.line(multi, (320, 0), (280, 512), 235, 3)
cv2.line(multi, (322, 0), (282, 512), 135, 2)

# Heavy white scratches
for _ in range(40):
    x1 = np.random.randint(0, 640)
    y1 = np.random.randint(0, 512)
    x2 = x1 + np.random.randint(-180, 180)
    y2 = y1 + np.random.randint(-80, 80)
    cv2.line(multi, (x1, y1), (x2, y2), 255, thickness=np.random.randint(1, 3))

# Dark scratches
for _ in range(25):
    x1 = np.random.randint(0, 640)
    y1 = np.random.randint(0, 512)
    x2 = x1 + np.random.randint(-100, 100)
    y2 = y1 + np.random.randint(-60, 60)
    cv2.line(multi, (x1, y1), (x2, y2), 20, thickness=1)

# Missing/faded region (overexposed patch)
multi[50:150, 400:550] = 245
multi[300:400, 20:120] = 245

# Salt-and-pepper noise
noise_mask = np.random.random(multi.shape)
multi[noise_mask < 0.03] = 255
multi[noise_mask > 0.97] = 0

cv2.imwrite(f"{OUT}/10_multidamage_bw.jpg", multi, [cv2.IMWRITE_JPEG_QUALITY, 85])
print("Created 10_multidamage_bw.jpg")

print(f"\nAll extra test images created in {OUT}/")
print("Images:", sorted(os.listdir(OUT)))
