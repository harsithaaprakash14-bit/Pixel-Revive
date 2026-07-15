"""
Create comprehensive test images for PixelRevive pipeline testing.
Covers: clean color, clean BW, faded, scratched, damaged, portrait-like, landscape-like, sepia.
"""
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import os

OUT = "/home/ubuntu/Projects/PixelRevive/test_images"
os.makedirs(OUT, exist_ok=True)

# 1. Clean color photo (already color, should skip colorization)
img = np.zeros((400, 600, 3), dtype=np.uint8)
img[:200, :] = [45, 100, 190]     # sky blue top
img[200:, :] = [60, 140, 80]      # green ground
for i in range(5):
    cv2.circle(img, (100 + i*100, 130), 30 + i*10, (255, 220, 50), -1)  # sun/circles
cv2.rectangle(img, (200, 220), (400, 380), (150, 90, 40), -1)  # brown building
cv2.imwrite(f"{OUT}/01_clean_color.jpg", img)
print("Created 01_clean_color.jpg")

# 2. Grayscale portrait-like (needs colorization)
gray = np.zeros((400, 400), dtype=np.uint8)
# Face-like oval 
cv2.ellipse(gray, (200, 200), (100, 130), 0, 0, 360, 200, -1)
# Eyes
cv2.ellipse(gray, (165, 165), (18, 12), 0, 0, 360, 80, -1)
cv2.ellipse(gray, (235, 165), (18, 12), 0, 0, 360, 80, -1)
# Nose
cv2.ellipse(gray, (200, 210), (12, 15), 0, 0, 360, 160, -1)
# Mouth
cv2.ellipse(gray, (200, 250), (28, 10), 0, 0, 180, 120, 2)
# Background gradient
for i in range(400):
    gray[:, i] = np.maximum(gray[:, i], int(30 + i * 0.15))
cv2.imwrite(f"{OUT}/02_grayscale_portrait.jpg", gray)
print("Created 02_grayscale_portrait.jpg")

# 3. Black and white photo with heavy scratches and noise (damaged)
bw = np.ones((500, 700), dtype=np.uint8) * 180  # gray background
# Add structure
cv2.rectangle(bw, (100, 100), (600, 400), 220, -1)
cv2.ellipse(bw, (350, 200), (150, 80), 0, 0, 360, 200, -1)
# Add realistic scratches  
for _ in range(30):
    x1, y1 = np.random.randint(0, 700), np.random.randint(0, 500)
    x2, y2 = x1 + np.random.randint(-200, 200), y1 + np.random.randint(-100, 100)
    cv2.line(bw, (x1, y1), (x2, y2), 255, thickness=np.random.randint(1, 3))
# Add dark scratches too
for _ in range(20):
    x1, y1 = np.random.randint(0, 700), np.random.randint(0, 500)
    x2, y2 = x1 + np.random.randint(-150, 150), y1 + np.random.randint(-80, 80)
    cv2.line(bw, (x1, y1), (x2, y2), 10, thickness=1)
# Add noise
noise = np.random.randint(-20, 20, bw.shape, dtype=np.int16)
bw = np.clip(bw.astype(np.int16) + noise, 0, 255).astype(np.uint8)
cv2.imwrite(f"{OUT}/03_bw_scratched.jpg", bw)
print("Created 03_bw_scratched.jpg")

# 4. Old faded/sepia photo
pil = Image.new("RGB", (600, 450))
draw = ImageDraw.Draw(pil)
# Draw a landscape scene
draw.rectangle([(0, 0), (600, 220)], fill=(200, 220, 255))  # sky
draw.rectangle([(0, 220), (600, 450)], fill=(160, 180, 120))  # ground
draw.ellipse([(80, 80), (160, 160)], fill=(255, 240, 120))  # sun
# Trees
for x in [100, 200, 350, 480]:
    draw.rectangle([(x - 8, 240), (x + 8, 340)], fill=(100, 70, 40))
    draw.ellipse([(x-35, 200), (x+35, 270)], fill=(60, 120, 60))
# Apply sepia effect
pil = pil.convert("L")
pil = np.array(pil)
# Sepia toning
sepia = np.zeros((pil.shape[0], pil.shape[1], 3), dtype=np.uint8)
sepia[:, :, 0] = np.minimum(255, pil * 1.0)        # blue
sepia[:, :, 1] = np.minimum(255, pil * 0.9)        # green  
sepia[:, :, 2] = np.minimum(255, pil * 0.75)       # red -- sepia warm tint
# Apply fading (reduce contrast)
sepia = (sepia * 0.6 + 100).astype(np.uint8)
# Apply blur for aged look
sepia_bgr = cv2.cvtColor(sepia, cv2.COLOR_RGB2BGR)
sepia_bgr = cv2.GaussianBlur(sepia_bgr, (3, 3), 0)
cv2.imwrite(f"{OUT}/04_sepia_faded.jpg", sepia_bgr)
print("Created 04_sepia_faded.jpg")

# 5. Color landscape (should skip colorization, test damage detection)
landscape = np.zeros((450, 800, 3), dtype=np.uint8)
# Sky gradient
for y in range(200):
    landscape[y, :] = [max(0, 220 - y), max(100, 180 - y//2), 255]  # blue sky
# Mountains
pts = np.array([[0, 200], [150, 100], [300, 150], [450, 80], [600, 120], [750, 90], [800, 200]], np.int32)
cv2.fillPoly(landscape, [pts], (100, 110, 130))
# Ground
landscape[200:, :] = [40, 120, 60]  # green
# River
cv2.ellipse(landscape, (400, 350), (200, 50), 0, 0, 360, (180, 130, 50), -1)
# Add some artifical "stains"
for _ in range(4):
    cx, cy = np.random.randint(50, 750), np.random.randint(50, 400)
    r = np.random.randint(15, 40)
    overlay = landscape.copy()
    cv2.circle(overlay, (cx, cy), r, (80, 60, 40), -1)
    landscape = cv2.addWeighted(overlay, 0.4, landscape, 0.6, 0)
cv2.imwrite(f"{OUT}/05_color_landscape.jpg", landscape)
print("Created 05_color_landscape.jpg")

# 6. Very small image (edge case: 100x100)
tiny = np.random.randint(100, 200, (100, 100, 3), dtype=np.uint8)
cv2.imwrite(f"{OUT}/06_tiny_image.jpg", tiny)
print("Created 06_tiny_image.jpg")

# 7. Large color portrait simulation (faces for CodeFormer)
portrait = np.zeros((600, 500, 3), dtype=np.uint8)
# Background
portrait[:] = [40, 40, 50]
# Skin
cv2.ellipse(portrait, (250, 250), (130, 160), 0, 0, 360, (140, 170, 210), -1)
# Eyes
cv2.ellipse(portrait, (195, 200), (25, 16), 0, 0, 360, (50, 40, 30), -1)
cv2.ellipse(portrait, (305, 200), (25, 16), 0, 0, 360, (50, 40, 30), -1)
# Pupils
cv2.circle(portrait, (195, 200), 10, (20, 15, 15), -1)
cv2.circle(portrait, (305, 200), 10, (20, 15, 15), -1)
# Nose
cv2.ellipse(portrait, (250, 260), (15, 18), 0, 0, 360, (110, 140, 180), -1)
# Mouth
cv2.ellipse(portrait, (250, 310), (35, 12), 0, 0, 180, (90, 80, 130), 2)
# Hair
cv2.ellipse(portrait, (250, 180), (145, 90), 0, 180, 360, (40, 30, 20), -1)
cv2.imwrite(f"{OUT}/07_color_portrait.jpg", portrait)
print("Created 07_color_portrait.jpg")

# 8. Black and white with creases (simulate old paper folds)
crease = np.ones((500, 600), dtype=np.uint8) * 200
# Add paper texture
for i in range(0, 500, 20):
    crease[i:i+2, :] = 185
for j in range(0, 600, 25):
    crease[:, j:j+2] = 185
# Add fold creases (bright lines typical of scanned old photos)
cv2.line(crease, (0, 250), (600, 230), 240, 3)    # horizontal crease
cv2.line(crease, (300, 0), (320, 500), 238, 3)    # vertical crease
cv2.line(crease, (0, 0), (600, 500), 235, 2)       # diagonal crease
# Dark crease shadows
cv2.line(crease, (1, 251), (601, 231), 140, 2)
cv2.line(crease, (301, 0), (321, 500), 145, 2)
cv2.imwrite(f"{OUT}/08_bw_creased.jpg", crease)
print("Created 08_bw_creased.jpg")

print(f"\nAll test images created in {OUT}/")
print("Images:", os.listdir(OUT))
