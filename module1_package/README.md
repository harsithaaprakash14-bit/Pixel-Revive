# PixelRevive — Module 1 v2.0: Damage Removal

**Member A | Blaze Instance | LaMa + OpenCV Crease Detection**

---

## What's New in v2.0

| Feature | v1.0 | v2.0 |
|---|---|---|
| Mask generation | Manual only | **Auto-detection** |
| Fold/crease detection | ❌ | ✅ |
| Scratch detection | ❌ | ✅ |
| Stain detection | ❌ | ✅ |
| Face protection | ❌ | ✅ |
| Mask preview export | ❌ | ✅ |
| Sensitivity control | ❌ | ✅ |

---

## Quick Start

### Install
```
pip install -r requirements.txt
```

### Auto mode (recommended — no mask needed)
```python
from damage_removal import restore_photo

restore_photo(
    input_path  = 'old_photo.jpg',
    output_path = 'restored.png'
)
```

### Manual mode (provide your own mask)
```python
restore_photo(
    input_path  = 'old_photo.jpg',
    output_path = 'restored.png',
    mask_path   = 'my_mask.png'
)
```

### Inspect the auto-generated mask
```python
restore_photo(
    input_path        = 'old_photo.jpg',
    output_path       = 'restored.png',
    save_mask_preview = 'mask_preview.png'  # saves mask so you can check it
)
```

### Adjust crease sensitivity
```python
# Less sensitive (fewer false positives on textured backgrounds)
restore_photo('photo.jpg', 'out.png', crease_sensitivity=0.5)

# More sensitive (catches very faint creases)
restore_photo('photo.jpg', 'out.png', crease_sensitivity=2.0)
```

---

## How Crease Detection Works

1. **Bright crease highlight detection** — fold lines appear as bright streaks → adaptive threshold finds them
2. **Hough line detection** — finds long straight lines (only horizontal/vertical = real folds)
3. **Dark crease shadow detection** — some folds appear as dark shadows → threshold finds these too
4. **Dilation** — widens the mask slightly to cover full crease width
5. **Face protection** — face regions detected and excluded from mask automatically

---

## Member D Integration

```python
from damage_removal import restore_photo

# In your FastAPI pipeline:
clean = restore_photo(
    input_path  = uploaded_photo_path,
    output_path = '/tmp/clean.png',
    crease_sensitivity = 1.0,       # tune if needed
    save_mask_preview  = '/tmp/mask_preview.png'  # optional debug
)
# Pass clean to Module 2 (colorization)
```
