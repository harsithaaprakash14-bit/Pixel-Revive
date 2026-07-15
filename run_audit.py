"""
Comprehensive PixelRevive Pipeline Audit
Runs all test images through the full 5-stage pipeline and reports results.
"""
import requests
import os
import sys
import json
import time

BASE_URL = "http://127.0.0.1:5000"

TEST_IMAGES = [
    ("test_images/01_clean_color.jpg",      "Clean color photo"),
    ("test_images/02_grayscale_portrait.jpg", "Grayscale portrait"),
    ("test_images/03_bw_scratched.jpg",     "B&W heavily scratched"),
    ("test_images/04_sepia_faded.jpg",      "Sepia faded landscape"),
    ("test_images/05_color_landscape.jpg",  "Color landscape with stains"),
    ("test_images/06_tiny_image.jpg",       "Tiny 100x100 edge case"),
    ("test_images/07_color_portrait.jpg",   "Color portrait simulation"),
    ("test_images/08_bw_creased.jpg",       "B&W with fold creases"),
    ("sample_photo.png",                    "Real-world sample photo"),
    ("audrey_grayscale.jpg",                "Audrey Grayscale portrait"),
]

results = []

def fmt_size(path):
    if os.path.exists(path):
        return f"{os.path.getsize(path)/1024:.0f}KB"
    return "N/A"

def test_image(img_path, description):
    print(f"\n{'='*60}")
    print(f"Testing: {description}")
    print(f"File: {img_path}  Size: {fmt_size(img_path)}")
    print('='*60)
    
    if not os.path.exists(img_path):
        print(f"  [SKIP] File not found: {img_path}")
        return {"image": img_path, "description": description, "status": "SKIP", "error": "File not found"}
    
    start = time.time()
    try:
        with open(img_path, 'rb') as f:
            ext = img_path.rsplit('.', 1)[-1].lower()
            mime = 'image/png' if ext == 'png' else 'image/jpeg'
            files = {'image': (os.path.basename(img_path), f, mime)}
            r = requests.post(f"{BASE_URL}/upload", files=files, timeout=600)
        
        elapsed = time.time() - start
        
        if r.status_code == 200:
            data = r.json()
            if data.get('status') == 'success':
                print(f"  [PASS] Status: SUCCESS")
                print(f"  Faces detected: {data.get('faces_detected', 0)}")
                print(f"  Duration: {data.get('duration', '?')}s")
                print(f"  Original: {data.get('original_image', '?')}")
                print(f"  Processed: {data.get('processed_image', '?')}")
                
                # Check output file exists and is valid
                proc = data.get('processed_image', '')
                out_path = f"outputs/{proc}"
                if os.path.exists(out_path):
                    out_size = os.path.getsize(out_path)
                    print(f"  Output file: {out_path} ({out_size/1024:.0f}KB)")
                    if out_size < 1000:
                        print(f"  [WARN] Output file is suspiciously small!")
                else:
                    print(f"  [WARN] Output file not found: {out_path}")
                
                return {
                    "image": img_path,
                    "description": description,
                    "status": "PASS",
                    "faces": data.get('faces_detected', 0),
                    "duration": data.get('duration', 0),
                    "original": data.get('original_image', ''),
                    "processed": data.get('processed_image', ''),
                    "elapsed": round(elapsed, 1),
                }
            else:
                print(f"  [FAIL] API returned: {data}")
                return {"image": img_path, "description": description, "status": "FAIL", "error": str(data)}
        else:
            print(f"  [FAIL] HTTP {r.status_code}: {r.text[:300]}")
            return {"image": img_path, "description": description, "status": "FAIL", "error": f"HTTP {r.status_code}: {r.text[:200]}"}
    
    except requests.exceptions.Timeout:
        elapsed = time.time() - start
        print(f"  [TIMEOUT] Request timed out after {elapsed:.0f}s")
        return {"image": img_path, "description": description, "status": "TIMEOUT", "elapsed": round(elapsed, 1)}
    except Exception as e:
        elapsed = time.time() - start
        print(f"  [ERROR] {type(e).__name__}: {e}")
        return {"image": img_path, "description": description, "status": "ERROR", "error": str(e), "elapsed": round(elapsed, 1)}


def test_all_endpoints():
    """Test non-pipeline endpoints"""
    print("\n\n=== Testing Non-Pipeline Endpoints ===")
    issues = []
    
    endpoints = [
        ("GET", "/", "Home page"),
        ("GET", "/history", "History page"),
        ("GET", "/admin", "Admin page"),
    ]
    
    for method, path, name in endpoints:
        try:
            r = requests.get(f"{BASE_URL}{path}", timeout=10)
            if r.status_code == 200:
                print(f"  [PASS] {name} ({path}) - {len(r.text)} bytes")
            else:
                print(f"  [FAIL] {name} ({path}) - HTTP {r.status_code}")
                issues.append(f"{name}: HTTP {r.status_code}")
        except Exception as e:
            print(f"  [ERROR] {name} ({path}) - {e}")
            issues.append(f"{name}: {e}")
    
    # Test invalid file upload
    try:
        r = requests.post(f"{BASE_URL}/upload", files={"image": ("test.txt", b"not an image", "text/plain")}, timeout=10)
        if r.status_code == 400:
            print(f"  [PASS] Invalid file type rejected correctly (HTTP 400)")
        else:
            print(f"  [FAIL] Invalid file type not rejected: HTTP {r.status_code}")
            issues.append("Invalid file type not rejected")
    except Exception as e:
        print(f"  [ERROR] Invalid file test: {e}")
    
    # Test empty upload
    try:
        r = requests.post(f"{BASE_URL}/upload", timeout=10)
        if r.status_code == 400:
            print(f"  [PASS] Empty upload rejected correctly (HTTP 400)")
        else:
            print(f"  [FAIL] Empty upload not rejected: HTTP {r.status_code}")
    except Exception as e:
        print(f"  [ERROR] Empty upload test: {e}")
    
    return issues


if __name__ == "__main__":
    print("=" * 70)
    print("PixelRevive Comprehensive Pipeline Audit")
    print("=" * 70)
    
    # Check server is running
    try:
        r = requests.get(f"{BASE_URL}/", timeout=5)
        print(f"Server status: Running (HTTP {r.status_code})")
    except Exception as e:
        print(f"ERROR: Server not running: {e}")
        sys.exit(1)
    
    # Test all endpoints
    endpoint_issues = test_all_endpoints()
    
    # Test pipeline with all images
    print("\n\n=== Pipeline Tests ===")
    for img_path, description in TEST_IMAGES:
        result = test_image(img_path, description)
        results.append(result)
    
    # Summary
    print("\n\n" + "="*70)
    print("AUDIT SUMMARY")
    print("="*70)
    
    passed = [r for r in results if r['status'] == 'PASS']
    failed = [r for r in results if r['status'] in ('FAIL', 'ERROR', 'TIMEOUT')]
    skipped = [r for r in results if r['status'] == 'SKIP']
    
    print(f"Total images tested: {len(results)}")
    print(f"  PASSED: {len(passed)}")
    print(f"  FAILED: {len(failed)}")
    print(f"  SKIPPED: {len(skipped)}")
    
    print("\nResults table:")
    print(f"{'Description':<35} {'Status':<10} {'Faces':<7} {'Duration':<12}")
    print("-" * 70)
    for r in results:
        status = r['status']
        faces = r.get('faces', '-')
        duration = f"{r.get('duration', r.get('elapsed', '-'))}s"
        print(f"{r['description']:<35} {status:<10} {str(faces):<7} {duration:<12}")
    
    if failed:
        print("\nFailed tests:")
        for r in failed:
            print(f"  - {r['description']}: {r.get('error', r['status'])[:100]}")
    
    if endpoint_issues:
        print("\nEndpoint issues:")
        for issue in endpoint_issues:
            print(f"  - {issue}")
    
    if not failed and not endpoint_issues:
        print("\n[SUCCESS] All tests passed!")
    else:
        print(f"\n[ISSUES] Found {len(failed)} pipeline failures, {len(endpoint_issues)} endpoint issues")
    
    # Save results JSON
    with open("audit_results.json", "w") as f:
        json.dump({"results": results, "endpoint_issues": endpoint_issues}, f, indent=2)
    print("\nDetailed results saved to audit_results.json")
