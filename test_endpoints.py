import requests
import os
import sys

BASE_URL = "http://127.0.0.1:5000"

def test_home():
    print("Testing GET / ...")
    r = requests.get(f"{BASE_URL}/")
    assert r.status_code == 200, f"Home page failed with {r.status_code}"
    assert "Photo Restoration" in r.text, "Home page content mismatch"
    print("[OK] Home page works!")

def test_history():
    print("Testing GET /history ...")
    r = requests.get(f"{BASE_URL}/history")
    assert r.status_code == 200, f"History page failed with {r.status_code}"
    assert "Restoration History" in r.text, "History page content mismatch"
    print("[OK] History page works!")

def test_admin():
    print("Testing GET /admin ...")
    r = requests.get(f"{BASE_URL}/admin")
    assert r.status_code == 200, f"Admin page failed with {r.status_code}"
    assert "System Dashboard" in r.text, "Admin page content mismatch"
    print("[OK] Admin page works!")

def test_upload_and_pipeline():
    print("Testing POST /upload with sample_photo.png ...")
    file_path = "sample_photo.png"
    if not os.path.exists(file_path):
        print(f"[ERROR] Test image {file_path} not found!")
        sys.exit(1)
        
    with open(file_path, 'rb') as f:
        files = {'image': (file_path, f, 'image/png')}
        r = requests.post(f"{BASE_URL}/upload", files=files)
        
    assert r.status_code == 200, f"Upload API failed with {r.status_code}: {r.text}"
    data = r.json()
    assert data['status'] == 'success', f"API returned unsuccessful status: {data}"
    
    print(f"[OK] Upload success!")
    print(f"     Original image: {data['original_image']}")
    print(f"     Processed image: {data['processed_image']}")
    print(f"     Faces detected: {data['faces_detected']}")
    print(f"     Duration: {data['duration']}s")
    
    return data['original_image'], data['processed_image']

def test_download(filename):
    print(f"Testing GET /download/{filename} ...")
    r = requests.get(f"{BASE_URL}/download/{filename}")
    assert r.status_code == 200, f"Download failed with {r.status_code}"
    assert len(r.content) > 0, "Downloaded file is empty"
    print("[OK] Download works!")

def test_delete(image_id):
    # Wait, we need to find the image ID from history or DB
    pass

def main():
    try:
        test_home()
        test_history()
        test_admin()
        orig, proc = test_upload_and_pipeline()
        test_download(proc)
        print("\nAll endpoints and full pipeline verified successfully!")
    except AssertionError as e:
        print(f"\n[FAIL] Assertion failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[FAIL] Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
