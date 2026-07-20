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
    print("Testing POST /upload with sample_photo.png (async)...")
    file_path = "sample_photo.png"
    if not os.path.exists(file_path):
        print(f"[ERROR] Test image {file_path} not found!")
        sys.exit(1)
        
    with open(file_path, 'rb') as f:
        files = {'image': (file_path, f, 'image/png')}
        r = requests.post(f"{BASE_URL}/upload", files=files)
        
    assert r.status_code == 202, f"Upload API failed with {r.status_code}: {r.text}"
    data = r.json()
    assert data['status'] == 'processing', f"Expected status 'processing', got {data}"
    job_id = data['job_id']
    print(f"[OK] Upload success! job_id: {job_id}")
    
    # Poll status
    print("Polling status...")
    import time
    start = time.time()
    while True:
        if time.time() - start > 300:
            raise AssertionError("Pipeline timeout (took >300s)")
        poll_resp = requests.get(f"{BASE_URL}/status/{job_id}").json()
        status = poll_resp.get("status")
        if status == "done":
            res = poll_resp["result"]
            print(f"[OK] Pipeline complete!")
            print(f"     Original image: {res['original_image']}")
            print(f"     Processed image: {res['processed_image']}")
            print(f"     Faces detected: {res['faces_detected']}")
            print(f"     Duration: {res['duration']}s")
            return res['original_image'], res['processed_image']
        elif status == "error":
            raise AssertionError(f"Pipeline processing failed: {poll_resp.get('error')}")
        time.sleep(2)

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
