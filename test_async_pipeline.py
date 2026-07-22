import sys, time, requests

BASE = "http://127.0.0.1:5000"
IMAGE = "sample_photo.png"
POLL_INTERVAL = 2.0
MAX_WAIT = 600

print(f"[1/3] Uploading {IMAGE} ...")
with open(IMAGE, 'rb') as f:
    resp = requests.post(f"{BASE}/upload", files={"image": (IMAGE, f, "image/png")}, timeout=30)
print(f"  HTTP {resp.status_code}: {resp.text[:200]}")
if resp.status_code != 202:
    print(f"[FAIL] Expected 202, got {resp.status_code}")
    sys.exit(1)

data = resp.json()
job_id = data.get("job_id")
if not job_id:
    print(f"[FAIL] No job_id in response: {data}")
    sys.exit(1)

print(f"[OK]   job_id = {job_id}")
print(f"[2/3] Polling /status/{job_id} every {POLL_INTERVAL}s ...")

start = time.time()
while True:
    elapsed = time.time() - start
    if elapsed > MAX_WAIT:
        print(f"[FAIL] Timed out after {MAX_WAIT}s")
        sys.exit(1)
    try:
        poll_resp = requests.get(f"{BASE}/status/{job_id}", timeout=10)
        poll_data = poll_resp.json()
    except Exception as e:
        print(f"  [{elapsed:5.1f}s] Poll failed: {e}. Retrying...")
        time.sleep(POLL_INTERVAL)
        continue
    status = poll_data.get("status")
    print(f"  [{elapsed:5.1f}s] status = {status}")
    if status == "done":
        result = poll_data["result"]
        print(f"\n[3/3] Pipeline complete:")
        print(f"  original_image   : {result['original_image']}")
        print(f"  processed_image  : {result['processed_image']}")
        print(f"  faces_detected   : {result['faces_detected']}")
        print(f"  duration         : {result['duration']}s")
        print("\n[PASS] Async pipeline verified successfully!")
        sys.exit(0)
    if status == "error":
        print(f"[FAIL] Pipeline error: {poll_data.get('error')}")
        sys.exit(1)
    time.sleep(POLL_INTERVAL)
