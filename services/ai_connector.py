"""
PixelRevive — services/ai_connector.py
=======================================
Main AI pipeline entry point.  Called by app.py on every image upload.

Pipeline order:
  Step 1: [ACTIVE]  Person 1 — LaMa Damage Removal
  Step 2: [ACTIVE]  Person 2 — DDColor Colorization
  Step 3: [ACTIVE]  Person 3 — Real-ESRGAN 4x Upscaling (isolated subprocess)

The pipeline is deliberately linear: each step receives the output of the previous.
GPU memory is explicitly flushed between every step so the HAMI 2 GiB virtual
GPU budget is maximally available for the next model.

MODIFICATIONS (from original):
  1. Activated Step 1 (LaMa) — previously commented out awaiting mask module.
     Auto-mask generation is now handled internally by services/damage_remover.py.
  2. Added _get_damage_remover() lazy-load cache (mirrors _get_colorizer()).
  3. Added GPU cleanup after Step 1 (same pattern as after Step 2).
  4. Added _cleanup_gpu_memory() utility called between every pipeline stage.
"""

import gc
import os

# ── Lazy torch import guard ────────────────────────────────────────────────────
# torch is available in the project venv (DDColor and LaMa depend on it) but
# we import it lazily here so ai_connector can be tested without a GPU present.
try:
    import torch as _torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# ─── Module-level model caches ────────────────────────────────────────────────
# Both LaMa (~200 MB) and DDColor (~900 MB) are expensive to reload per request.
# Each is loaded once on the first request and reused for all subsequent calls.
_damage_remover = None   # (SimpleLama, torch.device) tuple from services.damage_remover
_colorizer      = None   # ColorizationPipeline from services.colorizer


def _get_damage_remover():
    """
    Lazily load and cache the LaMa damage-removal model.
    Returns (SimpleLama, device) on subsequent calls without reloading.
    """
    global _damage_remover
    if _damage_remover is None:
        from services.damage_remover import load_damage_remover
        _damage_remover = load_damage_remover()
    return _damage_remover


def _get_colorizer():
    """
    Lazily load and cache the DDColor colorization model.
    Returns the cached ColorizationPipeline on subsequent calls.
    """
    global _colorizer
    if _colorizer is None:
        from services.colorizer import load_colorizer
        _colorizer = load_colorizer()
    return _colorizer


def _get_ram_usage():
    try:
        with open('/proc/self/status', 'r') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) / 1024.0
    except Exception:
        pass
    return 0.0

def _get_system_available_ram():
    try:
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if line.startswith('MemAvailable:'):
                    return int(line.split()[1]) / 1024.0
    except Exception:
        pass
    return 0.0

def _get_gpu_usage():
    if _TORCH_AVAILABLE and _torch.cuda.is_available():
        allocated = _torch.cuda.memory_allocated() / (1024 * 1024)
        reserved = _torch.cuda.memory_reserved() / (1024 * 1024)
        return allocated, reserved
    return 0.0, 0.0

def _get_device_free_vram():
    if _TORCH_AVAILABLE and _torch.cuda.is_available():
        try:
            free, total = _torch.cuda.mem_get_info()
            return free / (1024.0 * 1024.0)
        except Exception:
            pass
    return 0.0

def _log_mem(stage, point):
    proc_ram = _get_ram_usage()
    sys_ram = _get_system_available_ram()
    allocated, reserved = _get_gpu_usage()
    free_vram = _get_device_free_vram()
    print(f"[MEMORY LOG] Stage: {stage} | Point: {point} | Host RAM: {proc_ram:.2f} MB (Proc), {sys_ram:.2f} MB (Sys Avail) | GPU: {allocated:.2f} MB (Alloc), {reserved:.2f} MB (Res), {free_vram:.2f} MB (Device Free)")
    import sys
    sys.stdout.flush()

def _cleanup_gpu_memory():
    """
    Flush the CUDA allocator cache between pipeline steps.

    After each model's forward pass, PyTorch retains "reserved but unallocated"
    VRAM (visible in OOM error messages).  Under the HAMI 2 GiB virtual GPU cap,
    this stranded memory prevents the next model from allocating successfully.

    Steps:
      1. gc.collect()             — finalize Python objects holding tensor refs.
      2. torch.cuda.synchronize() — wait for in-flight CUDA kernels.
      3. torch.cuda.empty_cache() — return reserved-but-unallocated VRAM to OS.
      4. torch.cuda.synchronize() — confirm empty_cache completed.

    Does NOT unload model weights — those stay in VRAM for reuse.
    """
    gc.collect()
    if _TORCH_AVAILABLE and _torch.cuda.is_available():
        _torch.cuda.synchronize()
        _torch.cuda.empty_cache()
        _torch.cuda.synchronize()
        print("[OK] GPU cache cleared between pipeline steps.")
    import ctypes
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
        print("[OK] Host RAM trimmed.")
    except Exception as e:
        print(f"[Warning] Failed to trim host RAM: {e}")


def _unload_damage_remover():
    global _damage_remover
    _damage_remover = None
        
    try:
        import services.damage_remover
        services.damage_remover._lama = None
        services.damage_remover._lama_device = None
    except Exception:
        pass
        
    _cleanup_gpu_memory()


def _unload_colorizer():
    global _colorizer
    _colorizer = None
        
    _cleanup_gpu_memory()


def unload_all_models():
    print("  [Cleanup] Unloading all models from VRAM/RAM...")
    _unload_damage_remover()
    _unload_colorizer()
    try:
        from services.face_restorer import unload_codeformer_model
        unload_codeformer_model()
    except Exception as e:
        print(f"  [Cleanup] Warning unloading CodeFormer: {e}")
    _cleanup_gpu_memory()


def process_image(input_path):
    import os
    import gc
    import sys
    import subprocess

    sys_ram = _get_system_available_ram()
    free_vram = _get_device_free_vram()

    # Raise Host RAM threshold to 2500 MB for proactive low memory mode
    proactive_low_mem = False
    if sys_ram < 2500.0 or (free_vram > 0.0 and free_vram < 1500.0):
        proactive_low_mem = True
        print(f"[MEMORY CHECK] Proactive Low-Memory Mode triggered (Available RAM: {sys_ram:.2f} MB, Free VRAM: {free_vram:.2f} MB)")
    else:
        print(f"[MEMORY CHECK] Available RAM: {sys_ram:.2f} MB, Free VRAM: {free_vram:.2f} MB. Running in Normal Mode.")

    low_memory_mode = proactive_low_mem

    input_path  = os.path.abspath(input_path)
    filename    = os.path.basename(input_path)
    name, ext   = os.path.splitext(filename)
    outputs_dir = os.path.join(os.path.dirname(os.path.dirname(input_path)), "outputs")
    os.makedirs(outputs_dir, exist_ok=True)

    current_path = input_path

    # --- Stage 1: Damage Removal ---
    step1_output = os.path.join(outputs_dir, f"{name}_repaired{ext}")
    
    def run_stage_1(low_mem):
        if low_mem:
            unload_all_models()
            _log_mem("Damage Removal (Subprocess)", "Before")
            script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "damage_remover.py")
            env = os.environ.copy()
            env["PYTHONPATH"] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            command = [sys.executable, script_path, current_path, step1_output]
            result = subprocess.run(command, env=env, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"Damage Removal subprocess failed (code {result.returncode}):\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}")
            _log_mem("Damage Removal (Subprocess)", "After")
        else:
            _log_mem("Damage Removal", "Before")
            from services.damage_remover import restore_image
            restore_image(current_path, step1_output)
            _log_mem("Damage Removal", "After")

    try:
        run_stage_1(low_mem=low_memory_mode)
    except Exception as e:
        err_str = str(e).lower()
        is_oom = "out of memory" in err_str or "oom" in err_str or isinstance(e, MemoryError)
        if _TORCH_AVAILABLE and hasattr(_torch.cuda, 'OutOfMemoryError') and isinstance(e, _torch.cuda.OutOfMemoryError):
            is_oom = True
        
        if is_oom and not low_memory_mode:
            print(f"[MEMORY WARNING] OOM in Normal Mode during Damage Removal. Retrying in Low-Memory Mode...")
            low_memory_mode = True
            unload_all_models()
            run_stage_1(low_mem=True)
        else:
            raise e

    current_path = step1_output

    # --- Stage 2: Colorization ---
    step2_output = os.path.join(outputs_dir, f"{name}_colorized{ext}")

    from services.colorizer import is_grayscale_image
    import cv2
    import shutil
    
    img_cv = cv2.imread(current_path)
    if img_cv is not None and not is_grayscale_image(img_cv):
        print(f"[Colorizer] Image is already in color. Skipping colorization model loading.")
        shutil.copy(current_path, step2_output)
    else:
        def run_stage_2(low_mem):
            if low_mem:
                unload_all_models()
                _log_mem("Colorization (Subprocess)", "Before")
                script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "colorizer.py")
                env = os.environ.copy()
                env["PYTHONPATH"] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                command = [sys.executable, script_path, current_path, step2_output]
                result = subprocess.run(command, env=env, capture_output=True, text=True)
                if result.returncode != 0:
                    raise RuntimeError(f"Colorization subprocess failed (code {result.returncode}):\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}")
                _log_mem("Colorization (Subprocess)", "After")
            else:
                _log_mem("Colorization", "Before")
                from services.colorizer import load_colorizer, colorize_image
                c = _get_colorizer()
                colorize_image(c, current_path, step2_output)
                _log_mem("Colorization", "After")

        try:
            run_stage_2(low_mem=low_memory_mode)
        except Exception as e:
            err_str = str(e).lower()
            is_oom = "out of memory" in err_str or "oom" in err_str or isinstance(e, MemoryError)
            if _TORCH_AVAILABLE and hasattr(_torch.cuda, 'OutOfMemoryError') and isinstance(e, _torch.cuda.OutOfMemoryError):
                is_oom = True
            
            if is_oom and not low_memory_mode:
                print(f"[MEMORY WARNING] OOM in Normal Mode during Colorization. Retrying in Low-Memory Mode...")
                low_memory_mode = True
                unload_all_models()
                run_stage_2(low_mem=True)
            else:
                raise e

    current_path = step2_output

    # --- Stage 2.5: Face Restoration ---
    step2_5_output = os.path.join(outputs_dir, f"{name}_face_restored{ext}")
    faces_detected_count = 0

    def run_stage_2_5(low_mem):
        nonlocal faces_detected_count
        if low_mem:
            unload_all_models()
            _log_mem("Face Restoration (Subprocess)", "Before")
            script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_restorer.py")
            env = os.environ.copy()
            env["PYTHONPATH"] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            command = [sys.executable, script_path, current_path, step2_5_output]
            result = subprocess.run(command, env=env, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"Face Restoration subprocess failed (code {result.returncode}):\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}")
            
            # Parse faces detected count from stdout
            faces_detected_count = 0
            for line in result.stdout.splitlines():
                if "Built-in face detector found" in line:
                    try:
                        parts = line.split("found")
                        if len(parts) > 1:
                            count_str = parts[1].split("face")[0].strip()
                            faces_detected_count = int(count_str)
                    except Exception:
                        pass
            _log_mem("Face Restoration (Subprocess)", "After")
        else:
            _log_mem("Face Restoration", "Before")
            from services.face_restorer import restore_faces
            faces_detected_count = restore_faces(current_path, step2_5_output)
            _log_mem("Face Restoration", "After")

    try:
        run_stage_2_5(low_mem=low_memory_mode)
    except Exception as e:
        err_str = str(e).lower()
        is_oom = "out of memory" in err_str or "oom" in err_str or isinstance(e, MemoryError)
        if _TORCH_AVAILABLE and hasattr(_torch.cuda, 'OutOfMemoryError') and isinstance(e, _torch.cuda.OutOfMemoryError):
            is_oom = True
        
        if is_oom and not low_memory_mode:
            print(f"[MEMORY WARNING] OOM in Normal Mode during Face Restoration. Retrying in Low-Memory Mode...")
            low_memory_mode = True
            unload_all_models()
            run_stage_2_5(low_mem=True)
        else:
            raise e

    current_path = step2_5_output

    # --- Stage 3: Upscaling (Isolated Conda Env Subprocess) ---
    def run_stage_3(low_mem):
        if low_mem:
            unload_all_models()
        _log_mem("Upscaling", "Before")
        try:
            from services.upscaler import run_upscaler
            step3_output = os.path.join(outputs_dir, f"{name}_4x.png")
            run_upscaler(current_path, step3_output, scale=4, fmt="PNG")
            return step3_output
        finally:
            _log_mem("Upscaling", "After")

    try:
        step3_output = run_stage_3(low_mem=low_memory_mode)
    except Exception as e:
        err_str = str(e).lower()
        is_oom = "out of memory" in err_str or "oom" in err_str or isinstance(e, MemoryError)
        if _TORCH_AVAILABLE and hasattr(_torch.cuda, 'OutOfMemoryError') and isinstance(e, _torch.cuda.OutOfMemoryError):
            is_oom = True
        
        if is_oom and not low_memory_mode:
            print(f"[MEMORY WARNING] OOM in Normal Mode during Upscaling. Retrying in Low-Memory Mode...")
            low_memory_mode = True
            unload_all_models()
            step3_output = run_stage_3(low_mem=True)
        else:
            raise e

    current_path = step3_output

    return {
        'processed_path': current_path,
        'faces_detected': faces_detected_count
    }
