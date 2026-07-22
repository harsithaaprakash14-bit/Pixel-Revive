"""
PixelRevive — services/upscaler.py
====================================
Person 3 Integration: Real-ESRGAN 4x Super-Resolution

This module calls Person 3's upscaler as an isolated subprocess using the
dedicated 'pixelrevive_upscaler' conda environment. This design ensures:
  - No conflict with the project venv's local basicsr/ folder (used by Person 2)
  - Person 3's full pip-installed basicsr==1.4.2 runs in its own clean Python process
  - The Flask server is fully protected — any failure raises RuntimeError, not sys.exit

Entry point for the AI pipeline:
    from services.upscaler import run_upscaler
    output_path = run_upscaler(input_path, output_path, scale=4)
"""

import sys
import os
import subprocess

def _get_upscaler_python():
    if "UPSCALER_PYTHON" in os.environ and os.path.exists(os.environ["UPSCALER_PYTHON"]):
        return os.environ["UPSCALER_PYTHON"]
    
    linux_conda = "/home/ubuntu/miniconda3/envs/pixelrevive_upscaler/bin/python"
    if os.path.exists(linux_conda):
        return linux_conda
        
    if sys.platform == "win32":
        possible_win_paths = [
            os.path.expanduser(r"~\miniconda3\envs\pixelrevive_upscaler\python.exe"),
            os.path.expanduser(r"~\Anaconda3\envs\pixelrevive_upscaler\python.exe"),
            r"C:\ProgramData\miniconda3\envs\pixelrevive_upscaler\python.exe",
            r"C:\ProgramData\Anaconda3\envs\pixelrevive_upscaler\python.exe",
        ]
        for path in possible_win_paths:
            if os.path.exists(path):
                return path

    return sys.executable

UPSCALER_PYTHON = _get_upscaler_python()

# Path to Person 3's upscaler script (with MODEL_PATH and sys.exit already fixed)
UPSCALER_SCRIPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "person3_module", "person3_upscaler.py"
)


def run_upscaler(input_path, output_path, scale=4, fmt="PNG"):
    """
    Upscale an image using Real-ESRGAN via a subprocess in the isolated conda env.

    Parameters:
        input_path  (str): Path to the input image (output of Person 2 colorizer).
        output_path (str): Path to save the 4x upscaled image.
        scale       (int): Upscale factor — 2 or 4. Default: 4.
        fmt         (str): Output format — 'PNG' or 'JPEG'. Default: 'PNG'.

    Returns:
        str: output_path — path to the saved upscaled image.

    Raises:
        RuntimeError: If the upscaler subprocess fails for any reason.
        FileNotFoundError: If the upscaler script or Python interpreter is missing.
    """
    # Validate the isolated environment exists
    if not os.path.exists(UPSCALER_PYTHON):
        raise FileNotFoundError(
            f"Upscaler Python interpreter not found at: {UPSCALER_PYTHON}\n"
            "Please ensure the 'pixelrevive_upscaler' conda environment is set up."
        )

    # Validate the upscaler script exists
    script_path = os.path.realpath(UPSCALER_SCRIPT)
    if not os.path.exists(script_path):
        raise FileNotFoundError(
            f"Upscaler script not found at: {script_path}"
        )

    # Build the subprocess command using person3_upscaler.py's argparse CLI
    command = [
        UPSCALER_PYTHON,
        script_path,
        "--input",  input_path,
        "--output", output_path,
        "--scale",  str(scale),
        "--format", fmt,
    ]

    # Run the subprocess from person3_module/ so MODEL_PATH resolves correctly
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            cwd=os.path.dirname(script_path),
            timeout=300,  # 5-minute hard limit — prevents infinite hangs on CPU
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            "Real-ESRGAN upscaler subprocess timed out after 300 seconds. "
            "The image may be too large for CPU-only inference."
        )

    # Raise a proper error so Flask can handle it — never crashes the server
    if result.returncode != 0:
        raise RuntimeError(
            f"Real-ESRGAN upscaler failed (exit code {result.returncode}):\n"
            f"STDOUT: {result.stdout}\n"
            f"STDERR: {result.stderr}"
        )

    return output_path
