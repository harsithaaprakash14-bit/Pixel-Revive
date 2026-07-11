import os
import sys
import shutil
import cv2

# Insert CodeFormer directory to sys.path to allow proper import resolution of basicsr and facelib
CODEFORMER_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'CodeFormer')
if CODEFORMER_DIR in sys.path:
    sys.path.remove(CODEFORMER_DIR)
sys.path.append(CODEFORMER_DIR)

_net = None

def _get_torch():
    import torch
    return torch

def get_codeformer_model(device):
    global _net
    if _net is None:
        torch = _get_torch()
        print("  [FaceRestorer] Initializing CodeFormer architecture...")
        
        # We must import basicsr.archs to register the architecture in registry
        from basicsr.utils.registry import ARCH_REGISTRY
        import basicsr.archs
        basicsr.archs._ensure_arch_modules_imported()
        
        net = ARCH_REGISTRY.get('CodeFormer')(
            dim_embd=512, codebook_size=1024, n_head=8, n_layers=9,
            connect_list=['32', '64', '128', '256']
        ).to(device)
        
        # Pretrained weight URL and destination directory
        pretrain_model_url = 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth'
        model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'weights', 'CodeFormer')
        
        # Load weight using basicsr utility which downloads it automatically if missing
        from basicsr.utils.download_util import load_file_from_url
        print(f"  [FaceRestorer] Checking/Downloading CodeFormer weights from {pretrain_model_url}...")
        ckpt_path = load_file_from_url(
            url=pretrain_model_url,
            model_dir=model_dir,
            progress=True,
            file_name=None
        )
        
        print(f"  [FaceRestorer] Loading weights from {ckpt_path}...")
        checkpoint = torch.load(ckpt_path, map_location=device)['params_ema']
        net.load_state_dict(checkpoint)
        net.eval()
        _net = net
    return _net

def restore_faces(input_path, output_path, fidelity_weight=0.5):
    """
    Detect faces in the image, restore them using CodeFormer, and paste them back.
    If no faces are detected, skip the restoration process and copy input to output.
    """
    torch = _get_torch()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  [FaceRestorer] Starting face restoration on device: {device}")
    
    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to read image at {input_path}")
        
    from facelib.utils.face_restoration_helper import FaceRestoreHelper
    
    # Initialize helper. Since we only restore faces and paste back onto the original background
    # without general image upsampling, we set upscale=1.
    face_helper = FaceRestoreHelper(
        1,
        face_size=512,
        crop_ratio=(1, 1),
        det_model='retinaface_resnet50',
        save_ext='png',
        use_parse=True,
        device=device
    )
    
    face_helper.clean_all()
    face_helper.read_image(img)
    
    # Detect face landmarks. resize=640 reduces large images for faster detection.
    num_det_faces = face_helper.get_face_landmarks_5(
        only_center_face=False,
        resize=640,
        eye_dist_threshold=5
    )
    
    print(f"  [FaceRestorer] Built-in face detector found {num_det_faces} face(s).")
    
    if num_det_faces == 0:
        print("  [FaceRestorer] No faces detected. Skipping face restoration step.")
        shutil.copy(input_path, output_path)
        return 0
        
    # Align and warp faces
    face_helper.align_warp_face()
    
    # Load CodeFormer model
    net = get_codeformer_model(device)
    if device.type == 'cuda' and net is not None:
        print("  [FaceRestorer] Moving CodeFormer model to GPU for inference...")
        net.to(device)
    
    from basicsr.utils import img2tensor, tensor2img
    from torchvision.transforms.functional import normalize
    
    # Restore each cropped face
    for idx, cropped_face in enumerate(face_helper.cropped_faces):
        cropped_face_t = img2tensor(cropped_face / 255.0, bgr2rgb=True, float32=True)
        normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        cropped_face_t = cropped_face_t.unsqueeze(0).to(device)
        
        try:
            with torch.no_grad():
                output = net(cropped_face_t, w=fidelity_weight, adain=True)[0]
                restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
            del output
        except Exception as e:
            print(f"  [FaceRestorer] CodeFormer inference failed for face {idx}: {e}")
            restored_face = cropped_face
            
        restored_face = restored_face.astype('uint8')
        face_helper.add_restored_face(restored_face, cropped_face)
        
    # Paste restored faces back to background
    face_helper.get_inverse_affine(None)
    restored_img = face_helper.paste_faces_to_input_image()
    
    cv2.imwrite(output_path, restored_img)
    print(f"  [FaceRestorer] Restoration complete. Saved restored image to {output_path}")
    
    # Clean up GPU memory
    try:
        if hasattr(face_helper, 'face_det'):
            face_helper.face_det = None
        if hasattr(face_helper, 'face_parse'):
            face_helper.face_parse = None
    except Exception as e:
        print(f"  [FaceRestorer] Warning cleaning up sub-models: {e}")
        
    del face_helper
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    return num_det_faces

def unload_codeformer_model():
    global _net
    if _net is not None:
        print("  [Cleanup] Unloading CodeFormer model from VRAM...")
        _net = None


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python3 face_restorer.py <input_image> <output_image>")
        sys.exit(1)
        
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    restore_faces(sys.argv[1], sys.argv[2])

