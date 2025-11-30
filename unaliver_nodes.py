import os
import torch
import numpy as np
from PIL import Image, ImageOps

# NODE 1: THE PLANNER
_PLANNER_STATE = {}

class UnaliverPlanner:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory_path": ("STRING", {"default": "C:/Inspiration"}),
                "user_instruction": ("STRING", {"multiline": True, "default": "Create a memorial video sequence."}),
                "reset_plan": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("UNALIVER_BUNDLE", "STRING", "INT", "INT")
    RETURN_NAMES = ("Image_Bundle", "Prompt", "Step_Index", "Total_Steps")
    FUNCTION = "execute_plan"
    CATEGORY = "Unaliver"

    def execute_plan(self, directory_path, user_instruction, reset_plan):
        state_key = f"{directory_path}_final"
        
        if reset_plan or state_key not in _PLANNER_STATE:
            mock_plan = [
                {"images": ["IMG_1.png"], "prompt": "Scene 1: Subject on golden throne, clouds"},
                {"images": ["IMG_1.png", "IMG_2.png"], "prompt": "Scene 2: Subject and daughter at wedding"},
                {"images": ["IMG_1.png", "IMG_2.png", "IMG_3.png"], "prompt": "Scene 3: Family reunion, joyful"}
            ]
            _PLANNER_STATE[state_key] = {"plan": mock_plan, "idx": 0}

        state = _PLANNER_STATE[state_key]
        idx = state["idx"]
        plan = state["plan"]
        
        if idx >= len(plan): idx = len(plan) - 1
            
        current_step = plan[idx]
        filenames = current_step["images"]
        prompt = current_step["prompt"]
        
        print(f"▶️ UNALIVER Step {idx+1}: Loading {len(filenames)} images for Flux...")

        bundle = []
        for name in filenames:
            path = os.path.join(directory_path, name)
            if os.path.exists(path):
                try:
                    img = Image.open(path).convert("RGB")
                    img = ImageOps.exif_transpose(img)
                    i = np.array(img).astype(np.float32) / 255.0
                    tensor = torch.from_numpy(i)[None,] 
                    bundle.append(tensor)
                except Exception as e:
                    print(f"❌ Error loading {name}: {e}")
            else:
                print(f"⚠️ Image not found: {path}")

        if idx < len(plan) - 1:
            state["idx"] += 1
            
        return (bundle, prompt, idx + 1, len(plan))


# NODE 2: THE INJECTOR (v3.1 - Fixed NHWC format)
class FluxAdaptiveInjector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",), 
                "vae": ("VAE",),
                "image_bundle": ("UNALIVER_BUNDLE",),
            },
            "optional": {
                "megapixels": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "inject_references"
    CATEGORY = "Unaliver"

    def inject_references(self, conditioning, vae, image_bundle, megapixels=1.0):
        print(f"💉 FLUX INJECTOR v3.1: Encoding {len(image_bundle)} references at ~{megapixels} MP...")
        
        reference_latents = []
        
        # Find VAE device
        try:
            vae_device = vae.first_stage_model.device
        except:
            vae_device = torch.device("cpu")

        for i, image_tensor in enumerate(image_bundle):
            print(f"   📐 Input Image {i+1} shape: {image_tensor.shape}")
            
            # image_tensor is [1, H, W, 3] (NHWC)
            B, H, W, C = image_tensor.shape
            
            if C != 3 or H == 0 or W == 0:
                print(f"   ⚠️ Skipping invalid image {i+1}")
                continue
            
            # Calculate target size
            current_pixels = H * W
            target_pixels = megapixels * 1024 * 1024
            scale_factor = (target_pixels / current_pixels) ** 0.5
            new_H = int(H * scale_factor)
            new_W = int(W * scale_factor)
            
            # Round to multiples of 64
            new_H = ((new_H + 32) // 64) * 64
            new_W = ((new_W + 32) // 64) * 64
            new_H = max(64, new_H)
            new_W = max(64, new_W)
            
            print(f"   🔄 Resizing: {H}x{W} -> {new_H}x{new_W}")
            
            # Resize: need NCHW for interpolate
            pixels_nchw = image_tensor.permute(0, 3, 1, 2).contiguous()
            if new_H != H or new_W != W:
                pixels_nchw = torch.nn.functional.interpolate(
                    pixels_nchw, size=(new_H, new_W), mode="bilinear", align_corners=False
                )
            
            # Convert BACK to NHWC for VAE (ComfyUI standard)
            pixels_nhwc = pixels_nchw.permute(0, 2, 3, 1).contiguous()
            print(f"   📐 Final shape (NHWC): {pixels_nhwc.shape}")
            
            try:
                # Move to VAE device and encode
                # Do NOT scale to [-1,1] - vae.encode does this internally
                pixels_final = pixels_nhwc.to(vae_device)
                latent = vae.encode(pixels_final)
                
                # Handle return types
                if hasattr(latent, "sample"):
                    latent = latent.sample()
                elif isinstance(latent, dict) and "samples" in latent:
                    latent = latent["samples"]
                
                reference_latents.append(latent)
                print(f"   ✅ Encoded Image {i+1} successfully")
                
            except Exception as e:
                print(f"❌ VAE Error on Image {i+1}: {e}")
                import traceback
                traceback.print_exc()

        # Inject LIST of latents into conditioning
        c_out = []
        for t in conditioning:
            d = t[1].copy()
            
            if "reference_latents" in d:
                d["reference_latents"] = d["reference_latents"] + reference_latents
            else:
                d["reference_latents"] = reference_latents
            
            n = [t[0], d]
            c_out.append(n)

        print(f"✅ Injection Complete. {len(reference_latents)} references active.")
        return (c_out,)

NODE_CLASS_MAPPINGS = {
    "UnaliverPlanner": UnaliverPlanner,
    "FluxAdaptiveInjector": FluxAdaptiveInjector
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UnaliverPlanner": "Unaliver Planner 🧠",
    "FluxAdaptiveInjector": "Flux Adaptive Injector 💉"
}
