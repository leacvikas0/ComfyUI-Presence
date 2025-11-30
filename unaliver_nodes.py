import os
import torch
import numpy as np
from PIL import Image, ImageOps

# --------------------------------------------------------------------------------
# NODE 1: THE PLANNER (The Brain)
# Loads images, bundles them in a list, and sends them to the Injector.
# --------------------------------------------------------------------------------

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
        
        # --- 1. MOCK PLAN GENERATION (Replace with your LLM Logic) ---
        if reset_plan or state_key not in _PLANNER_STATE:
            # This mimics the Agent saying: "Step 1: Dad. Step 2: Dad + Mom."
            mock_plan = [
                {"images": ["IMG_1.png"], "prompt": "Scene 1: Subject on golden throne, clouds"},
                {"images": ["IMG_1.png", "IMG_2.png"], "prompt": "Scene 2: Subject and daughter at wedding"},
                {"images": ["IMG_1.png", "IMG_2.png", "IMG_3.png"], "prompt": "Scene 3: Family reunion, joyful"}
            ]
            _PLANNER_STATE[state_key] = {"plan": mock_plan, "idx": 0}
        # -------------------------------------------------------------

        state = _PLANNER_STATE[state_key]
        idx = state["idx"]
        plan = state["plan"]
        
        # Loop safety
        if idx >= len(plan): idx = len(plan) - 1
            
        current_step = plan[idx]
        filenames = current_step["images"]
        prompt = current_step["prompt"]
        
        print(f"▶️ UNALIVER Step {idx+1}: Loading {len(filenames)} images for Flux...")

        # --- 2. LOAD IMAGES INTO BUNDLE ---
        # We assume files are standardised (IMG_1, IMG_2...)
        bundle = []
        for name in filenames:
            path = os.path.join(directory_path, name)
            if os.path.exists(path):
                try:
                    # Load Raw
                    img = Image.open(path).convert("RGB")
                    img = ImageOps.exif_transpose(img)
                    
                    # Convert to simple Tensor [1, H, W, C]
                    # We do NOT resize here. We let the Injector handle the VAE math.
                    i = np.array(img).astype(np.float32) / 255.0
                    tensor = torch.from_numpy(i)[None,] 
                    bundle.append(tensor)
                except Exception as e:
                    print(f"❌ Error loading {name}: {e}")
            else:
                print(f"⚠️ Image not found: {path}")

        # Advance Step
        if idx < len(plan) - 1:
            state["idx"] += 1
            
        return (bundle, prompt, idx + 1, len(plan))


# --------------------------------------------------------------------------------
# NODE 2: THE INJECTOR (The Engine)
# 1. Enforces 1MP Limit (Safety).
# 2. Encodes to VAE Latents.
# 3. Injects into Conditioning exactly like 'ReferenceLatent'.
# --------------------------------------------------------------------------------

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
                # Default 1.0 MP = 1024x1024 roughly. This is Flux's native resolution.
                "megapixels": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "inject_references"
    CATEGORY = "Unaliver"

    def inject_references(self, conditioning, vae, image_bundle, megapixels=1.0):
        print(f"💉 FLUX INJECTOR: Encoding {len(image_bundle)} references at ~{megapixels} MP...")
        
        reference_latents = []
        
        # Helper to find VAE device (CPU vs CUDA)
        try:
            vae_device = vae.first_stage_model.device
        except:
            vae_device = torch.device("cpu")

        for i, image_tensor in enumerate(image_bundle):
            # DEBUG: Print input shape
            print(f"   📐 Input Image {i+1} shape: {image_tensor.shape}")
            
            # 1. VALIDATE AND RESIZE TO target MEGAPIXELS (Keep Aspect Ratio)
            B, H, W, C = image_tensor.shape
            
            # Safety check
            if C != 3:
                print(f"   ⚠️ WARNING: Image {i+1} has {C} channels (expected 3). Skipping.")
                continue
            if H == 0 or W == 0:
                print(f"   ⚠️ WARNING: Image {i+1} has invalid dimensions {H}x{W}. Skipping.")
                continue
            
            current_pixels = H * W
            target_pixels = megapixels * 1024 * 1024
            
            # Rescale if needed (up or down) to hit the target pixel count
            scale_factor = (target_pixels / current_pixels) ** 0.5
            new_H = max(64, int(H * scale_factor))  # Minimum 64px
            new_W = max(64, int(W * scale_factor))  # Minimum 64px
            
            print(f"   🔄 Resizing: {H}x{W} -> {new_H}x{new_W} (scale: {scale_factor:.2f}x)")
            
            # Permute for torch resize: [B, C, H, W]
            pixels = image_tensor.permute(0, 3, 1, 2)
            print(f"   📐 After permute: {pixels.shape}")
            
            # Bilinear resize is best for photos
            if new_H != H or new_W != W:
                pixels = torch.nn.functional.interpolate(pixels, size=(new_H, new_W), mode="bilinear", align_corners=False)
                print(f"   📐 After resize: {pixels.shape}")
            
            # 2. MOVE TO GPU & ENCODE
            pixels = pixels.to(vae_device)
            
            try:
                # Encode
                latent_dist = vae.encode(pixels)
                
                # Handle "Distribution" vs "Tensor" ambiguity
                if hasattr(latent_dist, "sample"):
                    latent = latent_dist.sample()
                elif isinstance(latent_dist, dict) and "samples" in latent_dist:
                    latent = latent_dist["samples"]
                else:
                    latent = latent_dist
                
                # 3. ADD TO LIST
                reference_latents.append(latent)
                print(f"   ✅ Encoded Image {i+1} successfully")
                
            except Exception as e:
                print(f"❌ VAE Error on Image {i+1}: {e}")
                print(f"   Tensor shape at error: {pixels.shape}")

        # 4. INJECT INTO CONDITIONING
        # This is the exact logic from the 'ReferenceLatent' node
        c_out = []
        for t in conditioning:
            d = t[1].copy()
            
            key = "reference_latents" # The specific Flux 2 Key
            
            if key in d:
                # Append to existing references
                d[key] = d[key] + reference_latents
            else:
                # Create new list
                d[key] = reference_latents
            
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
