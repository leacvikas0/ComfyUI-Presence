import os
import torch
import numpy as np
from PIL import Image, ImageOps

# NODE 1: THE INJECTOR (v4.0 - Research-backed NHWC implementation)
class FluxAdaptiveInjector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",), 
                "vae": ("VAE",),
                "image_bundle": ("UNALIVER_BUNDLE",),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "inject_references"
    CATEGORY = "PresenceAI"

    def inject_references(self, conditioning, vae, image_bundle):
        print(f"💉 FLUX INJECTOR v4.0 (NHWC-correct): Encoding {len(image_bundle)} references...")
        
        reference_latents = []
        
        # Find VAE device
        try:
            vae_device = vae.first_stage_model.device
        except:
            vae_device = torch.device("cpu")
        
        print(f"   VAE device: {vae_device}")

        for i, image_tensor in enumerate(image_bundle):
            # image_tensor is already [1, H, W, 3] NHWC from Director
            B, H, W, C = image_tensor.shape
            print(f"   📐 Image {i+1} input shape: {image_tensor.shape} (NHWC)")
            
            if C != 3 or H == 0 or W == 0:
                print(f"   ⚠️ Skipping invalid image {i+1}")
                continue
            
            # Images come pre-sized from Director - no resizing here
            # Convert to NCHW for format consistency
            pixels_nchw = image_tensor.permute(0, 3, 1, 2).contiguous()
            
            # Convert BACK to NHWC (ComfyUI VAE requirement)
            pixels_nhwc = pixels_nchw.permute(0, 2, 3, 1).contiguous()
            
            # Verify format before VAE
            assert pixels_nhwc.shape[-1] == 3, f"Expected 3 channels, got {pixels_nhwc.shape[-1]}"
            assert pixels_nhwc.min() >= 0 and pixels_nhwc.max() <= 1, f"Range must be [0,1], got [{pixels_nhwc.min():.2f}, {pixels_nhwc.max():.2f}]"
            
            print(f"   📐 Pre-VAE shape: {pixels_nhwc.shape} (NHWC, range [{pixels_nhwc.min():.2f}, {pixels_nhwc.max():.2f}])")
            
            try:
                # Move to VAE device
                pixels_final = pixels_nhwc.to(vae_device).contiguous()
                print(f"   🖥️ Moved to: {pixels_final.device}")
                
                # VAE encodes [0, 1] NHWC tensor
                latent = vae.encode(pixels_final)
                
                # Handle return types
                if hasattr(latent, "sample"):
                    latent = latent.sample()
                elif isinstance(latent, dict) and "samples" in latent:
                    latent = latent["samples"]
                
                reference_latents.append(latent)
                print(f"   ✅ Encoded Image {i+1} → latent shape: {latent.shape}")
                
            except Exception as e:
                print(f"❌ VAE Error on Image {i+1}: {e}")
                import traceback
                traceback.print_exc()

        # Inject LIST of latents into conditioning (Flux 2 expects list)
        if len(reference_latents) > 0:
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
        else:
            print("⚠️ No references to inject. Passing conditioning through unchanged.")
            return (conditioning,)


# NODE 2: THE SAVER
class PresenceSaver:
    """
    💾 PRESENCE SAVER
    - Saves images to the active_folder using the filename from the Director.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "active_folder": ("STRING", {"default": "C:/Presence/Job_01"}),
                "filename": ("STRING", {"default": "gen"}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "PresenceAI"

    def save_images(self, images, active_folder, filename):
        import folder_paths
        
        if not os.path.exists(active_folder):
            os.makedirs(active_folder, exist_ok=True)
            
        # Clean filename
        if not filename or filename.strip() == "":
            filename = "gen"
            
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            filename = os.path.splitext(filename)[0]
            
        results = []
        for i, tensor in enumerate(images):
            array = 255. * tensor.cpu().numpy()
            img = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))
            
            if len(images) == 1:
                fname = f"{filename}.png"
            else:
                fname = f"{filename}_{i+1}.png"
                
            # Save to user's folder
            user_path = os.path.join(active_folder, fname)
            img.save(user_path, compress_level=4)
            print(f"   💾 Saved to folder: {user_path}")
            
            # ALSO save to ComfyUI temp for preview
            temp_dir = folder_paths.get_temp_directory()
            temp_path = os.path.join(temp_dir, fname)
            img.save(temp_path, compress_level=4)
            
            results.append({
                "filename": fname,
                "subfolder": "",
                "type": "temp"
            })
            
        return {"ui": {"images": results}}


NODE_CLASS_MAPPINGS = {
    "FluxAdaptiveInjector": FluxAdaptiveInjector,
    "PresenceSaver": PresenceSaver
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxAdaptiveInjector": "💉 Flux Adaptive Injector",
    "PresenceSaver": "💾 Presence Saver"
}
