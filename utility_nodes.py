# =============================================================================
# PRESENCE UTILITY NODES
# =============================================================================
# FluxAdaptiveInjector: Injects reference images into Flux conditioning
# PresenceSaver: Saves generated images to active folder
# =============================================================================

import os
import torch
import numpy as np
from PIL import Image, ImageOps


class FluxAdaptiveInjector:
    """
    Encodes reference images into latents and injects into Flux conditioning.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",), 
                "vae": ("VAE",),
                "images": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "inject_references"
    CATEGORY = "PresenceAI"

    def inject_references(self, conditioning, vae, images):
        print(f"\n[INJECTOR] Encoding {len(images)} reference images...")
        
        reference_latents = []
        
        # Find VAE device
        try:
            vae_device = vae.first_stage_model.device
        except:
            vae_device = torch.device("cpu")
        
        for i, image_tensor in enumerate(images):
            # Handle both single image and batch
            if len(image_tensor.shape) == 3:
                image_tensor = image_tensor.unsqueeze(0)
            
            B, H, W, C = image_tensor.shape
            
            if C != 3 or H == 0 or W == 0:
                print(f"   Skipping invalid image {i+1}")
                continue
            
            try:
                pixels = image_tensor.to(vae_device).contiguous()
                latent = vae.encode(pixels)
                
                if hasattr(latent, "sample"):
                    latent = latent.sample()
                elif isinstance(latent, dict) and "samples" in latent:
                    latent = latent["samples"]
                
                reference_latents.append(latent)
                print(f"   Encoded image {i+1}: {H}x{W} -> latent {latent.shape}")
                
            except Exception as e:
                print(f"   Error encoding image {i+1}: {e}")

        if len(reference_latents) > 0:
            c_out = []
            for t in conditioning:
                d = t[1].copy()
                
                if "reference_latents" in d:
                    d["reference_latents"] = d["reference_latents"] + reference_latents
                else:
                    d["reference_latents"] = reference_latents
                
                c_out.append([t[0], d])
            
            print(f"[INJECTOR] Done. {len(reference_latents)} references injected.")
            return (c_out,)
        else:
            print("[INJECTOR] No references to inject.")
            return (conditioning,)


class PresenceSaver:
    """
    Saves generated images to the active folder with the specified filename.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "active_folder": ("STRING", {"default": "C:/Presence/Job_01"}),
                "filename": ("STRING", {"default": "output"}),
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
            filename = "output"
            
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
            print(f"[SAVER] Saved: {user_path}")
            
            # Also save to ComfyUI temp for preview
            temp_dir = folder_paths.get_temp_directory()
            temp_path = os.path.join(temp_dir, fname)
            img.save(temp_path, compress_level=4)
            
            results.append({
                "filename": fname,
                "subfolder": "",
                "type": "temp"
            })
            
        return {"ui": {"images": results}}

class PresencePreview:
    """
    Preview node to see what images are being sent from the Director.
    Shows the image in ComfyUI's preview panel.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
            },
            "optional": {
                "prompt": ("STRING", {"forceInput": True, "default": ""}),
                "filename": ("STRING", {"forceInput": True, "default": ""}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "preview"
    OUTPUT_NODE = True
    CATEGORY = "PresenceAI"

    def preview(self, images, prompt="", filename=""):
        import folder_paths
        
        print(f"\n[PREVIEW] Showing {len(images)} images")
        print(f"   Prompt: {prompt[:80]}..." if len(prompt) > 80 else f"   Prompt: {prompt}")
        print(f"   Filename: {filename}")
        
        results = []
        for i, tensor in enumerate(images):
            # Get dimensions
            if len(tensor.shape) == 3:
                H, W, C = tensor.shape
            else:
                H, W, C = tensor.shape[1], tensor.shape[2], tensor.shape[3]
            
            print(f"   Image {i+1}: {W}x{H}")
            
            # Save to temp for preview
            array = 255. * tensor.cpu().numpy()
            img = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))
            
            temp_dir = folder_paths.get_temp_directory()
            fname = f"preview_{i}.png"
            temp_path = os.path.join(temp_dir, fname)
            img.save(temp_path, compress_level=4)
            
            results.append({
                "filename": fname,
                "subfolder": "",
                "type": "temp"
            })
        
        return {"ui": {"images": results}, "result": (images,)}


# Register nodes
NODE_CLASS_MAPPINGS = {
    "FluxAdaptiveInjector": FluxAdaptiveInjector,
    "PresenceSaver": PresenceSaver,
    "PresencePreview": PresencePreview
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxAdaptiveInjector": "Flux Injector",
    "PresenceSaver": "Presence Saver",
    "PresencePreview": "👁 Presence Preview"
}

