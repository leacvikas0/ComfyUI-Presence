import torch

class FluxAdaptiveInjector:
    """
    NODE 2: THE ENGINE
    - Receives the 'Bundle' (List of raw images).
    - Encodes each image INDIVIDUALLY using the VAE (preserving distinct aspect ratios).
    - Injects them into Flux using the 'reference_latents' key.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",), # From CLIP Text Encode
                "vae": ("VAE",),
                "image_bundle": ("UNALIVER_BUNDLE",), # Connects to Planner
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "inject_quality"
    CATEGORY = "Unaliver"

    def inject_quality(self, conditioning, vae, image_bundle):
        print(f"💉 FLUX INJECTOR: Processing {len(image_bundle)} images...")
        
        reference_latents = []

        # 1. ENCODE LOOP
        # We iterate through the bundle. 
        # Since they are in a list, they can all be different sizes (Portrait/Landscape).
        for i, image_tensor in enumerate(image_bundle):
            # image_tensor shape: [1, H, W, C]
            
            # Permute to VAE format: [1, C, H, W]
            pixels = image_tensor.permute(0, 3, 1, 2)
            
            # Encode
            # Note: We access the VAE's encode method directly.
            # Ideally, we should move to GPU, but Comfy's VAE wrapper often handles this.
            try:
                latent_dist = vae.encode(pixels)
                # Flux VAE usually returns a distribution, we need the samples.
                # In standard Comfy nodes, this is usually just the result.
                # We assume standard Comfy VAE behavior here.
                latent = latent_dist
            except Exception as e:
                print(f"❌ VAE Encode Error on Image {i+1}: {e}")
                continue

            reference_latents.append(latent)
            print(f"   - Encoded Image {i+1} (Shape: {latent.shape})")

        # 2. INJECT INTO CONDITIONING
        # We modify the conditioning dictionary to include our list of latents.
        c_out = []
        for t in conditioning:
            # t is [embedding, dictionary]
            d = t[1].copy()
            
            # The Magic Key for Flux Redux / Reference
            key = "reference_latents"
            
            if key in d:
                # If existing references (e.g. from previous nodes), append ours
                d[key] = d[key] + reference_latents
            else:
                # Create new list
                d[key] = reference_latents
            
            n = [t[0], d]
            c_out.append(n)

        print("✅ Injection Complete.")
        return (c_out,)

NODE_CLASS_MAPPINGS = {
    "FluxAdaptiveInjector": FluxAdaptiveInjector
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxAdaptiveInjector": "Flux Adaptive Injector 💉"
}
