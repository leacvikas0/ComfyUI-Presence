import torch
import numpy as np

class UnaliverBundlePreview:
    """
    NODE 3: THE VIEWER
    - Receives the 'UNALIVER_BUNDLE' (List of tensors).
    - Converts it to a standard ComfyUI 'IMAGE' batch.
    - Handles variable aspect ratios by padding to the largest image dimensions.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_bundle": ("UNALIVER_BUNDLE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "unbundle"
    CATEGORY = "Unaliver"

    def unbundle(self, image_bundle):
        if not image_bundle:
            # Return a tiny black image if empty to prevent errors
            return (torch.zeros((1, 64, 64, 3)),)
            
        print(f"📦 UNBUNDLER: Processing {len(image_bundle)} images...")
        
        # 1. FIND MAX DIMENSIONS
        max_h = 0
        max_w = 0
        for img in image_bundle:
            # img shape is [1, H, W, C]
            _, h, w, _ = img.shape
            max_h = max(max_h, h)
            max_w = max(max_w, w)
            
        print(f"   - Target Batch Size: {max_w}x{max_h}")

        # 2. CREATE BATCH TENSOR
        # Shape: [N, Max_H, Max_W, 3]
        batch_size = len(image_bundle)
        # Initialize with zeros (black padding)
        out_batch = torch.zeros((batch_size, max_h, max_w, 3), dtype=torch.float32)
        
        # 3. PASTE IMAGES
        for i, img in enumerate(image_bundle):
            _, h, w, _ = img.shape
            
            # Center the image? Or Top-Left?
            # Let's do Center for nicer preview
            pad_h = (max_h - h) // 2
            pad_w = (max_w - w) // 2
            
            # Copy content
            out_batch[i, pad_h:pad_h+h, pad_w:pad_w+w, :] = img[0]
            
        print("✅ Unbundling Complete.")
        return (out_batch,)

NODE_CLASS_MAPPINGS = {
    "UnaliverBundlePreview": UnaliverBundlePreview
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UnaliverBundlePreview": "Unaliver Bundle to Image 📦"
}
