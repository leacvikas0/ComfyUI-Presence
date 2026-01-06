# =============================================================================
# PRESENCE PADDING TESTER
# =============================================================================
# Standalone node to test padding values and see results
# =============================================================================

import os
import torch
import numpy as np
from PIL import Image, ImageOps


class PresencePaddingTester:
    """
    Test node for experimenting with padding percentages.
    Adds gray padding to one side and auto-adjusts to 16:9.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "direction": (["left", "right", "top", "bottom"],),
                "percentage": ("INT", {"default": 20, "min": 5, "max": 50, "step": 5}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("padded_image", "mask")
    FUNCTION = "pad_image"
    CATEGORY = "PresenceAI"
    OUTPUT_NODE = True

    def pad_image(self, image, direction, percentage):
        import folder_paths
        
        # Convert tensor to PIL
        img_array = image[0].cpu().numpy()
        img_array = (img_array * 255).astype(np.uint8)
        img = Image.fromarray(img_array)
        
        original_w, original_h = img.width, img.height
        print(f"\n[PADDING TESTER]")
        print(f"   Input: {original_w}x{original_h}")
        print(f"   Direction: {direction}, Percentage: {percentage}%")
        
        # Calculate padding
        if direction in ["left", "right"]:
            pad_px = int(img.width * percentage / 100)
        else:
            pad_px = int(img.height * percentage / 100)
        
        print(f"   Padding pixels: {pad_px}px")
        
        # Create canvas with gray fill
        if direction == "left":
            new_w = img.width + pad_px
            canvas = Image.new("RGB", (new_w, img.height), (128, 128, 128))
            canvas.paste(img, (pad_px, 0))
            mask = Image.new("L", (new_w, img.height), 0)
            mask.paste(Image.new("L", (pad_px, img.height), 255), (0, 0))
            
        elif direction == "right":
            new_w = img.width + pad_px
            canvas = Image.new("RGB", (new_w, img.height), (128, 128, 128))
            canvas.paste(img, (0, 0))
            mask = Image.new("L", (new_w, img.height), 0)
            mask.paste(Image.new("L", (pad_px, img.height), 255), (img.width, 0))
            
        elif direction == "top":
            new_h = img.height + pad_px
            canvas = Image.new("RGB", (img.width, new_h), (128, 128, 128))
            canvas.paste(img, (0, pad_px))
            mask = Image.new("L", (img.width, new_h), 0)
            mask.paste(Image.new("L", (img.width, pad_px), 255), (0, 0))
            
        else:  # bottom
            new_h = img.height + pad_px
            canvas = Image.new("RGB", (img.width, new_h), (128, 128, 128))
            canvas.paste(img, (0, 0))
            mask = Image.new("L", (img.width, new_h), 0)
            mask.paste(Image.new("L", (img.width, pad_px), 255), (0, img.height))
        
        print(f"   After padding: {canvas.width}x{canvas.height}")
        
        # Adjust to 16:9
        target_ratio = 16 / 9
        current_ratio = canvas.width / canvas.height
        
        if abs(current_ratio - target_ratio) > 0.01:
            if current_ratio < target_ratio:
                # Too tall, add width on both sides
                new_w = int(canvas.height * target_ratio)
                pad_total = new_w - canvas.width
                pad_l = pad_total // 2
                
                new_canvas = Image.new("RGB", (new_w, canvas.height), (128, 128, 128))
                new_canvas.paste(canvas, (pad_l, 0))
                
                new_mask = Image.new("L", (new_w, canvas.height), 255)
                new_mask.paste(mask, (pad_l, 0))
                
                canvas = new_canvas
                mask = new_mask
            else:
                # Too wide, add height on top/bottom
                new_h = int(canvas.width / target_ratio)
                pad_total = new_h - canvas.height
                pad_t = pad_total // 2
                
                new_canvas = Image.new("RGB", (canvas.width, new_h), (128, 128, 128))
                new_canvas.paste(canvas, (0, pad_t))
                
                new_mask = Image.new("L", (canvas.width, new_h), 255)
                new_mask.paste(mask, (0, pad_t))
                
                canvas = new_canvas
                mask = new_mask
        
        print(f"   After 16:9: {canvas.width}x{canvas.height}")
        
        # Ensure dimensions divisible by 16
        new_w = canvas.width - (canvas.width % 16)
        new_h = canvas.height - (canvas.height % 16)
        
        if new_w != canvas.width or new_h != canvas.height:
            canvas = canvas.resize((new_w, new_h), Image.LANCZOS)
            mask = mask.resize((new_w, new_h), Image.LANCZOS)
        
        print(f"   Final (div by 16): {canvas.width}x{canvas.height}")
        
        # Calculate mask percentage
        mask_array = np.array(mask)
        mask_pct = (mask_array > 128).sum() / mask_array.size * 100
        print(f"   Mask covers: {mask_pct:.1f}% of image")
        
        # Convert to tensors
        img_tensor = torch.from_numpy(np.array(canvas).astype(np.float32) / 255.0)[None,]
        mask_tensor = torch.from_numpy(np.array(mask).astype(np.float32) / 255.0)[None,]
        
        # Save preview
        temp_dir = folder_paths.get_temp_directory()
        canvas.save(os.path.join(temp_dir, "padding_test.png"))
        mask.save(os.path.join(temp_dir, "padding_mask.png"))
        
        return {"ui": {"images": [{"filename": "padding_test.png", "subfolder": "", "type": "temp"}]},
                "result": (img_tensor, mask_tensor)}


# Register
NODE_CLASS_MAPPINGS = {
    "PresencePaddingTester": PresencePaddingTester
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PresencePaddingTester": "🔲 Padding Tester"
}
