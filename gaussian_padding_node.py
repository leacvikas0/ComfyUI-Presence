# =============================================================================
# GAUSSIAN NOISE PADDING NODE
# =============================================================================
# Adds Gaussian noise padding to images
# Useful for outpainting preparation where solid colors look unnatural
# =============================================================================

import torch
import numpy as np
from PIL import Image

class GaussianNoisePadding:
    """
    Adds Gaussian noise padding to specified sides of an image.
    Perfect for preparing images for outpainting/inpainting.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "left": ("INT", {"default": 0, "min": 0, "max": 2000, "step": 8}),
                "right": ("INT", {"default": 0, "min": 0, "max": 2000, "step": 8}),
                "top": ("INT", {"default": 0, "min": 0, "max": 2000, "step": 8}),
                "bottom": ("INT", {"default": 0, "min": 0, "max": 2000, "step": 8}),
                "noise_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "base_color": (["gray", "white", "black", "match_edges"], {"default": "gray"}),
                "maintain_16_9": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffff}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("padded_image",)
    FUNCTION = "add_noise_padding"
    CATEGORY = "PresenceAI/Utils"

    def add_noise_padding(self, image, left, right, top, bottom, noise_strength, base_color, maintain_16_9, seed):
        """Add Gaussian noise padding to image"""
        
        # Set seed for reproducibility (numpy only accepts 32-bit seeds)
        np.random.seed(seed % (2**32))
        
        # Convert tensor to PIL
        # image shape: (batch, height, width, channels)
        img_np = image[0].cpu().numpy()
        img_np = (img_np * 255).astype(np.uint8)
        img = Image.fromarray(img_np)
        
        original_w, original_h = img.width, img.height
        
        print(f"\n[GaussianNoisePadding]")
        print(f"   Original: {original_w}x{original_h}")
        print(f"   Padding: L={left}, R={right}, T={top}, B={bottom}")
        print(f"   Noise strength: {noise_strength}")
        print(f"   Base color: {base_color}")
        
        # Calculate new dimensions
        new_w = original_w + left + right
        new_h = original_h + top + bottom
        
        # Determine base color
        if base_color == "gray":
            base_rgb = (128, 128, 128)
        elif base_color == "white":
            base_rgb = (255, 255, 255)
        elif base_color == "black":
            base_rgb = (0, 0, 0)
        elif base_color == "match_edges":
            # Sample edge pixels and use average
            img_array = np.array(img)
            edge_pixels = []
            if left > 0:
                edge_pixels.extend(img_array[:, 0, :].tolist())
            if right > 0:
                edge_pixels.extend(img_array[:, -1, :].tolist())
            if top > 0:
                edge_pixels.extend(img_array[0, :, :].tolist())
            if bottom > 0:
                edge_pixels.extend(img_array[-1, :, :].tolist())
            if edge_pixels:
                base_rgb = tuple(np.mean(edge_pixels, axis=0).astype(int))
            else:
                base_rgb = (128, 128, 128)
            print(f"   Edge-matched color: RGB{base_rgb}")
        else:
            base_rgb = (128, 128, 128)
        
        # Create new image with noise
        noise_image = self._create_noise_canvas(new_w, new_h, base_rgb, noise_strength)
        
        # Paste original image
        noise_image.paste(img, (left, top))
        
        print(f"   After padding: {noise_image.width}x{noise_image.height}")
        
        # Maintain 16:9 if requested
        if maintain_16_9:
            noise_image = self._adjust_to_16_9(noise_image, base_rgb, noise_strength)
        
        print(f"   Final: {noise_image.width}x{noise_image.height} (ratio: {noise_image.width/noise_image.height:.3f})")
        
        # Convert back to tensor
        result_np = np.array(noise_image).astype(np.float32) / 255.0
        result_tensor = torch.from_numpy(result_np)[None,]
        
        return (result_tensor,)

    def _create_noise_canvas(self, width, height, base_rgb, noise_strength):
        """Create a canvas filled with Gaussian noise"""
        
        # Create base color array
        canvas = np.zeros((height, width, 3), dtype=np.float32)
        canvas[:, :, 0] = base_rgb[0]
        canvas[:, :, 1] = base_rgb[1]
        canvas[:, :, 2] = base_rgb[2]
        
        # Add Gaussian noise
        noise = np.random.randn(height, width, 3) * 50 * noise_strength
        canvas = canvas + noise
        
        # Clamp to valid range
        canvas = np.clip(canvas, 0, 255).astype(np.uint8)
        
        return Image.fromarray(canvas)

    def _adjust_to_16_9(self, img, base_rgb, noise_strength):
        """Adjust image to 16:9 ratio by adding noise padding"""
        
        target_ratio = 16 / 9
        current_ratio = img.width / img.height
        
        if abs(current_ratio - target_ratio) < 0.01:
            return img
        
        if current_ratio < target_ratio:
            # Too tall - add width
            required_w = int(img.height * target_ratio)
            pad_total = required_w - img.width
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            
            new_canvas = self._create_noise_canvas(required_w, img.height, base_rgb, noise_strength)
            new_canvas.paste(img, (pad_left, 0))
            return new_canvas
        else:
            # Too wide - add height
            required_h = int(img.width / target_ratio)
            pad_total = required_h - img.height
            pad_top = pad_total // 2
            pad_bottom = pad_total - pad_top
            
            new_canvas = self._create_noise_canvas(img.width, required_h, base_rgb, noise_strength)
            new_canvas.paste(img, (0, pad_top))
            return new_canvas


# Register node
NODE_CLASS_MAPPINGS = {
    "GaussianNoisePadding": GaussianNoisePadding
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GaussianNoisePadding": "Gaussian Noise Padding"
}
