import os
import torch
import numpy as np
from PIL import Image, ImageOps

# Global memory to store the Story Plan between queue runs
_PLANNER_STATE = {}

class UnaliverPlanner:
    """
    NODE 1: THE BRAIN
    - Reads the Directory.
    - Generates a Step-by-Step Plan (Mocked for testing).
    - Loads images at ORIGINAL RESOLUTION (No resizing/padding).
    - Bundles them into a custom list object.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory_path": ("STRING", {"default": "C:/Inspiration"}),
                "user_instruction": ("STRING", {"multiline": True, "default": "Create a memorial video sequence."}),
                "reset_plan": ("BOOLEAN", {"default": False}),
            }
        }
    
    # We use a custom type "UNALIVER_BUNDLE" to prevent connecting this to standard nodes by accident
    RETURN_TYPES = ("UNALIVER_BUNDLE", "STRING", "INT", "INT")
    RETURN_NAMES = ("Image_Bundle", "Prompt", "Step_Index", "Total_Steps")
    FUNCTION = "execute_plan"
    CATEGORY = "Unaliver"

    def execute_plan(self, directory_path, user_instruction, reset_plan):
        # Create a unique key for this directory so multiple workflows don't clash
        state_key = f"{directory_path}_quality_mode"
        
        # 1. GENERATE PLAN (Runs once or on reset)
        if reset_plan or state_key not in _PLANNER_STATE:
            print(f"🎬 UNALIVER: Generating new plan for {directory_path}...")
            
            # --- MOCK PLAN FOR TESTING ---
            # In production, this is where you call Gemini/Qwen to generate this JSON
            # This simulates the Agent deciding: 1 image -> 3 images -> 2 images
            mock_plan = [
                {
                    "images": ["IMG_1.png"], 
                    "prompt": "Step 1: The subject sits on a golden throne in heaven, cinematic lighting, 8k."
                },
                {
                    "images": ["IMG_1.png", "IMG_2.png", "IMG_3.png"], 
                    "prompt": "Step 2: The subject attends a wedding with family, joyful atmosphere, detailed faces."
                },
                {
                    "images": ["IMG_4.png", "IMG_5.png"], 
                    "prompt": "Step 3: A candid selfie in New York City, shallow depth of field, realistic texture."
                }
            ]
            # -----------------------------
            
            _PLANNER_STATE[state_key] = {"plan": mock_plan, "idx": 0}

        # 2. GET CURRENT STEP
        state = _PLANNER_STATE[state_key]
        idx = state["idx"]
        plan = state["plan"]
        
        # Safety: Stop at the last step if we run out
        if idx >= len(plan):
            idx = len(plan) - 1
            
        current_step = plan[idx]
        filenames = current_step["images"]
        prompt = current_step["prompt"]
        
        print(f"▶️ Executing Step {idx+1}/{len(plan)}")
        print(f"   Files: {filenames}")
        print(f"   Prompt: {prompt[:50]}...")

        # 3. LOAD IMAGES (QUALITY MODE)
        # We do NOT resize to a square. We do NOT pad. 
        # We load them as-is and put them in a Python list.
        bundle = []
        
        for name in filenames:
            path = os.path.join(directory_path, name)
            
            # Check file existence (with simple fallback)
            if not os.path.exists(path):
                # Try finding it without extension or just grabbing the first file for testing
                print(f"⚠️ Warning: {path} not found. Looking for alternatives...")
                if os.path.exists(directory_path):
                     all_files = [f for f in os.listdir(directory_path) if f.endswith(('.png', '.jpg'))]
                     if all_files:
                         path = os.path.join(directory_path, all_files[0]) # Fallback to first image
            
            if os.path.exists(path):
                try:
                    img = Image.open(path).convert("RGB")
                    img = ImageOps.exif_transpose(img)
                    
                    # Convert to Tensor [1, Height, Width, Channels]
                    # Note: We keep Batch Size = 1 per image
                    i = np.array(img).astype(np.float32) / 255.0
                    tensor = torch.from_numpy(i)[None,] 
                    bundle.append(tensor)
                except Exception as e:
                    print(f"❌ Error loading {name}: {e}")
            else:
                print(f"❌ Critical: Could not find image {name}")

        # 4. ADVANCE STEP
        if idx < len(plan) - 1:
            state["idx"] += 1
            
        return (bundle, prompt, idx + 1, len(plan))

NODE_CLASS_MAPPINGS = {
    "UnaliverPlanner": UnaliverPlanner
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UnaliverPlanner": "Unaliver Planner 🧠"
}
