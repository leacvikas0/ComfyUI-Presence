# Unaliver Step Iterator Node - FULLY INDEPENDENT
# Gets a multi-step editing plan from AI in ONE call, then outputs each step sequentially

import os
import json
import requests
import base64
import re
import torch
import numpy as np
from PIL import Image, ImageOps
from typing import Dict

# Global state to persist iterator positions
_UNALIVER_STATE: Dict[str, Dict] = {}

class UnaliverStepIterator:
    """
    AI-driven multi-step image editing planner.
    
    Workflow:
    1. First execution: Sends all images to AI, gets full editing plan (up to 20 steps)
    2. Subsequent executions: Outputs one step at a time
    3. Each step contains: images mentioned + editing prompt
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        default_prompt = """You are 'Unaliver', an AI image editing director.

You will receive multiple images named IMG_1, IMG_2, etc.
Create a step-by-step editing plan (up to 20 steps).

OUTPUT FORMAT (one step per line):
Step 1: [IMG_2, IMG_3] → "The woman now stands with the man"
Step 2: [IMG_1] → "Add ethereal background lighting"
Step 3: [IMG_1, IMG_5] → "Merge family portrait with surreal elements"

Each step should describe a specific editing action."""
        
        return {
            "required": {
                "directory_path": ("STRING", {"default": "C:/Inspiration"}),
                "openrouter_api_key": ("STRING", {"multiline": False}),
                "system_prompt": ("STRING", {"multiline": True, "default": default_prompt}),
                "user_instruction": ("STRING", {"multiline": True, "default": "Create a creative editing sequence."}),
                "loop": ("BOOLEAN", {"default": False}),
                "reset": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "STRING", "INT", "INT", "INT")
    RETURN_NAMES = ("Image_1", "Image_2", "Image_3", "Image_4", "Step_Prompt", "step_index", "total_steps", "done")
    FUNCTION = "next_step"
    CATEGORY = "Unaliver/Archive"
    
    def __init__(self):
        self.api_key_hardcoded = "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"
    
    def next_step(self, directory_path, openrouter_api_key, system_prompt, user_instruction, loop, reset):
        key = f"{directory_path}|{user_instruction}"
        state = _UNALIVER_STATE.get(key)
        
        # FIRST RUN or RESET: Call AI and get full plan
        if reset or state is None:
            print("\n" + "="*60)
            print("🎬 GETTING EDITING PLAN FROM AI...")
            print("="*60)
            
            steps = self._get_ai_plan(directory_path, openrouter_api_key, system_prompt, user_instruction)
            
            if not steps:
                raise ValueError("AI returned no valid steps!")
            
            print(f"\n📋 AI Plan: {len(steps)} steps")
            for i, (images, prompt) in enumerate(steps, 1):
                print(f"   Step {i}: {images} → {prompt[:50]}...")
            
            state = {"steps": steps, "index": 0, "directory": directory_path}
            _UNALIVER_STATE[key] = state
        
        # Get current step
        steps = state["steps"]
        idx = state["index"]
        total = len(steps)
        
        # Check if done
        if idx >= total:
            if loop:
                idx = 0
            else:
                idx = total - 1
                image_names, prompt = steps[idx]
                images = self._load_step_images(directory_path, image_names)
                state["index"] = idx
                return (*images, prompt, idx, total, 1)
        
        # Load current step
        image_names, prompt = steps[idx]
        images = self._load_step_images(directory_path, image_names)
        
        # Advance index
        next_idx = idx + 1
        if next_idx >= total:
            next_idx = 0 if loop else total
        state["index"] = next_idx
        
        done_flag = 0 if (loop or next_idx < total) else 1
        print(f"\n▶️  Step {idx+1}/{total}: {image_names} → {prompt[:50]}...")
        
        return (*images, prompt, idx, total, done_flag)
    
    def _get_ai_plan(self, directory_path, api_key, system_prompt, instruction):
        api_key = self.api_key_hardcoded if not api_key else api_key
        self._standardize_files(directory_path)
        image_files = self._get_sorted_images(directory_path)
        
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://antigravity.ai",
            "X-Title": "Unaliver Step Iterator",
        }
        
        messages = [{"role": "system", "content": system_prompt}]
        content_list = []
        for img_file in image_files:
            img_path = os.path.join(directory_path, img_file)
            base64_str = self._encode_image_thumbnail(img_path)
            content_list.append({"type": "text", "text": f"Filename: {img_file}"})
            content_list.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_str}"}})
        
        content_list.append({"type": "text", "text": f"Instruction: {instruction}"})
        messages.append({"role": "user", "content": content_list})
        
        data = {"model": "qwen/qwen2.5-vl-72b-instruct", "messages": messages}
        
        try:
            print(f"🌐 Calling OpenRouter with {len(image_files)} images...")
            response = requests.post(url, headers=headers, data=json.dumps(data))
            response.raise_for_status()
            ai_response = response.json()['choices'][0]['message']['content']
            print(f"✅ AI responded with {len(ai_response)} characters")
        except Exception as e:
            print(f"❌ API Error: {e}")
            raise
        
        return self._parse_steps(ai_response)
    
    def _parse_steps(self, response_text):
        steps = []
        pattern = r'Step\s+\d+:\s*\[([^\]]+)\]\s*(?:→|->)\s*[""](.+?)[""]'
        matches = re.findall(pattern, response_text, re.DOTALL)
        
        for images_str, prompt in matches:
            image_names = re.findall(r'IMG_\d+', images_str)
            if image_names:
                steps.append((image_names[:4], prompt.strip()))
        return steps
    
    def _get_sorted_images(self, path):
        if not os.path.exists(path):
            return []
        files = [f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
        def sort_key(f):
            match = re.search(r'IMG_(\d+)', f)
            return int(match.group(1)) if match else 9999
        return sorted(files, key=sort_key)
    
    def _standardize_files(self, path):
        if not os.path.exists(path):
            return
        existing_files = [f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
        if not existing_files:
            return
        
        def natural_sort_key(filename):
            parts = re.split(r'(\d+)', filename.lower())
            return [int(part) if part.isdigit() else part for part in parts]
        
        existing_files.sort(key=natural_sort_key)
        
        is_standardized = all(os.path.splitext(existing_files[i])[0] == f"IMG_{i+1}" for i in range(len(existing_files)))
        
        if is_standardized:
            print(f"✅ Files already standardized ({len(existing_files)} images)")
            return
        
        print(f"\n🔄 Standardizing {len(existing_files)} files...")
        import uuid
        temp_mapping = {}
        
        for i, filename in enumerate(existing_files):
            ext = os.path.splitext(filename)[1]
            temp_name = f"RENAME_{uuid.uuid4().hex[:8]}{ext}"
            os.rename(os.path.join(path, filename), os.path.join(path, temp_name))
            temp_mapping[temp_name] = (i + 1, ext)
        
        for temp_name, (img_num, ext) in temp_mapping.items():
            os.rename(os.path.join(path, temp_name), os.path.join(path, f"IMG_{img_num}{ext}"))
        
        print(f"✅ Standardization complete!\n")
    
    def _encode_image_thumbnail(self, image_path):
        try:
            import io
            with Image.open(image_path) as img:
                img.thumbnail((512, 512))
                if img.mode in ('RGBA', 'LA'):
                    background = Image.new(img.mode[:-1], img.size, (255, 255, 255))
                    background.paste(img, img.split()[-1])
                    img = background
                img = img.convert("RGB")
                buffered = io.BytesIO()
                img.save(buffered, format="JPEG", quality=85)
                return base64.b64encode(buffered.getvalue()).decode('utf-8')
        except Exception as e:
            print(f"Error encoding thumbnail: {e}")
            return ""
    
    def _load_step_images(self, directory, image_names):
        images = []
        for name in image_names[:4]:
            images.append(self._load_image_robust(directory, name))
        while len(images) < 4:
            images.append(self._create_black_image())
        return images
    
    def _load_image_robust(self, directory, filename):
        path = os.path.join(directory, filename)
        if os.path.exists(path) and os.path.isfile(path):
            return self._load_image(path)
        
        if '.' not in filename:
            for ext in ['.png', '.jpg', '.jpeg', '.webp']:
                test_path = path + ext
                if os.path.exists(test_path):
                    return self._load_image(test_path)
        
        basename = os.path.splitext(filename)[0]
        try:
            for f in os.listdir(directory):
                if os.path.splitext(f)[0] == basename:
                    return self._load_image(os.path.join(directory, f))
        except Exception:
            pass
        
        return self._create_black_image()
    
    def _load_image(self, path):
        try:
            img = Image.open(path)
            img = ImageOps.exif_transpose(img)
            img = img.convert("RGB")
            image = np.array(img).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            return image
        except Exception:
            return self._create_black_image()
    
    def _create_black_image(self):
        img = Image.new('RGB', (512, 512), color='black')
        image = np.array(img).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        return image
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")


    CATEGORY = "Unaliver/Archive"
    
    # ... (rest of methods)

NODE_CLASS_MAPPINGS = {
    "UnaliverStepIterator": UnaliverStepIterator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UnaliverStepIterator": "Unaliver Step Iterator 🎬 (Archived)"
}
