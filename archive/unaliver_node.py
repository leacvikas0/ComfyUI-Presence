import os
import json
import requests
import base64
import re
import torch
import numpy as np
from PIL import Image, ImageOps

class UnaliverNode:
    """
    Unaliver: An autonomous Art Director node that manages assets, maintains conversation history,
    and drives downstream generation using a Multimodal LLM.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory_path": ("STRING", {"default": "C:/Inspiration"}),
                "openrouter_api_key": ("STRING", {"multiline": False}),
                "system_prompt": ("STRING", {
                    "multiline": True, 
                    "default": (
                        "You are 'Unaliver', an avant-garde Art Director with a keen eye for creative fusion.\n\n"
                        "INSTRUCTIONS:\n"
                        "1. View all available images (IMG_1, IMG_2, IMG_3, etc.)\n"
                        "2. Select 1-4 images that would create an interesting blend\n"
                        "3. Create a wild, creative fusion prompt mixing their concepts, styles, or subjects\n\n"
                        "OUTPUT FORMAT (STRICT):\n"
                        "[IMG_X, IMG_Y, IMG_Z] → \"Your detailed creative prompt here\"\n\n"
                        "EXAMPLES:\n"
                        "[IMG_1, IMG_3] → \"Combine the woman's elegant outfit with the ethereal background, creating a dreamlike fashion portrait\"\n"
                        "[IMG_2] → \"Enhance this portrait with cinematic lighting and dramatic depth of field\"\n"
                        "[IMG_1, IMG_2, IMG_4] → \"Merge the family tableau with surreal elements, blending love and tradition\""
                    )
                }),
                "instruction": ("STRING", {"multiline": True, "default": "Find two images to merge."}),
                "force_generate": ("BOOLEAN", {"default": True}),
                "reset_chat": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("Image_1", "Image_2", "Image_3", "Image_4", "Prompt_Output")
    FUNCTION = "execute_director"
    CATEGORY = "Unaliver/Archive"
    OUTPUT_NODE = False  # Allow ComfyUI caching when force_generate is False

    def execute_director(self, directory_path, openrouter_api_key, system_prompt, instruction, force_generate, reset_chat):
        # BYPASS CACHE: Add random value when force_generate is True
        if force_generate:
            import time
            cache_buster = time.time()
            print(f"🔥 Force generate enabled - bypassing cache ({cache_buster})")
        
        # HARDCODED API KEY FOR LOCAL TESTING
        HARDCODED_KEY = "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"
        api_key = HARDCODED_KEY if not openrouter_api_key else openrouter_api_key

        # 1. Standardization (The Cleaner) - Auto-checks if needed
        # This now intelligently checks if files need renaming
        self.standardize_files(directory_path)

        # 2. State Management (The Memory)
        history_file = os.path.join(directory_path, "skull_history.json")
        history = self.manage_history(history_file, system_prompt, instruction, reset_chat)

        # 3. LLM Interaction
        image_files = self.get_sorted_images(directory_path)
        response_text = self.call_llm(api_key, history, image_files, directory_path)
        
        # CONSOLE OUTPUT - Show what AI said
        print("\n" + "="*60)
        print("🎨 UNALIVER AI RESPONSE:")
        print("="*60)
        print(response_text)
        print("="*60 + "\n")

        # Update history with assistant response
        history.append({"role": "assistant", "content": response_text})
        with open(history_file, "w") as f:
            json.dump(history, f, indent=2)

        # 4. Output Parsing (The Protocol)
        image_names, prompt = self.parse_response(response_text)

        # 5. Load ONLY the images mentioned by AI (up to 4)
        images = []
        print(f"📸 AI mentioned {len(image_names)} images: {image_names}")
        
        for name in image_names[:4]:  # Max 4 images
            images.append(self.load_image_robust(directory_path, name))
        
        # Pad with black images if fewer than 4
        while len(images) < 4:
            images.append(self.create_black_image())
        
        print(f"✅ Outputting {len(image_names[:4])} real images + {4-len(image_names[:4])} black images")
        return (images[0], images[1], images[2], images[3], prompt)

    def get_sorted_images(self, path):
        if not os.path.exists(path):
            return []
        files = [f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
        def sort_key(f):
            match = re.search(r'IMG_(\d+)', f)
            return int(match.group(1)) if match else 9999
        return sorted(files, key=sort_key)

    def standardize_files(self, path):
        """Smart file standardization with DETERMINISTIC alphabetical sorting"""
        if not os.path.exists(path):
            print("⚠️  Directory doesn't exist, skipping standardization")
            return
            
        existing_files = [f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
        if not existing_files:
            print("⚠️  No image files found, skipping standardization")
            return
        
        # DETERMINISTIC SORT: Alphabetical (case-insensitive) with natural number sorting
        # This ensures the SAME ORDER every time, regardless of OS or file system
        import re
        def natural_sort_key(filename):
            """Sort files naturally: file1, file2, file10 (not file1, file10, file2)"""
            parts = re.split(r'(\d+)', filename.lower())
            return [int(part) if part.isdigit() else part for part in parts]
        
        existing_files.sort(key=natural_sort_key)
        
        # CHECK IF ALREADY STANDARDIZED
        is_standardized = True
        for i, filename in enumerate(existing_files):
            basename = os.path.splitext(filename)[0]
            expected_name = f"IMG_{i+1}"
            if basename != expected_name:
                is_standardized = False
                break
        
        if is_standardized:
            print(f"✅ Files already standardized ({len(existing_files)} images)")
            return
        
        print(f"\n🔄 Standardizing {len(existing_files)} files...")
        print("📋 MAPPING (alphabetical order):")
        
        # SAFE RENAME: Use UUID temp names to avoid any collisions
        import uuid
        temp_mapping = {}
        
        # Step 1: Show mapping and rename all to unique temp names
        for i, filename in enumerate(existing_files):
            ext = os.path.splitext(filename)[1]
            new_name = f"IMG_{i+1}{ext}"
            print(f"   {filename} → {new_name}")
            
            temp_name = f"RENAME_{uuid.uuid4().hex[:8]}{ext}"
            old_path = os.path.join(path, filename)
            temp_path = os.path.join(path, temp_name)
            os.rename(old_path, temp_path)
            temp_mapping[temp_name] = (i + 1, ext)
        
        # Step 2: Rename temp files to final IMG_X names
        for temp_name, (img_num, ext) in temp_mapping.items():
            temp_path = os.path.join(path, temp_name)
            final_name = f"IMG_{img_num}{ext}"
            final_path = os.path.join(path, final_name)
            os.rename(temp_path, final_path)
        
        print(f"✅ Standardization complete!\n")

    def manage_history(self, history_path, system_prompt, instruction, reset):
        if reset or not os.path.exists(history_path):
            history = [{"role": "system", "content": system_prompt}]
            print("💀 History reset with new system prompt")
        else:
            try:
                with open(history_path, "r") as f:
                    history = json.load(f)
                    
                # AUTO-RESET: If system prompt changed, start NEW CHAT
                if history and history[0].get("role") == "system":
                    if history[0]["content"] != system_prompt:
                        print("🔄 System prompt changed - STARTING NEW CHAT")
                        history = [{"role": "system", "content": system_prompt}]
                        
            except (json.JSONDecodeError, IOError):
                history = [{"role": "system", "content": system_prompt}]
                print("⚠️  Corrupted history - created new one")
        
        history.append({"role": "user", "content": instruction})
        return history

    def call_llm(self, api_key, history, image_files, directory_path):
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://antigravity.ai",
            "X-Title": "Unaliver Node",
        }

        messages = []
        for msg in history[:-1]:
            messages.append(msg)
            
        last_user_msg = history[-1]
        content_list = []
        
        for img_file in image_files:
            img_path = os.path.join(directory_path, img_file)
            base64_str = self.encode_image_thumbnail(img_path)
            content_list.append({"type": "text", "text": f"Filename: {img_file}"})
            content_list.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_str}"}})
            
        content_list.append({"type": "text", "text": f"Instruction: {last_user_msg['content']}"})
        messages.append({"role": "user", "content": content_list})

        data = {
            "model": "qwen/qwen2.5-vl-72b-instruct",
            "messages": messages
        }

        try:
            print(f"🌐 Calling OpenRouter API with {len(image_files)} images...")
            response = requests.post(url, headers=headers, data=json.dumps(data))
            response.raise_for_status()
            result = response.json()
            
            # Debug logging
            print(f"✅ API Response Status: {response.status_code}")
            if 'choices' in result and len(result['choices']) > 0:
                ai_response = result['choices'][0]['message']['content']
                print(f"📝 AI returned {len(ai_response)} characters")
                return ai_response
            else:
                error_msg = f"Empty response from API. Full result: {result}"
                print(f"❌ {error_msg}")
                return f"[IMG_1] → \"Error: API returned empty response\""
                
        except requests.exceptions.HTTPError as e:
            error_detail = e.response.text if hasattr(e, 'response') else str(e)
            print(f"❌ HTTP Error {e.response.status_code}: {error_detail}")
            return f"[IMG_1] → \"HTTP Error: {error_detail[:100]}\""
        except Exception as e:
            print(f"❌ LLM Error: {type(e).__name__}: {str(e)}")
            return f"[IMG_1] → \"Error calling LLM: {str(e)}\""

    def encode_image_thumbnail(self, image_path):
        try:
            with Image.open(image_path) as img:
                img.thumbnail((512, 512))
                if img.mode in ('RGBA', 'LA'):
                    background = Image.new(img.mode[:-1], img.size, (255, 255, 255))
                    background.paste(img, img.split()[-1])
                    img = background
                img = img.convert("RGB")
                import io
                buffered = io.BytesIO()
                img.save(buffered, format="JPEG", quality=85)
                return base64.b64encode(buffered.getvalue()).decode('utf-8')
        except Exception as e:
            print(f"Error encoding thumbnail for {image_path}: {e}")
            return ""

    def parse_response(self, response_text):
        # ARROW FORMAT: [IMG_1, IMG_2, IMG_4] → "Prompt text here"
        print(f"\n🔍 Parsing AI response...")
        
        # Try arrow format first (supports → or ->)
        arrow_pattern = r'\[([^\]]+)\]\s*(?:→|->)\s*[""](.+?)[""]'
        match = re.search(arrow_pattern, response_text, re.DOTALL)
        
        if match:
            images_str = match.group(1)
            prompt = match.group(2)
            image_matches = re.findall(r'IMG_\d+', images_str)
            unique_images = []
            for img in image_matches:
                if img not in unique_images:
                    unique_images.append(img)
            print(f"✅ Arrow format parsed: {unique_images} → {prompt[:50]}...")
            return unique_images[:4], prompt
        
        # Fallback: find any IMG_X mentions
        image_matches = re.findall(r'IMG_\d+', response_text)
        unique_images = []
        for img in image_matches:
            if img not in unique_images:
                unique_images.append(img)
        
        prompt_match = re.search(r'[""](.+?)[""]', response_text, re.DOTALL)
        prompt = prompt_match.group(1) if prompt_match else "Creative fusion of selected images"
        
        print(f"⚠️  Fallback parsing: found {unique_images}")
        return unique_images[:4] if unique_images else ["IMG_1"], prompt

    def load_image_robust(self, directory, filename):
        print(f"🔍 Trying to load: {filename} from {directory}")
        
        # Check exact match first
        path = os.path.join(directory, filename)
        print(f"   Checking exact path: {path}")
        if os.path.exists(path) and os.path.isfile(path):
            print(f"   ✅ Found exact match!")
            return self.load_image(path)
        
        # Check if filename has no extension, try adding common ones
        if '.' not in filename:
            print(f"   No extension detected, trying common extensions...")
            for ext in ['.png', '.jpg', '.jpeg', '.webp']:
                test_path = path + ext
                print(f"      Trying: {test_path}")
                if os.path.exists(test_path):
                    print(f"      ✅ Found with extension {ext}!")
                    return self.load_image(test_path)
        
        # If still not found, try to find by basename in the directory
        print(f"   Searching directory for basename match...")
        basename = os.path.splitext(filename)[0]
        try:
            for f in os.listdir(directory):
                if os.path.splitext(f)[0] == basename:
                    found_path = os.path.join(directory, f)
                    print(f"   ✅ Found basename match: {f}")
                    return self.load_image(found_path)
        except Exception as e:
            print(f"   ❌ Error listing directory: {e}")

        print(f"   ⚠️  Image not found, returning black image")
        return self.create_black_image()

    def load_image(self, path):
        try:
            img = Image.open(path)
            img = ImageOps.exif_transpose(img)
            img = img.convert("RGB")
            image = np.array(img).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            return image
        except Exception:
            return self.create_black_image()

    def create_black_image(self):
        img = Image.new('RGB', (512, 512), color='black')
        image = np.array(img).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        return image

# Mappings
    CATEGORY = "Unaliver/Archive"

    # ... (rest of methods)

# Mappings
NODE_CLASS_MAPPINGS = {
    "UnaliverNode": UnaliverNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UnaliverNode": "Unaliver (Archived)"
}
