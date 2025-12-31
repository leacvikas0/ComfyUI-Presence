# =============================================================================
# ⚡ PRESENCE DIRECTOR - Gemini 3 Flash
# =============================================================================
# AI-directed image generation using Gemini 3 Flash
# Simplified JSON format, streaming with visible thinking
# =============================================================================

import os
import json
import io
import torch
import numpy as np
from PIL import Image, ImageOps
from datetime import datetime

# Google GenAI SDK
try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    print("[Director] google-genai not installed. Run: pip install google-genai>=1.51.0")

# Global state
DIRECTOR_STATE = {}


class PresenceDirector:
    """
    ⚡ PRESENCE DIRECTOR (Gemini 3 Flash)
    
    Simplified flow:
    - Queue empty? -> Call AI (streaming with [THINKING] and [RESPONSE])
    - Queue has jobs? -> Execute one job, remove from queue
    
    AI outputs simple JSON:
    {
        "queue": [
            {"output_name": "...", "prompt": "...", "load": ["file.jpg"], ...}
        ],
        "done": true  // only when project complete
    }
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "active_folder": ("STRING", {"default": "C:/Presence/Job_01"}),
                "api_key": ("STRING", {"default": ""}),
                "system_prompt": ("STRING", {"multiline": True, "default": "You are a Presence Director..."}),
                "user_input": ("STRING", {"multiline": True, "default": "", "placeholder": "Optional message to AI..."}),
                "reset": ("BOOLEAN", {"default": False, "label_on": "RESET ALL", "label_off": "Continue"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "INT", "INT", "STRING")
    RETURN_NAMES = ("images", "prompt", "width", "height", "filename")
    FUNCTION = "run"
    CATEGORY = "PresenceAI"

    def run(self, active_folder, api_key, system_prompt, user_input, reset, seed):
        """Main execution loop"""
        
        if not GENAI_AVAILABLE:
            print("[Director] ERROR: google-genai not available!")
            return self._empty_output()
        
        # Ensure folder exists
        os.makedirs(active_folder, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"⚡ PRESENCE DIRECTOR")
        print(f"{'='*60}")
        print(f"   Folder: {active_folder}")
        
        global DIRECTOR_STATE
        state_file = os.path.join(active_folder, "director_state.json")
        
        # Handle reset
        if reset:
            print(f"   [RESET] Clearing all state...")
            DIRECTOR_STATE[active_folder] = {
                "chat_history": [],
                "seen_files": set(),
                "queue": [],
                "last_input": ""
            }
            if os.path.exists(state_file):
                os.remove(state_file)
            print(f"   [RESET] Done. All files will be treated as new.")
        
        # Initialize state
        if active_folder not in DIRECTOR_STATE:
            DIRECTOR_STATE[active_folder] = {
                "chat_history": [],
                "seen_files": set(),
                "queue": [],
                "last_input": ""
            }
            # Load from disk
            if os.path.exists(state_file):
                try:
                    with open(state_file, "r") as f:
                        disk = json.load(f)
                        DIRECTOR_STATE[active_folder]["seen_files"] = set(disk.get("seen_files", []))
                        DIRECTOR_STATE[active_folder]["queue"] = disk.get("queue", [])
                except:
                    pass
        
        state = DIRECTOR_STATE[active_folder]
        print(f"   Queue: {len(state['queue'])} jobs")
        print(f"   Seen: {len(state['seen_files'])} files")
        
        # Execute from queue or call AI
        if len(state["queue"]) > 0:
            job = state["queue"].pop(0)
            self._save_state(active_folder, state)
            return self._execute_job(active_folder, job)
        else:
            return self._call_ai(active_folder, state, api_key, system_prompt, user_input)

    def _call_ai(self, folder, state, api_key, system_prompt, user_input):
        """Call Gemini 3 Flash with streaming"""
        
        try:
            client = genai.Client(api_key=api_key)
            
            # Scan for images
            all_files = [f for f in os.listdir(folder) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
            current_files = set(all_files)
            new_files = current_files - state["seen_files"]
            
            print(f"   Files: {len(current_files)} total, {len(new_files)} new")
            
            # Check if we have work to do
            input_changed = user_input and user_input != state.get("last_input", "")
            
            if len(new_files) == 0 and not input_changed:
                print(f"   [IDLE] No new files or input.")
                return self._empty_output()
            
            # Build message
            file_list = "\n".join([f"- {f}" for f in sorted(current_files)])
            new_list = ", ".join(sorted(new_files)) if new_files else "(none)"
            
            message = f"""
**NEW FILES:** {new_list}

**ALL FILES IN FOLDER:**
{file_list}

Analyze and respond with your JSON plan.
"""
            
            if user_input and input_changed:
                message = f"**USER MESSAGE:** {user_input}\n\n" + message
                state["last_input"] = user_input
            
            # Prepare images for AI (resize to 2MP max)
            upload_parts = []
            for f in new_files:
                path = os.path.join(folder, f)
                try:
                    img = Image.open(path)
                    img = ImageOps.exif_transpose(img)
                    
                    # Resize if > 2MP
                    pixels = img.width * img.height
                    if pixels > 2 * 1024 * 1024:
                        scale = (2 * 1024 * 1024 / pixels) ** 0.5
                        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
                    
                    buffered = io.BytesIO()
                    img.save(buffered, format="PNG")
                    upload_parts.append(types.Part.from_bytes(data=buffered.getvalue(), mime_type="image/png"))
                    upload_parts.append(types.Part.from_text(text=f"[Image: {f}]"))
                except Exception as e:
                    print(f"   Error loading {f}: {e}")
            
            # Build content
            content_parts = [types.Part.from_text(text=message)] + upload_parts
            
            # Build messages with history
            messages = []
            for msg in state["chat_history"]:
                messages.append(types.Content(role=msg["role"], parts=[types.Part.from_text(text=msg["content"])]))
            messages.append(types.Content(role="user", parts=content_parts))
            
            print(f"\n   Calling Gemini 3 Flash (streaming)...")
            
            # Stream response
            stream = client.models.generate_content_stream(
                model="gemini-3-flash-preview",
                contents=messages,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.7,
                    max_output_tokens=16384,
                    thinking_config=types.ThinkingConfig(
                        thinking_budget=8192,
                        include_thoughts=True
                    )
                )
            )
            
            thinking_text = ""
            response_text = ""
            current_mode = None
            
            print(f"\n{'='*60}")
            
            for chunk in stream:
                if hasattr(chunk, 'candidates') and chunk.candidates:
                    for candidate in chunk.candidates:
                        if hasattr(candidate, 'content') and candidate.content:
                            for part in candidate.content.parts:
                                if part.text:
                                    if part.thought:
                                        if current_mode != "thinking":
                                            current_mode = "thinking"
                                            print("\n[THINKING] ", end="", flush=True)
                                        print(part.text, end="", flush=True)
                                        thinking_text += part.text
                                    else:
                                        if current_mode != "response":
                                            current_mode = "response"
                                            print("\n\n[RESPONSE] ", end="", flush=True)
                                        print(part.text, end="", flush=True)
                                        response_text += part.text
            
            print(f"\n{'='*60}\n")
            
            # Update history
            state["chat_history"].append({"role": "user", "content": message})
            state["chat_history"].append({"role": "model", "content": response_text})
            
            # Mark files as seen
            for f in new_files:
                state["seen_files"].add(f)
            
            # Parse JSON
            data = self._parse_json(response_text)
            
            # Handle done
            if data.get("done"):
                print("[DONE] Project complete!")
                state["chat_history"] = []
                state["seen_files"] = set()
                state["queue"] = []
                self._save_state(folder, state)
                return self._empty_output()
            
            # Add jobs to queue
            if "queue" in data:
                state["queue"].extend(data["queue"])
                print(f"   Added {len(data['queue'])} jobs to queue")
            
            self._save_state(folder, state)
            
            # Execute first job if available
            if len(state["queue"]) > 0:
                job = state["queue"].pop(0)
                self._save_state(folder, state)
                return self._execute_job(folder, job)
            
            return self._empty_output()
            
        except Exception as e:
            print(f"[ERROR] {e}")
            import traceback
            traceback.print_exc()
            return self._empty_output()

    def _execute_job(self, folder, job):
        """Execute a job from the queue"""
        
        output_name = job.get("output_name", "output")
        prompt = job.get("prompt", "")
        load_list = job.get("load", [])
        quality = job.get("quality", "normal")  # "high" = 2MP, "normal" = 1MP
        w = job.get("w", 1920)
        h = job.get("h", 1080)
        pad = job.get("pad", None)  # e.g. "left 20%"
        
        print(f"\n[EXECUTING] {output_name}")
        print(f"   Prompt: {prompt[:80]}...")
        print(f"   Load: {load_list}")
        print(f"   Quality: {quality}, Size: {w}x{h}")
        if pad:
            print(f"   Pad: {pad}")
        # Always cap at 2MP, but don't upscale smaller images
        images = []
        
        for filename in load_list:
            path = os.path.join(folder, filename)
            if not os.path.exists(path):
                print(f"   [ERROR] File not found: {filename}")
                continue
            
            try:
                img = Image.open(path).convert("RGB")
                img = ImageOps.exif_transpose(img)
                print(f"   Loaded: {filename} ({img.width}x{img.height})")
                
                # Apply padding if specified
                if pad:
                    img = self._apply_pad(img, pad)
                
                # Only downscale if above 2MP, never upscale
                pixels = img.width * img.height
                max_pixels = 2 * 1024 * 1024  # 2MP max
                if pixels > max_pixels:
                    scale = (max_pixels / pixels) ** 0.5
                    new_w = int(img.width * scale)
                    new_h = int(img.height * scale)
                    # Ensure even dimensions
                    new_w = new_w - (new_w % 2)
                    new_h = new_h - (new_h % 2)
                    img = img.resize((new_w, new_h), Image.LANCZOS)
                    print(f"   Downscaled to 2MP: {new_w}x{new_h}")
                else:
                    print(f"   Keeping original size ({pixels/1024/1024:.1f}MP)")
                
                # Convert to tensor (NHWC)
                arr = np.array(img).astype(np.float32) / 255.0
                tensor = torch.from_numpy(arr)[None,]
                images.append(tensor)
                
            except Exception as e:
                print(f"   [ERROR] Loading {filename}: {e}")
        
        if len(images) == 0:
            print(f"   [ERROR] No images loaded")
            return self._empty_output()
        
        # Stack images
        if len(images) == 1:
            output_images = images[0]
        else:
            output_images = torch.cat(images, dim=0)
        
        print(f"   Output: {output_images.shape}")
        return (output_images, prompt, w, h, output_name)

    def _apply_pad(self, img, pad_spec):
        """Apply padding: 'left 20%' or 'right 30%' etc."""
        
        parts = pad_spec.lower().split()
        if len(parts) < 2:
            return img
        
        side = parts[0]
        amount = parts[1]
        
        # Parse percentage
        if "%" in amount:
            percent = float(amount.replace("%", ""))
            if side in ["left", "right"]:
                pad_px = int(img.width * percent / 100)
            else:
                pad_px = int(img.height * percent / 100)
        else:
            pad_px = int(amount)
        
        print(f"   Padding: {side} +{pad_px}px")
        
        # Create solid gray canvas
        if side == "left":
            new_w = img.width + pad_px
            canvas = self._gray_canvas(new_w, img.height)
            canvas.paste(img, (pad_px, 0))
        elif side == "right":
            new_w = img.width + pad_px
            canvas = self._gray_canvas(new_w, img.height)
            canvas.paste(img, (0, 0))
        elif side == "top":
            new_h = img.height + pad_px
            canvas = self._gray_canvas(img.width, new_h)
            canvas.paste(img, (0, pad_px))
        elif side == "bottom":
            new_h = img.height + pad_px
            canvas = self._gray_canvas(img.width, new_h)
            canvas.paste(img, (0, 0))
        else:
            return img
        
        img = canvas
        
        # Adjust to 16:9
        img = self._adjust_16_9(img)
        
        # Resize to 2MP (padding always outputs high quality)
        pixels = img.width * img.height
        target = 2 * 1024 * 1024
        if abs(pixels - target) > 10000:
            scale = (target / pixels) ** 0.5
            new_w = int(img.width * scale) - (int(img.width * scale) % 2)
            new_h = int(img.height * scale) - (int(img.height * scale) % 2)
            img = img.resize((new_w, new_h), Image.LANCZOS)
        
        print(f"   After pad+16:9+2MP: {img.width}x{img.height}")
        return img

    def _gray_canvas(self, w, h):
        """Create solid gray canvas (RGB 128, 128, 128)"""
        canvas = np.full((h, w, 3), 128, dtype=np.uint8)
        return Image.fromarray(canvas)

    def _adjust_16_9(self, img):
        """Adjust image to 16:9 by adding gray padding"""
        target = 16 / 9
        current = img.width / img.height
        
        if abs(current - target) < 0.01:
            return img
        
        if current < target:
            # Too tall, add width
            new_w = int(img.height * target)
            pad = new_w - img.width
            pad_l = pad // 2
            canvas = self._gray_canvas(new_w, img.height)
            canvas.paste(img, (pad_l, 0))
            return canvas
        else:
            # Too wide, add height
            new_h = int(img.width / target)
            pad = new_h - img.height
            pad_t = pad // 2
            canvas = self._gray_canvas(img.width, new_h)
            canvas.paste(img, (0, pad_t))
            return canvas

    def _parse_json(self, text):
        """Extract JSON from response"""
        # Try direct
        try:
            return json.loads(text.strip())
        except:
            pass
        
        # Try code block
        if "```json" in text:
            try:
                start = text.find("```json") + 7
                end = text.find("```", start)
                return json.loads(text[start:end].strip())
            except:
                pass
        
        # Try finding { }
        brace = text.find("{")
        if brace != -1:
            depth = 0
            for i, c in enumerate(text[brace:]):
                if c == "{": depth += 1
                elif c == "}": depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[brace:brace+i+1])
                    except:
                        break
        
        return {}

    def _save_state(self, folder, state):
        """Save state to disk"""
        path = os.path.join(folder, "director_state.json")
        with open(path, "w") as f:
            json.dump({
                "seen_files": list(state["seen_files"]),
                "queue": state["queue"]
            }, f, indent=2)

    def _empty_output(self):
        """Return empty output when nothing to do"""
        empty = torch.zeros((1, 64, 64, 3))
        return (empty, "", 1920, 1080, "")


# Register
NODE_CLASS_MAPPINGS = {
    "PresenceDirector": PresenceDirector
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PresenceDirector": "⚡ Presence Director"
}
