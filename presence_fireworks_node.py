import os
import time
import json
import shutil
import io
import requests
import base64
import torch
import numpy as np
from PIL import Image, ImageOps

# --------------------------------------------------------------------------------
# GLOBAL STATE MANAGEMENT
# Key: active_folder_path
# Value: {
#   "chat_history": [],
#   "seen_files": set(),
#   "queue": []
# }
# --------------------------------------------------------------------------------
NODE_STATE = {}

class PresenceDirectorFireworks:
    """
    🔥 PRESENCE DIRECTOR (Fireworks AI Edition - Qwen3-VL)
    - Uses Fireworks AI HTTP API with Qwen3-VL-235B-A22B-Thinking
    - 10-20x cheaper than Gemini 3 Pro
    - Switches between 'Brain Mode' (Analysis) and 'Robot Mode' (Queue Execution)
    - Manages File System, Context, and Flux Generation
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "active_folder": ("STRING", {"default": "C:/Presence/Job_01"}),
                "api_key": ("STRING", {"default": "fw_..."}),
                "system_prompt": ("STRING", {"multiline": True, "default": "PASTE SYSTEM PROMPT HERE"}),
                "user_input": ("STRING", {"multiline": True, "default": "", "placeholder": "Type intervention here (sent once)..."}),
                "reset_history": ("BOOLEAN", {"default": False, "label_on": "RESET ON NEXT RUN", "label_off": "Keep History"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("UNALIVER_BUNDLE", "STRING", "INT", "INT", "INT", "STRING", "STRING")
    RETURN_NAMES = ("context_bundle", "flux_prompt", "width", "height", "batch_size", "status", "filename")
    FUNCTION = "run_director"
    CATEGORY = "PresenceAI"

    def run_director(self, active_folder, api_key, system_prompt, user_input, reset_history, seed):
        """Main execution - switches between Brain and Robot modes"""
        
        print(f"\n{'='*80}")
        print(f"🔥 PRESENCE DIRECTOR (Fireworks AI - Qwen3-VL)")
        print(f"{'='*80}")
        print(f"   📂 Active folder: {active_folder}")
        print(f"   🔄 Reset requested: {reset_history}")
        
        # Ensure folder exists
        if not os.path.exists(active_folder):
            os.makedirs(active_folder, exist_ok=True)
            print(f"   📁 Created folder: {active_folder}")
        
        global NODE_STATE
        state_file = os.path.join(active_folder, "presence_state.json")
        
        # =========================================================
        # ROBUST RESET: Complete nuclear option
        # =========================================================
        if reset_history:
            print(f"\n{'='*60}")
            print(f"🔄 RESET TRIGGERED - NUCLEAR OPTION")
            print(f"{'='*60}")
            
            # Step 1: Remove from global state entirely
            if active_folder in NODE_STATE:
                old_state = NODE_STATE[active_folder]
                print(f"   📊 BEFORE RESET:")
                print(f"      - seen_files: {len(old_state.get('seen_files', set()))} files")
                print(f"      - queue: {len(old_state.get('queue', []))} jobs")
                print(f"      - chat_history: {len(old_state.get('chat_history', []))} messages")
                del NODE_STATE[active_folder]
                print(f"   🗑️ Removed from memory (NODE_STATE)")
            else:
                print(f"   ℹ️ Not in memory (first run or already cleared)")
            
            # Step 2: Delete disk state file
            if os.path.exists(state_file):
                try:
                    os.remove(state_file)
                    print(f"   🗑️ Deleted disk state: {state_file}")
                except Exception as e:
                    print(f"   ⚠️ Could not delete disk state: {e}")
            else:
                print(f"   ℹ️ No disk state file to delete")
            
            # Step 3: Create completely fresh state
            NODE_STATE[active_folder] = {
                "chat_history": [],
                "seen_files": set(),
                "queue": [],
                "last_input": ""
            }
            
            print(f"   ✨ Created fresh state")
            print(f"   📊 AFTER RESET:")
            print(f"      - seen_files: 0 files")
            print(f"      - queue: 0 jobs")
            print(f"      - chat_history: 0 messages")
            print(f"{'='*60}")
            print(f"✅ RESET COMPLETE - All files will be treated as NEW\n")
        
        # =========================================================
        # NORMAL INIT: Load or create state
        # =========================================================
        else:
            # Initialize if not exists
            if active_folder not in NODE_STATE:
                NODE_STATE[active_folder] = {
                    "chat_history": [],
                    "seen_files": set(),
                    "queue": [],
                    "last_input": ""
                }
                print(f"   ✨ Initialized new state for folder")
            
            # Load persistent state from disk
            if os.path.exists(state_file):
                try:
                    with open(state_file, "r") as f:
                        disk_state = json.load(f)
                        NODE_STATE[active_folder]["seen_files"] = set(disk_state.get("seen_files", []))
                        NODE_STATE[active_folder]["queue"] = disk_state.get("queue", [])
                        print(f"   💾 Loaded state: {len(NODE_STATE[active_folder]['seen_files'])} seen files, {len(NODE_STATE[active_folder]['queue'])} queued jobs")
                except Exception as e:
                    print(f"   ⚠️ Could not load state file: {e}")
        
        # Get current state reference
        state = NODE_STATE[active_folder]
        
        # Mode decision
        if len(state["queue"]) > 0:
            print(f"\n🤖 ROBOT MODE: {len(state['queue'])} jobs in queue...")
            print(f"   ⚡ Executing first job...")
            job = state["queue"].pop(0)
            self._save_state(active_folder, state)
            return self._execute_job(active_folder, job)
        else:
            print(f"\n🧠 BRAIN MODE: Analyzing folder state...")
            return self._brain_mode(active_folder, state, api_key, system_prompt, user_input)

    def _brain_mode(self, active_folder, state, api_key, system_prompt, user_input):
        """Brain Mode: Analyze new files and plan next actions"""
        
        try:
            # Scan for all image files
            all_files = [f for f in os.listdir(active_folder) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            current_files = set(all_files)
            
            # Detailed logging
            print(f"   📂 Folder: {active_folder}")
            print(f"   📁 Total images in folder: {len(current_files)}")
            print(f"   👁️ Already seen: {len(state['seen_files'])} files")
            
            # Find new files
            new_files = current_files - state["seen_files"]
            
            print(f"   🆕 New files to process: {len(new_files)}")
            if len(new_files) > 0:
                for f in sorted(new_files):
                    print(f"      → {f}")
            
            # Check for user input change
            input_changed = user_input != state.get("last_input", "")
            if user_input:
                print(f"   💬 User input: \"{user_input[:50]}...\"" if len(user_input) > 50 else f"   💬 User input: \"{user_input}\"")
                print(f"   🔄 Input changed: {input_changed}")
            
            if len(new_files) == 0 and not input_changed:
                print(f"\n   💤 IDLE: No new files, no new input.")
                print(f"      Tip: Toggle 'reset_history' to force re-analysis of all files.")
                return ([], "", 1024, 1024, 1, "IDLE", "")
            
            # Build file manifest
            file_list_text = "**CURRENT FILES IN FOLDER:**\n"
            sorted_files = sorted(list(current_files))
            for f in sorted_files:
                file_list_text += f"- {f}\n"
            
            upload_images = []
            
            # Send ORIGINAL resolution to Qwen (cheap, no resizing)
            print(f"   - Found {len(new_files)} new images to upload (ORIGINAL resolution).")
            for f in new_files:
                path = os.path.join(active_folder, f)
                try:
                    img = Image.open(path)
                    img = ImageOps.exif_transpose(img)
                    print(f"   📤 Uploading {f}: {img.width}x{img.height} (original)")
                    upload_images.append((f, img))
                except Exception as e:
                    print(f"   ❌ Error reading {f}: {e}")
            
            # Build message for Fireworks API
            base_instruction = f"""
**NEW FILES DETECTED:**
{', '.join(new_files) if new_files else '(none)'}

{file_list_text}
"""
            
            if user_input and user_input != state.get("last_input", ""):
                base_instruction += f"\n**USER INPUT:**\n{user_input}\n\n"
                state["last_input"] = user_input
            
            base_instruction += "\nAnalyze the current state and respond with your JSON plan."
            
            # Build messages array for Fireworks
            message_content = [{"type": "text", "text": base_instruction}]
            
            # Add images as base64
            for filename, img in upload_images:
                # Convert to base64
                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                
                message_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_base64}"}
                })
                
                # Add caption
                message_content.append({
                    "type": "text",
                    "text": f"☝️ {filename}"
                })
            
            # Prepare Fireworks API request
            messages = state["chat_history"] + [{
                "role": "user",
                "content": message_content
            }]
            
            # Add system prompt at the beginning if chat history is empty
            if len(state["chat_history"]) == 0:
                messages = [
                    {"role": "system", "content": system_prompt}
                ] + messages
            
            payload = {
                "model": "accounts/fireworks/models/qwen3-vl-235b-a22b-thinking",
                "max_tokens": 150000,  # Very high - model can think as long as needed
                "temperature": 0.6,
                "messages": messages
            }
            
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            
            print(f"   🔥 Calling Fireworks AI (Qwen3-VL)...")
            print(f"      Model: {payload['model']}")
            print(f"      Max tokens: {payload['max_tokens']}")
            
            response = requests.post(
                "https://api.fireworks.ai/inference/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=180  # Increased timeout for large responses
            )
            
            response.raise_for_status()
            result = response.json()
            
            # Log token usage
            if "usage" in result:
                usage = result["usage"]
                input_tokens = usage.get("prompt_tokens", 0)
                output_tokens = usage.get("completion_tokens", 0)
                total_tokens = usage.get("total_tokens", 0)
                
                # Cost calculation (Qwen3-VL pricing)
                cost = (input_tokens * 0.22 / 1_000_000) + (output_tokens * 0.88 / 1_000_000)
                
                print(f"\n   📊 TOKEN USAGE:")
                print(f"      Input:  {input_tokens:,} tokens")
                print(f"      Output: {output_tokens:,} tokens")
                print(f"      Total:  {total_tokens:,} tokens")
                print(f"      Cost:   ${cost:.4f}")
            
            # Extract response
            response_text = result["choices"][0]["message"]["content"]
            
            # Diagnostic: Show raw response info
            print(f"\n   📥 RAW RESPONSE RECEIVED:")
            print(f"      Length: {len(response_text)} chars")
            print(f"      Contains <think>: {'<think>' in response_text}")
            print(f"      Contains </think>: {'</think>' in response_text}")
            print(f"      Starts with {{: {response_text.strip().startswith('{')}")
            
            # Check if response was truncated (hit max_tokens)
            if "usage" in result:
                if result["usage"].get("completion_tokens", 0) >= 16380:
                    print(f"      ⚠️ WARNING: Output hit token limit! Response may be truncated.")
            if "</think>" in response_text:
                thinking_part, final_answer = response_text.split("</think>", 1)
                
                # Clean up thinking part (remove <think> tag if present)
                if "<think>" in thinking_part:
                    thinking_part = thinking_part.split("<think>", 1)[1]
                
                print(f"\n{'='*80}")
                print(f"🧠 AI THINKING PROCESS ({len(thinking_part)} chars):")
                print(f"{'='*80}")
                # Show FULL thinking output
                print(thinking_part.strip())
                print(f"{'='*80}")
                
                response_text = final_answer.strip()
                
                print(f"\n   📤 EXTRACTED JSON RESPONSE:")
                print(f"      Length: {len(response_text)} chars")
                print(f"      Starts with {{: {response_text.strip().startswith('{')}")
            else:
                print(f"\n   ⚠️ NO </think> TAG FOUND IN RESPONSE!")
                print(f"      Possible causes:")
                print(f"      1. Response hit token limit while AI was still thinking")
                print(f"      2. Model didn't use thinking mode")
                print(f"      3. Response was truncated")
                print(f"      Raw response will be treated as final answer.")
            
            print(f"\n{'='*80}")
            print(f"📋 FINAL JSON RESPONSE:")
            print(f"{'='*80}")
            # Limit display to prevent log spam
            if len(response_text) > 2000:
                print(response_text[:2000])
                print(f"... ({len(response_text) - 2000} more chars)")
            else:
                print(response_text)
            print(f"{'='*80}\n")
            
            # Update chat history
            state["chat_history"].append({"role": "user", "content": message_content})
            state["chat_history"].append({"role": "assistant", "content": response_text})
            
            # Mark files as seen
            for f in new_files:
                state["seen_files"].add(f)
            
            # Parse JSON
            data = self._parse_json(response_text)
            
            if "ops" in data:
                for op in data["ops"]:
                    self._handle_file_op(active_folder, op)
            
            if "queue" in data:
                state["queue"].extend(data["queue"])
                print(f"   📋 Added {len(data['queue'])} jobs to queue.")
            
            if data.get("refresh_context"):
                print("   🔄 Refresh context requested - clearing seen files.")
                state["seen_files"] = set()
            
            if data.get("status") == "DONE":
                print("   ✅ Job marked DONE by AI.")
                self._save_done(active_folder, response_text)
                state["chat_history"] = []
                state["seen_files"] = set()
                state["queue"] = []
                self._save_state(active_folder, state)
                return ([], "", 1024, 1024, 1, "DONE", "")
            
            self._save_state(active_folder, state)

            if len(state["queue"]) > 0:
                print("   ⚡ Immediate Trigger: Executing first job...")
                job = state["queue"].pop(0)
                self._save_state(active_folder, state)
                return self._execute_job(active_folder, job)
            else:
                print("   💤 No jobs in queue. Waiting for next auto-queue.")
                return ([], "", 1024, 1024, 1, "WORKING", "")

        except Exception as e:
            print(f"❌ Error in Brain Mode: {e}")
            import traceback
            traceback.print_exc()
            return ([], "", 1024, 1024, 1, "ERROR", "")

    def _execute_job(self, folder, job):
        """Executes a generation job from the queue"""
        prompt = job.get("prompt", "")
        w = job.get("w", 1024)
        h = job.get("h", 1024)
        batch = job.get("batch", 1)
        output_name = job.get("output_name", "gen")
        load_list = job.get("load", [])
        padding_spec = job.get("padding", None)
        
        bundle = []
        print(f"   📦 Bundling: {load_list}")
        
        for item in load_list:
            # Parse item - can be string "file.jpg" or dict {"file": "file.jpg", "mp": 2}
            if isinstance(item, dict):
                filename = item.get("file")
                target_mp = item.get("mp", 1)  # Default 1MP
            else:
                filename = item
                target_mp = 1  # Default 1MP
            
            path = os.path.join(folder, filename)
            if os.path.exists(path):
                try:
                    img = Image.open(path).convert("RGB")
                    img = ImageOps.exif_transpose(img)
                    
                    print(f"   ✅ Loaded {filename}: {img.width}x{img.height}")
                    
                    # Resize to target MP for Flux
                    current_pixels = img.width * img.height
                    target_pixels = target_mp * 1024 * 1024
                    
                    if current_pixels > target_pixels:
                        scale_factor = (target_pixels / current_pixels) ** 0.5
                        new_width = int(img.width * scale_factor)
                        new_height = int(img.height * scale_factor)
                        img = img.resize((new_width, new_height), Image.LANCZOS)
                        print(f"      🔽 Resized to {target_mp}MP: {img.width}x{img.height}")
                    else:
                        print(f"      ✓ Using as-is ({target_mp}MP target, already smaller)")
                    
                    # Apply padding if specified (padding spec applies to ALL loaded images)
                    if padding_spec:
                        img = self._apply_padding(img, padding_spec, filename)
                    
                    # Convert to tensor
                    i = np.array(img).astype(np.float32) / 255.0
                    tensor = torch.from_numpy(i)[None,]
                    bundle.append(tensor)
                    print(f"      → Tensor: {tensor.shape} (NHWC)")
                except Exception as e:
                    print(f"     ❌ Failed to load {filename}: {e}")
            else:
                print(f"     ⚠️ File not found: {filename}")
                print(f"     🛑 CRITICAL ERROR: Missing file. Aborting Queue.")
                global NODE_STATE
                if folder in NODE_STATE:
                    NODE_STATE[folder]["queue"] = []
                return ([], "", 1024, 1024, 1, "ERROR", "")
        
        if len(bundle) == 0:
            print("   ⚠️ No images bundled. Returning empty.")
            return ([], "", w, h, batch, "WORKING", output_name)
        
        print(f"   ✅ Bundle complete: {len(bundle)} images")
        print(f"   🎬 Prompt: {prompt[:100]}...")
        print(f"   📐 Dimensions: {w}x{h}, Batch: {batch}")
        
        return (bundle, prompt, w, h, batch, "WORKING", output_name)

    def _apply_padding(self, img, padding_spec, filename="image"):
        """Apply smart two-stage padding"""
        original_w, original_h = img.width, img.height
        
        print(f"\n   🎨 PADDING: {filename}")
        print(f"   ├─ Original: {original_w}x{original_h}")
        
        # Stage 1: Directional Padding
        if "directional" in padding_spec:
            print(f"   ├─ STAGE 1: Directional Padding")
            directional = padding_spec["directional"]
            
            def parse_value(val, dimension):
                if isinstance(val, str) and "%" in val:
                    percent = float(val.replace("%", ""))
                    return int(dimension * percent / 100)
                return int(val)
            
            pad_left = parse_value(directional.get("left", 0), original_w)
            pad_right = parse_value(directional.get("right", 0), original_w)
            pad_top = parse_value(directional.get("top", 0), original_h)
            pad_bottom = parse_value(directional.get("bottom", 0), original_h)
            
            if pad_left: print(f"   │  └─ Left: +{pad_left}px")
            if pad_right: print(f"   │  └─ Right: +{pad_right}px")
            if pad_top: print(f"   │  └─ Top: +{pad_top}px")
            if pad_bottom: print(f"   │  └─ Bottom: +{pad_bottom}px")
            
            new_w = original_w + pad_left + pad_right
            new_h = original_h + pad_top + pad_bottom
            fill_color = self._get_fill_color(padding_spec.get("fill_color", "white"))
            
            temp_img = Image.new("RGB", (new_w, new_h), fill_color)
            temp_img.paste(img, (pad_left, pad_top))
            img = temp_img
            print(f"   └─ After Stage 1: {img.width}x{img.height}")
        
        # Stage 2: Aspect Ratio Padding
        if "target_aspect" in padding_spec:
            print(f"\n   ├─ STAGE 2: Aspect Ratio Target")
            target_w, target_h = map(int, padding_spec["target_aspect"].split(":"))
            target_ratio = target_w / target_h
            current_ratio = img.width / img.height
            
            print(f"   │  ├─ Current ratio: {current_ratio:.3f}")
            print(f"   │  └─ Target ratio: {target_ratio:.3f} ({padding_spec['target_aspect']})")
            
            position = padding_spec.get("position", "center")
            fill_color = self._get_fill_color(padding_spec.get("fill_color", "white"))
            
            if current_ratio < target_ratio:
                # Too tall, add width
                required_w = int(img.height * target_ratio)
                pad_total = required_w - img.width
                print(f"   │  ├─ Need: +{pad_total}px width")
                
                if position == "left":
                    pad_left, pad_right = 0, pad_total
                elif position == "right":
                    pad_left, pad_right = pad_total, 0
                else:  # center
                    pad_left = pad_total // 2
                    pad_right = pad_total - pad_left
                
                print(f"   │  └─ Distribution: L+{pad_left}px, R+{pad_right}px")
                
                final_w = img.width + pad_left + pad_right
                final_h = img.height
                padded = Image.new("RGB", (final_w, final_h), fill_color)
                padded.paste(img, (pad_left, 0))
                img = padded
                
            elif current_ratio > target_ratio:
                # Too wide, add height
                required_h = int(img.width / target_ratio)
                pad_total = required_h - img.height
                print(f"   │  ├─ Need: +{pad_total}px height")
                
                if position == "top":
                    pad_top, pad_bottom = 0, pad_total
                elif position == "bottom":
                    pad_top, pad_bottom = pad_total, 0
                else:  # center
                    pad_top = pad_total // 2
                    pad_bottom = pad_total - pad_top
                
                print(f"   │  └─ Distribution: T+{pad_top}px, B+{pad_bottom}px")
                
                final_w = img.width
                final_h = img.height + pad_top + pad_bottom
                padded = Image.new("RGB", (final_w, final_h), fill_color)
                padded.paste(img, (0, pad_top))
                img = padded
            
            final_ratio = img.width / img.height
            print(f"   └─ Final: {img.width}x{img.height} ({padding_spec['target_aspect']} ✓)")
        
        return img

    def _get_fill_color(self, color_name):
        """Convert color name to RGB tuple"""
        colors = {
            "white": (255, 255, 255),
            "black": (0, 0, 0),
            "gray": (128, 128, 128)
        }
        return colors.get(color_name.lower(), (255, 255, 255))

    def _parse_json(self, text):
        """Extract and parse JSON from response with detailed logging"""
        
        print(f"\n{'─'*60}")
        print(f"🔍 PARSING JSON...")
        print(f"{'─'*60}")
        
        # Check if response is empty
        if not text or text.strip() == "":
            print(f"   ❌ ERROR: Response is empty!")
            return {}
        
        # Show response length and preview
        print(f"   📏 Response length: {len(text)} characters")
        preview = text[:200].replace('\n', ' ')
        print(f"   📝 Preview: {preview}...")
        
        # Method 1: Try direct JSON parse
        print(f"\n   🔧 Method 1: Direct JSON parse...")
        try:
            data = json.loads(text)
            print(f"   ✅ SUCCESS! Parsed JSON with {len(data)} keys: {list(data.keys())}")
            self._validate_json_structure(data)
            return data
        except json.JSONDecodeError as e:
            print(f"   ⚠️ Failed: {e.msg} at position {e.pos}")
        
        # Method 2: Try to extract from ```json block
        print(f"\n   🔧 Method 2: Extract from ```json block...")
        if "```json" in text:
            try:
                start = text.find("```json") + 7
                end = text.find("```", start)
                if end > start:
                    json_text = text[start:end].strip()
                    print(f"   📦 Found JSON block: {len(json_text)} chars")
                    data = json.loads(json_text)
                    print(f"   ✅ SUCCESS! Parsed JSON with {len(data)} keys: {list(data.keys())}")
                    self._validate_json_structure(data)
                    return data
            except json.JSONDecodeError as e:
                print(f"   ⚠️ Failed: {e.msg}")
        else:
            print(f"   ⚠️ No ```json block found")
        
        # Method 3: Try generic code block
        print(f"\n   🔧 Method 3: Extract from generic ``` block...")
        if "```" in text:
            try:
                start = text.find("```") + 3
                # Skip language identifier if present
                newline = text.find("\n", start)
                if newline > start and newline - start < 20:
                    start = newline + 1
                end = text.find("```", start)
                if end > start:
                    json_text = text[start:end].strip()
                    print(f"   📦 Found code block: {len(json_text)} chars")
                    data = json.loads(json_text)
                    print(f"   ✅ SUCCESS! Parsed JSON with {len(data)} keys: {list(data.keys())}")
                    self._validate_json_structure(data)
                    return data
            except json.JSONDecodeError as e:
                print(f"   ⚠️ Failed: {e.msg}")
        else:
            print(f"   ⚠️ No code block found")
        
        # Method 4: Try to find JSON object in text
        print(f"\n   🔧 Method 4: Search for JSON object {{...}}...")
        brace_start = text.find("{")
        if brace_start != -1:
            # Find matching closing brace
            depth = 0
            for i, char in enumerate(text[brace_start:]):
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        json_text = text[brace_start:brace_start + i + 1]
                        try:
                            print(f"   📦 Found object: {len(json_text)} chars")
                            data = json.loads(json_text)
                            print(f"   ✅ SUCCESS! Parsed JSON with {len(data)} keys: {list(data.keys())}")
                            self._validate_json_structure(data)
                            return data
                        except json.JSONDecodeError as e:
                            print(f"   ⚠️ Failed: {e.msg}")
                        break
        else:
            print(f"   ⚠️ No {{ found in response")
        
        # All methods failed
        print(f"\n{'='*60}")
        print(f"❌ JSON PARSE FAILED - ALL METHODS EXHAUSTED")
        print(f"{'='*60}")
        print(f"📄 FULL RESPONSE (first 1000 chars):")
        print(f"{'─'*60}")
        print(text[:1000])
        print(f"{'─'*60}")
        print(f"\n💡 POSSIBLE CAUSES:")
        print(f"   1. AI rambled without producing JSON")
        print(f"   2. Response got cut off (check max_tokens)")
        print(f"   3. AI misunderstood the task")
        print(f"\n💡 SUGGESTED FIX:")
        print(f"   - Check your system prompt")
        print(f"   - Make sure it clearly asks for JSON output")
        print(f"   - Try resetting history and running again")
        print(f"{'='*60}\n")
        
        return {}
    
    def _validate_json_structure(self, data):
        """Validate and report JSON structure"""
        print(f"\n   📋 JSON STRUCTURE VALIDATION:")
        
        # Check for required fields
        if "thought" in data:
            thought_preview = data["thought"][:80].replace('\n', ' ') if data["thought"] else "(empty)"
            print(f"   ✓ thought: \"{thought_preview}...\"")
        else:
            print(f"   ⚠️ thought: MISSING (optional but recommended)")
        
        if "status" in data:
            print(f"   ✓ status: {data['status']}")
        else:
            print(f"   ⚠️ status: MISSING (will default to WORKING)")
        
        if "queue" in data:
            print(f"   ✓ queue: {len(data['queue'])} jobs")
            for i, job in enumerate(data["queue"]):
                load_count = len(job.get("load", []))
                prompt_preview = job.get("prompt", "")[:50]
                print(f"      Job {i+1}: {load_count} images, prompt=\"{prompt_preview}...\"")
        else:
            print(f"   ⚠️ queue: MISSING (no generation jobs)")
        
        if "ops" in data:
            print(f"   ✓ ops: {len(data['ops'])} file operations")
        
        if "refresh_context" in data:
            print(f"   ✓ refresh_context: {data['refresh_context']}")

    def _handle_file_op(self, folder, op):
        """Execute a file operation"""
        action = op.get("action")
        if action == "delete":
            filepath = os.path.join(folder, op["file"])
            if os.path.exists(filepath):
                os.remove(filepath)
                print(f"   🗑️ Deleted: {op['file']}")
        elif action == "rename":
            src = os.path.join(folder, op["src"])
            dest = os.path.join(folder, op["dest"])
            if os.path.exists(src):
                shutil.move(src, dest)
                print(f"   ✏️ Renamed: {op['src']} → {op['dest']}")

    def _save_state(self, folder, state):
        """Save state to disk"""
        state_file = os.path.join(folder, "presence_state.json")
        with open(state_file, "w") as f:
            json.dump({
                "seen_files": list(state["seen_files"]),
                "queue": state["queue"]
            }, f, indent=2)

    def _save_done(self, folder, response):
        """Save DONE response"""
        done_file = os.path.join(folder, "DONE.json")
        with open(done_file, "w") as f:
            f.write(response)
        print(f"💾 Saved DONE state to {done_file}")

# Register node
NODE_CLASS_MAPPINGS = {
    "PresenceDirectorFireworks": PresenceDirectorFireworks
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PresenceDirectorFireworks": "🔥 Presence Director (Fireworks AI)"
}
