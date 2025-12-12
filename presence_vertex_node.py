import os
import time
import json
import shutil
import io
import vertexai
from vertexai.generative_models import GenerativeModel, Part
from google.oauth2 import service_account
import torch
import numpy as np
from PIL import Image, ImageOps
from datetime import datetime

# --------------------------------------------------------------------------------
# GLOBAL STATE MANAGEMENT
# Key: active_folder_path
# Value: {
#   "chat": Vertex_Chat_Object,
#   "seen_files": set(),
#   "queue": []
# }
# --------------------------------------------------------------------------------
NODE_STATE = {}

# --------------------------------------------------------------------------------
# PROJECT LOGGER - Writes to both console AND project log file
# --------------------------------------------------------------------------------
class ProjectLogger:
    def __init__(self, folder):
        self.folder = folder
        self.log_file = os.path.join(folder, "presence_log.txt")
        
    def log(self, message, also_print=True):
        """Log message to file and optionally to console"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"[{timestamp}] {message}"
        
        # Write to file
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(log_line + "\n")
        except Exception as e:
            print(f"⚠️ Could not write to log: {e}")
        
        # Also print to console
        if also_print:
            print(message)
    
    def section(self, title):
        """Log a section header"""
        line = "=" * 60
        self.log(line)
        self.log(title)
        self.log(line)
    
    def subsection(self, title):
        """Log a subsection header"""
        line = "-" * 40
        self.log(line)
        self.log(title)
        self.log(line)
    
    def job_details(self, job):
        """Log full job details"""
        self.subsection("📋 JOB DETAILS")
        self.log(f"   Prompt: {job.get('prompt', 'N/A')}")
        self.log(f"   Output Name: {job.get('output_name', 'gen')}")
        self.log(f"   Width: {job.get('w', 1024)}")
        self.log(f"   Height: {job.get('h', 1024)}")
        self.log(f"   Batch: {job.get('batch', 1)}")
        self.log(f"   MP: {job.get('mp', 1)}MP")
        
        load_list = job.get('load', [])
        self.log(f"   Load List ({len(load_list)} items):")
        for item in load_list:
            if isinstance(item, dict):
                self.log(f"      - {item.get('file')} @ {item.get('mp', 1)}MP")
            else:
                self.log(f"      - {item} @ 1MP (default)")
        
        padding = job.get('padding')
        if padding:
            self.log(f"   Padding: {json.dumps(padding)}")
        else:
            self.log(f"   Padding: None")

class PresenceDirectorVertex:
    """
    🏭 PRESENCE DIRECTOR (Vertex AI Edition - Gemini 3 Pro)
    - Uses Vertex AI SDK with Service Account authentication
    - Gemini 3 Pro ONLY (location='global')
    - Switches between 'Brain Mode' (Gemini) and 'Robot Mode' (Queue Execution)
    - Manages File System, Context, and Flux Generation
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "active_folder": ("STRING", {"default": "C:/Presence/Job_01"}),
                "service_account_json": ("STRING", {"default": "C:/path/to/your-service-account.json"}),
                "project_id": ("STRING", {"default": "your-project-id"}),
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

    def run_director(self, active_folder, service_account_json, project_id, system_prompt, user_input, reset_history, seed):
        """Main execution - switches between Brain and Robot modes"""
        
        # Ensure folder exists first
        if not os.path.exists(active_folder):
            os.makedirs(active_folder, exist_ok=True)
        
        # Initialize project logger
        logger = ProjectLogger(active_folder)
        
        logger.section("🏭 PRESENCE DIRECTOR (Vertex AI - Gemini 3 Pro)")
        logger.log(f"   📂 Active folder: {active_folder}")
        logger.log(f"   🔄 Reset requested: {reset_history}")
        logger.log(f"   🌱 Seed: {seed}")
        logger.log(f"   📝 System prompt length: {len(system_prompt)} chars")
        
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
                print(f"      - chat: {'Active' if old_state.get('chat') else 'None'}")
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
                "chat": None,
                "seen_files": set(),
                "queue": [],
                "last_input": ""
            }
            
            print(f"   ✨ Created fresh state")
            print(f"   📊 AFTER RESET:")
            print(f"      - seen_files: 0 files")
            print(f"      - queue: 0 jobs")
            print(f"      - chat: None")
            print(f"{'='*60}")
            print(f"✅ RESET COMPLETE - All files will be treated as NEW\n")
        
        # =========================================================
        # NORMAL INIT: Load or create state
        # =========================================================
        else:
            # Initialize if not exists
            if active_folder not in NODE_STATE:
                NODE_STATE[active_folder] = {
                    "chat": None,
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
        
        # =================================================================================================
        # 🤖 MODE A: THE ROBOT (EXECUTOR)
        # =================================================================================================
        if len(state["queue"]) > 0:
            print(f"\n🤖 ROBOT MODE: {len(state['queue'])} jobs in queue...")
            print(f"   ⚡ Executing first job...")
            job = state["queue"].pop(0)
            
            # DEBUG: Show job immediately after popping
            print(f"   📋 JOB FROM QUEUE:")
            print(f"      w = {job.get('w', 'NOT FOUND (default 1024)')}")
            print(f"      h = {job.get('h', 'NOT FOUND (default 1024)')}")
            print(f"      output_name = {job.get('output_name', 'gen')}")
            
            # Save state after popping job
            self._save_state(active_folder, state)
            
            return self._execute_job(active_folder, job)

        # =================================================================================================
        # 🧠 MODE B: THE BRAIN (ITERATOR)
        # =================================================================================================
        print(f"\n🧠 BRAIN MODE: Analyzing folder state...")
        
        # 1. SCAN & FILTER
        current_files = set(f for f in os.listdir(active_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')))
        new_files = list(current_files - state["seen_files"])
        
        # Detailed logging
        print(f"   📂 Folder: {active_folder}")
        print(f"   📁 Total images in folder: {len(current_files)}")
        print(f"   👁️ Already seen: {len(state['seen_files'])} files")
        print(f"   🆕 New files to process: {len(new_files)}")
        if len(new_files) > 0:
            for f in sorted(new_files):
                print(f"      → {f}")
        
        state["seen_files"].update(new_files)
        
        # 2. PREPARE PAYLOAD
        file_list_text = "CURRENT FILE MANIFEST:\n"
        sorted_files = sorted(list(current_files))
        for f in sorted_files:
            file_list_text += f"- {f}\n"
            
        upload_images = []
        
        # Send ORIGINAL resolution to Gemini/Qwen (cheap, no resizing)
        print(f"   - Found {len(new_files)} new images to upload (ORIGINAL resolution).")
        for f in new_files:
            path = os.path.join(active_folder, f)
            try:
                img = Image.open(path)
                img = ImageOps.exif_transpose(img)
                print(f"   📤 Uploading {f}: {img.width}x{img.height} (original)")
                upload_images.append(img)
            except Exception as e:
                print(f"   ❌ Error reading {f}: {e}")

        # --- SAFETY CHECK: EMPTY FOLDER ---
        if len(current_files) == 0 and len(state["seen_files"]) == 0:
            print("   ⚠️ FOLDER IS EMPTY. Waiting for inputs...")
            return ([], "", 1024, 1024, 1, "IDLE", "")
        # ----------------------------------

        # 3. CALL GEMINI
        try:
            if state["chat"] is None:
                self._init_vertex_ai(state, service_account_json, project_id, system_prompt)
            
            chat = state["chat"]
            
            base_instruction = "Review the current state. Execute File Ops. Determine Next Steps."
            if user_input and user_input.strip() != "" and user_input != state.get("last_input", ""):
                print(f"   📢 USER INTERVENTION DETECTED: {user_input}")
                base_instruction += f"\n\n🚨 USER COMMAND: {user_input}"
                state["last_input"] = user_input

            # Build message with text and images
            user_message = [base_instruction, file_list_text]
            
            # Convert PIL images to Vertex AI Part objects
            for img in upload_images:
                # Convert PIL image to bytes
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='PNG')
                img_byte_arr = img_byte_arr.getvalue()
                
                # Create Part object
                image_part = Part.from_data(data=img_byte_arr, mime_type="image/png")
                user_message.append(image_part)
            
            print("   🚀 Sending to Gemini 3 Pro...")
            response = chat.send_message(user_message)
            response_text = response.text
            
            print("\n" + "="*50)
            print("🤖 FULL GEMINI 3 PRO RESPONSE:")
            print(response_text)
            print("="*50 + "\n")
            
            data = self._parse_json(response_text)
            
            if "ops" in data:
                for op in data["ops"]:
                    self._handle_file_op(active_folder, op)
            
            if data.get("refresh_context", False):
                print("   🔄 Gemini requested Context Refresh. Clearing memory.")
                state["seen_files"] = set()
            
            if data.get("status") == "DONE":
                print("   ✅ JOB DONE.")
                with open(os.path.join(active_folder, "DONE.json"), "w") as f:
                    json.dump(data, f, indent=2)
                state["chat"] = None
                state["seen_files"] = set()
                state["queue"] = []
                self._save_state(active_folder, state)
                return ([], "", 1024, 1024, 1, "DONE", "")

            if "queue" in data and isinstance(data["queue"], list):
                for item in data["queue"]:
                    state["queue"].append(item)
                print(f"   📥 Added {len(data['queue'])} jobs to Queue.")
            
            # Save state
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
        # Initialize logger for this project
        logger = ProjectLogger(folder)
        
        prompt = job.get("prompt", "")
        w = job.get("w", 1024)
        h = job.get("h", 1024)
        batch = job.get("batch", 1)
        output_name = job.get("output_name", "gen")
        load_list = job.get("load", [])
        padding_spec = job.get("padding", None)
        mp = job.get("mp", 1)
        
        # Log full job details
        logger.section("🤖 ROBOT MODE: EXECUTING JOB")
        logger.job_details(job)
        
        # DEBUG: Show explicitly parsed values
        logger.log(f"\n   🔍 PARSED VALUES FROM JOB:")
        logger.log(f"      w = {w} (from job.get('w', 1024))")
        logger.log(f"      h = {h} (from job.get('h', 1024))")
        logger.log(f"      batch = {batch}")
        logger.log(f"      mp = {mp}")
        logger.log(f"      Raw job dict: {json.dumps(job, indent=2)[:500]}...")
        
        bundle = []
        logger.log(f"\n   📦 Loading images for Flux...")
        
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
        
        return (bundle, prompt, w, h, int(batch), "WORKING", output_name)

    def _init_vertex_ai(self, state, service_account_json, project_id, system_prompt):
        """Initializes Vertex AI with Service Account and Gemini 3 Pro"""
        print(f"   🔐 Authenticating with service account: {service_account_json}")
        
        try:
            # Load credentials
            credentials = service_account.Credentials.from_service_account_file(
                service_account_json,
                scopes=['https://www.googleapis.com/auth/cloud-platform']
            )
            
            # Initialize Vertex AI
            # CRITICAL: location='global' is required for Gemini 3 Pro
            vertexai.init(
                project=project_id,
                location="global",  # Gemini 3 Pro is global-only
                credentials=credentials
            )
            
            print(f"   ✅ Vertex AI initialized (project={project_id}, location=global)")
            
            # Create Gemini 3 Pro model
            model = GenerativeModel(
                "gemini-3-pro-preview",
                system_instruction=system_prompt
            )
            
            # Start chat session
            state["chat"] = model.start_chat(history=[])
            
            print(f"   ✨ Session Started with Gemini 3 Pro")
            
        except Exception as e:
            print(f"   ❌ Failed to initialize Vertex AI: {e}")
            raise Exception(f"Vertex AI initialization failed: {e}")

    def _parse_json(self, text):
        """Robust JSON extraction"""
        try:
            text = text.strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            return json.loads(text)
        except:
            print("   ❌ JSON Parse Error")
            return {}

    def _handle_file_op(self, folder, op):
        """Executes Delete or Rename"""
        action = op.get("action")
        
        if action == "delete":
            target = op.get("file")
            path = os.path.join(folder, target)
            if os.path.exists(path):
                try:
                    os.remove(path)
                    print(f"     🗑️ Deleted: {target}")
                except Exception as e:
                    print(f"     ❌ Delete Failed: {e}")
                    
        elif action == "rename":
            src = op.get("src")
            dest = op.get("dest")
            src_path = os.path.join(folder, src)
            dest_path = os.path.join(folder, dest)
            if os.path.exists(src_path):
                try:
                    if os.path.exists(dest_path):
                        os.remove(dest_path)
                    os.rename(src_path, dest_path)
                    print(f"     🏷️ Renamed: {src} -> {dest}")
                except Exception as e:
                    print(f"     ❌ Rename Failed: {e}")
    
    def _apply_padding(self, img, padding_spec, filename="image"):
        """Apply smart two-stage padding system"""
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
            
            if pad_left: print(f"   │  ├─ Left: +{pad_left}px")
            if pad_right: print(f"   │  ├─ Right: +{pad_right}px")
            if pad_top: print(f"   │  ├─ Top: +{pad_top}px")
            if pad_bottom: print(f"   │  └─ Bottom: +{pad_bottom}px")
            
            new_w = original_w + pad_left + pad_right
            new_h = original_h + pad_top + pad_bottom
            
            fill_color = self._get_fill_color(padding_spec.get("fill_color", "white"))
            temp_img = Image.new("RGB", (new_w, new_h), fill_color)
            temp_img.paste(img, (pad_left, pad_top))
            img = temp_img
            print(f"   └─ After Stage 1: {new_w}x{new_h}")
        
        # Stage 2: Aspect Ratio Padding
        if "target_aspect" in padding_spec:
            print(f"   ├─ STAGE 2: Aspect Ratio Target")
            target_aspect = padding_spec["target_aspect"]
            target_w, target_h = map(int, target_aspect.split(":"))
            target_ratio = target_w / target_h
            current_ratio = img.width / img.height
            
            print(f"   │  ├─ Current ratio: {current_ratio:.3f}")
            print(f"   │  └─ Target ratio: {target_ratio:.3f} ({target_aspect})")
            
            if abs(current_ratio - target_ratio) < 0.01:
                print(f"   └─ Already at {target_aspect}, no padding needed")
                return img
            
            position = padding_spec.get("position", "center")
            
            if current_ratio < target_ratio:
                # Too tall, add width
                required_w = int(img.height * target_ratio)
                pad_total = required_w - img.width
                
                if position == "left":
                    pad_left, pad_right = 0, pad_total
                elif position == "right":
                    pad_left, pad_right = pad_total, 0
                else:
                    pad_left = pad_total // 2
                    pad_right = pad_total - pad_left
                
                pad_top = pad_bottom = 0
                print(f"   │  ├─ Need: +{pad_total}px width")
                print(f"   │  └─ Distribution: L+{pad_left}px, R+{pad_right}px")
            else:
                # Too wide, add height
                required_h = int(img.width / target_ratio)
                pad_total = required_h - img.height
                
                if position == "top":
                    pad_top, pad_bottom = 0, pad_total
                elif position == "bottom":
                    pad_top, pad_bottom = pad_total, 0
                else:
                    pad_top = pad_total // 2
                    pad_bottom = pad_total - pad_top
                
                pad_left = pad_right = 0
                print(f"   │  ├─ Need: +{pad_total}px height")
                print(f"   │  └─ Distribution: T+{pad_top}px, B+{pad_bottom}px")
            
            final_w = img.width + pad_left + pad_right
            final_h = img.height + pad_top + pad_bottom
            
            fill_color = self._get_fill_color(padding_spec.get("fill_color", "white"))
            padded = Image.new("RGB", (final_w, final_h), fill_color)
            padded.paste(img, (pad_left, pad_top))
            
            print(f"   └─ Final: {final_w}x{final_h} ({target_aspect} ✓)\n")
            return padded
        
        elif "left" in padding_spec or "right" in padding_spec:
            # Explicit pixel padding (Mode 3)
            pad_left = padding_spec.get("left", 0)
            pad_right = padding_spec.get("right", 0)
            pad_top = padding_spec.get("top", 0)
            pad_bottom = padding_spec.get("bottom", 0)
            
            final_w = img.width + pad_left + pad_right
            final_h = img.height + pad_top + pad_bottom
            
            fill_color = self._get_fill_color(padding_spec.get("fill_color", "white"))
            padded = Image.new("RGB", (final_w, final_h), fill_color)
            padded.paste(img, (pad_left, pad_top))
            
            print(f"   └─ Explicit padding: {final_w}x{final_h}\n")
            return padded
        
        return img
    
    def _get_fill_color(self, color_spec):
        """Convert color specification to RGB tuple"""
        if color_spec == "white":
            return (255, 255, 255)
        elif color_spec == "black":
            return (0, 0, 0)
        elif color_spec == "gray":
            return (128, 128, 128)
        else:
            return (255, 255, 255)  # Default white
    
    def _save_state(self, folder, state):
        """Save state to disk for persistence"""
        state_file = os.path.join(folder, "presence_state.json")
        try:
            disk_state = {
                "seen_files": list(state["seen_files"]),
                "queue": state["queue"]
            }
            with open(state_file, "w") as f:
                json.dump(disk_state, f, indent=2)
        except Exception as e:
            print(f"⚠️ Could not save state: {e}")

class PresenceSaver:
    """
    💾 PRESENCE SAVER
    - Saves images to the active_folder using the filename from the Director.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "active_folder": ("STRING", {"default": "C:/Presence/Job_01"}),
                "filename": ("STRING", {"default": "gen"}),
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
            filename = "gen"
            
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
            print(f"   💾 Saved to folder: {user_path}")
            
            # ALSO save to ComfyUI temp for preview
            temp_dir = folder_paths.get_temp_directory()
            temp_path = os.path.join(temp_dir, fname)
            img.save(temp_path, compress_level=4)
            
            results.append({
                "filename": fname,
                "subfolder": "",
                "type": "temp"
            })
            
        return {"ui": {"images": results}}

NODE_CLASS_MAPPINGS = {
    "PresenceDirectorVertex": PresenceDirectorVertex,
    "PresenceSaver": PresenceSaver
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PresenceDirectorVertex": "Presence Director (Vertex AI) 🏭",
    "PresenceSaver": "Presence Saver 💾"
}
