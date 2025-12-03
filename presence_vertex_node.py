import os
import time
import json
import shutil
import vertexai
from vertexai.generative_models import GenerativeModel
from google.oauth2 import service_account
import torch
import numpy as np
from PIL import Image, ImageOps

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
        
        print(f"\n{'='*80}")
        print(f"🏭 PRESENCE DIRECTOR (Vertex AI - Gemini 3 Pro)")
        print(f"{'='*80}")
        
        # Ensure folder exists
        if not os.path.exists(active_folder):
            os.makedirs(active_folder, exist_ok=True)
            print(f"📁 Created folder: {active_folder}")
        
        # Initialize state for this folder
        global NODE_STATE
        if active_folder not in NODE_STATE:
            NODE_STATE[active_folder] = {
                "chat": None,
                "seen_files": set(),
                "queue": [],
                "last_input": ""
            }
        
        state = NODE_STATE[active_folder]
        
        # Handle reset
        if reset_history:
            print("🔄 RESET TRIGGERED. Clearing history...")
            state["chat"] = None
            state["seen_files"] = set()
            state["queue"] = []
            state["last_input"] = ""
        
        # Load persistent state from disk
        state_file = os.path.join(active_folder, "presence_state.json")
        if os.path.exists(state_file) and not reset_history:
            try:
                with open(state_file, "r") as f:
                    disk_state = json.load(f)
                    state["seen_files"] = set(disk_state.get("seen_files", []))
                    state["queue"] = disk_state.get("queue", [])
                    print(f"📂 Loaded state: {len(state['seen_files'])} seen files, {len(state['queue'])} queued jobs")
            except Exception as e:
                print(f"⚠️ Could not load state: {e}")
        
        # =================================================================================================
        # 🤖 MODE A: THE ROBOT (EXECUTOR)
        # =================================================================================================
        if len(state["queue"]) > 0:
            print(f"🤖 ROBOT MODE: Executing job 1 of {len(state['queue'])}...")
            job = state["queue"].pop(0)
            
            # Save state after popping job
            self._save_state(active_folder, state)
            
            return self._execute_job(active_folder, job)

        # =================================================================================================
        # 🧠 MODE B: THE BRAIN (ITERATOR)
        # =================================================================================================
        print("🧠 BRAIN MODE: Scanning folder...")
        
        # 1. SCAN & FILTER
        current_files = set(f for f in os.listdir(active_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')))
        new_files = list(current_files - state["seen_files"])
        state["seen_files"].update(new_files)
        
        # 2. PREPARE PAYLOAD
        file_list_text = "CURRENT FILE MANIFEST:\n"
        sorted_files = sorted(list(current_files))
        for f in sorted_files:
            file_list_text += f"- {f}\n"
            
        upload_images = []
        print(f"   - Found {len(new_files)} new images to upload.")
        for f in new_files:
            path = os.path.join(active_folder, f)
            try:
                img = Image.open(path)
                img = ImageOps.exif_transpose(img)
                
                # Resize to ~1 megapixel for Gemini (saves API costs)
                current_pixels = img.width * img.height
                target_pixels = 1024 * 1024  # 1MP
                if current_pixels > target_pixels:
                    scale_factor = (target_pixels / current_pixels) ** 0.5
                    new_width = int(img.width * scale_factor)
                    new_height = int(img.height * scale_factor)
                    img = img.resize((new_width, new_height), Image.LANCZOS)
                    print(f"   📐 Resized {f}: {img.width}x{img.height} (~1MP)")
                
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

            user_message = [base_instruction, file_list_text]
            user_message.extend(upload_images)
            
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
        prompt = job.get("prompt", "")
        w = job.get("w", 1024)
        h = job.get("h", 1024)
        batch = job.get("batch", 1)
        output_name = job.get("output_name", "gen")
        load_list = job.get("load", [])
        
        bundle = []
        print(f"   📦 Bundling: {load_list}")
        
        for filename in load_list:
            path = os.path.join(folder, filename)
            if os.path.exists(path):
                try:
                    img = Image.open(path).convert("RGB")
                    img = ImageOps.exif_transpose(img)
                    i = np.array(img).astype(np.float32) / 255.0
                    tensor = torch.from_numpy(i)[None,]
                    bundle.append(tensor)
                except Exception as e:
                    print(f"     ❌ Failed to load {filename}: {e}")
            else:
                print(f"     ⚠️ File not found: {filename}")
                print("     🛑 CRITICAL ERROR: Missing file. Aborting Queue.")
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
