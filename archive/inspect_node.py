import inspect
import nodes
import sys
import os

print("\n" + "="*50)
print("🕵️ INSPECTING ReferenceLatent NODE...")
print("="*50)

found = False

# 1. Search in main NODE_CLASS_MAPPINGS
if "ReferenceLatent" in nodes.NODE_CLASS_MAPPINGS:
    cls = nodes.NODE_CLASS_MAPPINGS["ReferenceLatent"]
    print(f"✅ Found 'ReferenceLatent' in core nodes!")
    try:
        source = inspect.getsource(cls)
        print("\n--- SOURCE CODE START ---")
        print(source)
        print("--- SOURCE CODE END ---\n")
        
        # Write to file so we can read it
        output_path = os.path.join(os.path.dirname(__file__), "reference_latent_source.txt")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(source)
        print(f"✅ Source code written to: {output_path}")
        
        found = True
    except Exception as e:
        print(f"❌ Could not get source: {e}")

# 2. If not found, search all loaded custom nodes
if not found:
    print("⚠️  Not found in core mappings. Searching all loaded modules...")
    # This is harder because custom nodes register themselves.
    # But we can check if it's in the global mappings if they are exposed.
    # Usually nodes.NODE_CLASS_MAPPINGS contains everything after load.
    
    # Let's try to find it in the internal object graph if possible
    pass

print("="*50 + "\n")

class InspectNode:
    """
    Placeholder node just to run the inspection code on import.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}
    RETURN_TYPES = ()
    FUNCTION = "do_nothing"
    CATEGORY = "Unaliver/Archive"

    def do_nothing(self):
        return ()

NODE_CLASS_MAPPINGS = {
    "InspectNode": InspectNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "InspectNode": "Inspect Node 🕵️ (Archived)"
}
