class BeautifulTextNode:
    """
    Beautiful Text: A node that displays text output directly in the UI.
    """
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
            }
        }

class BeautifulTextNode:
    """
    Beautiful Text: A node that displays text output directly in the UI.
    """
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "show_text"
    CATEGORY = "Unaliver/Archive"
    OUTPUT_NODE = True

    def show_text(self, text):
        # Display text directly on the node
        return {"ui": {"text": [text]}}

# Mappings
NODE_CLASS_MAPPINGS = {
    "BeautifulTextNode": BeautifulTextNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BeautifulTextNode": "Beautiful Text 📜 (Archived)"
}
