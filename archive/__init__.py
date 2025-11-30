from .unaliver_node import UnaliverNode
from .show_text_node import BeautifulTextNode
from .unaliver_step_iterator import UnaliverStepIterator
from .inspect_node import InspectNode
from .bundle_preview import UnaliverBundlePreview

NODE_CLASS_MAPPINGS = {
    "UnaliverNode": UnaliverNode,
    "BeautifulTextNode": BeautifulTextNode,
    "UnaliverStepIterator": UnaliverStepIterator,
    "InspectNode": InspectNode,
    "UnaliverBundlePreview": UnaliverBundlePreview
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UnaliverNode": "Unaliver (Archived)",
    "BeautifulTextNode": "Beautiful Text 📜 (Archived)",
    "UnaliverStepIterator": "Unaliver Step Iterator 🎬 (Archived)",
    "InspectNode": "Inspect Node 🕵️ (Archived)",
    "UnaliverBundlePreview": "Unaliver Bundle to Image 📦 (Archived)"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
