# =============================================================================
# PRESENCE DIRECTOR - Multi-Agent System
# =============================================================================
# ⚡ Gemini 3 Flash powered AI-directed image generation
# =============================================================================

from .presence_gemini3_node import PresenceDirector
from .utility_nodes import FluxAdaptiveInjector, PresenceSaver
from .gaussian_padding_node import GaussianNoisePadding

NODE_CLASS_MAPPINGS = {
    "PresenceDirector": PresenceDirector,
    "FluxAdaptiveInjector": FluxAdaptiveInjector,
    "PresenceSaver": PresenceSaver,
    "GaussianNoisePadding": GaussianNoisePadding
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PresenceDirector": "⚡ Presence Director",
    "FluxAdaptiveInjector": "Flux Injector",
    "PresenceSaver": "Presence Saver",
    "GaussianNoisePadding": "Gaussian Noise Padding"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

print("[PresenceAI] Nodes loaded:")
print("   ⚡ Presence Director")
print("   Flux Injector")
print("   Presence Saver")
print("   Gaussian Noise Padding")
