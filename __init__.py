# =============================================================================
# PRESENCE DIRECTOR - Multi-Agent System
# =============================================================================
# ⚡ Gemini 3 Flash powered AI-directed image generation
# =============================================================================

from .presence_gemini3_node import PresenceDirector
from .utility_nodes import FluxAdaptiveInjector, PresenceSaver, PresencePreview
from .gaussian_padding_node import GaussianNoisePadding
from .padding_tester_node import PresencePaddingTester

NODE_CLASS_MAPPINGS = {
    "PresenceDirector": PresenceDirector,
    "FluxAdaptiveInjector": FluxAdaptiveInjector,
    "PresenceSaver": PresenceSaver,
    "PresencePreview": PresencePreview,
    "GaussianNoisePadding": GaussianNoisePadding,
    "PresencePaddingTester": PresencePaddingTester
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PresenceDirector": "⚡ Presence Director",
    "FluxAdaptiveInjector": "Flux Injector",
    "PresenceSaver": "Presence Saver",
    "PresencePreview": "👁 Presence Preview",
    "GaussianNoisePadding": "Gaussian Noise Padding",
    "PresencePaddingTester": "🔲 Padding Tester"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

print("[PresenceAI] Nodes loaded:")
print("   ⚡ Presence Director")
print("   Flux Injector")
print("   Presence Saver")
print("   👁 Presence Preview")
print("   🔲 Padding Tester")

