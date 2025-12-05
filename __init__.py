from .unaliver_nodes import FluxAdaptiveInjector, PresenceSaver
from .presence_vertex_node import PresenceDirectorVertex
from .presence_fireworks_node import PresenceDirectorFireworks

NODE_CLASS_MAPPINGS = {
    "FluxAdaptiveInjector": FluxAdaptiveInjector,
    "PresenceSaver": PresenceSaver,
    "PresenceDirectorVertex": PresenceDirectorVertex,
    "PresenceDirectorFireworks": PresenceDirectorFireworks
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxAdaptiveInjector": "💉 Flux Adaptive Injector",
    "PresenceSaver": "💾 Presence Saver",
    "PresenceDirectorVertex": "🏭 Presence Director (Vertex AI)",
    "PresenceDirectorFireworks": "🔥 Presence Director (Fireworks AI)"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
