from .archive import NODE_CLASS_MAPPINGS as ARCHIVE_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as ARCHIVE_DISPLAY_MAPPINGS
from .unaliver_nodes import FluxAdaptiveInjector
from .presence_node import PresenceDirector, PresenceSaver
from .presence_vertex_node import PresenceDirectorVertex

NODE_CLASS_MAPPINGS = {
    **ARCHIVE_MAPPINGS,
    "FluxAdaptiveInjector": FluxAdaptiveInjector,
    "PresenceDirector": PresenceDirector,
    "PresenceDirectorVertex": PresenceDirectorVertex,
    "PresenceSaver": PresenceSaver
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **ARCHIVE_DISPLAY_MAPPINGS,
    "FluxAdaptiveInjector": "Flux Adaptive Injector 💉",
    "PresenceDirector": "Presence Director 🏭",
    "PresenceDirectorVertex": "Presence Director (Vertex AI) 🚀",
    "PresenceSaver": "Presence Saver 💾"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
