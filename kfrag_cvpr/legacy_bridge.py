"""Bridge to the existing K-FRAG base implementation.

The final 100K refiner depends on the earlier K-FRAG watermark encoder,
independent decoder, fusion model, spatial-search helpers, and strict recovery.
Those components must be imported from the existing repository rather than
reimplemented here, to avoid silently changing the CVPR model.
"""

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class LegacyKFRAGBridge:
    embed_full_spatial_image: Callable
    evaluation_source_id: Callable
    extract_raw_logits20: Callable
    old_fusion_logits20: Callable
    blind_crop_search_spatial_fast: Callable
    strict_recover_spatial_candidates: Callable
    place_crop_on_canvas: Callable
    split_16_regions: Callable
    v3_fragments_to_bits20: Callable
    device: Any

    def audit(self):
        missing = [
            name
            for name, value in self.__dict__.items()
            if value is None
        ]
        if missing:
            raise RuntimeError("Missing K-FRAG bridge objects: " + ", ".join(missing))
        return True
