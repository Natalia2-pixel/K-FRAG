"""Template adapter for your existing K-FRAG runtime.

Copy this to an importable module in your repository and replace only the
imports/wrapping needed to call your already-validated functions. Do not change
model layers, thresholds, search parameters, or final fusion logic.
"""

# Example ONLY. Replace these imports with the actual module paths in your repo.
# from kfrag_existing_runtime import ui_add_watermark, ui_verify_image


def add_watermark(image, source_label):
    raise NotImplementedError(
        "Wire this function to the existing validated ui_add_watermark()."
    )


def verify_image(image):
    raise NotImplementedError(
        "Wire this function to the existing validated ui_verify_image()."
    )
