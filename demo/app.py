"""Gradio entry point for the K-FRAG demo.

This file intentionally does not contain a secret authentication key and does
not re-create missing legacy K-FRAG components. Wire your existing repository
implementation through build_runtime() before publishing the demo.
"""

import os
import gradio as gr


def build_runtime():
    """Import the existing K-FRAG demo runtime without altering the model.

    Set KFRAG_DEMO_MODULE to the module that exposes:
      - add_watermark(image, source_label)
      - verify_image(image)

    Example after you add a thin adapter to your existing repo:
        KFRAG_DEMO_MODULE=kfrag.demo_runtime
    """
    module_name = os.environ.get("KFRAG_DEMO_MODULE")
    if not module_name:
        raise RuntimeError(
            "KFRAG_DEMO_MODULE is not set. Point it to your existing K-FRAG "
            "runtime adapter. Do not put the authentication key in source code."
        )
    import importlib
    module = importlib.import_module(module_name)
    if not hasattr(module, "add_watermark") or not hasattr(module, "verify_image"):
        raise RuntimeError(
            f"{module_name} must expose add_watermark() and verify_image()."
        )
    return module


runtime = build_runtime()

with gr.Blocks(title="K-FRAG Provenance Verification") as demo:
    gr.Markdown("# K-FRAG\nFragment-authenticated image provenance verification")
    with gr.Tab("Add watermark"):
        image_in = gr.Image(type="pil", label="Input image")
        source_label = gr.Textbox(label="Source label")
        embed_button = gr.Button("Add K-FRAG watermark")
        image_out = gr.Image(type="pil", label="Watermarked image")
        embed_metadata = gr.JSON(label="Metadata")
        embed_button.click(
            fn=runtime.add_watermark,
            inputs=[image_in, source_label],
            outputs=[image_out, embed_metadata],
        )
    with gr.Tab("Verify fragment"):
        fragment_in = gr.Image(type="pil", label="Image or cropped fragment")
        verify_button = gr.Button("Verify")
        status = gr.Textbox(label="Status")
        verify_metadata = gr.JSON(label="Verification result")
        verify_button.click(
            fn=runtime.verify_image,
            inputs=[fragment_in],
            outputs=[status, verify_metadata],
        )


if __name__ == "__main__":
    demo.launch()
