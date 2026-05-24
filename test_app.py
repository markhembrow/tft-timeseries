#!/usr/bin/env python3
"""
Minimal Gradio app to test basic functionality.
"""

import gradio as gr

def greet(name):
    return f"Hello {name}!"

with gr.Blocks() as demo:
    gr.Markdown("# Hello World")
    name = gr.Textbox(label="Name")
    output = gr.Textbox(label="Greeting")
    name.submit(greet, inputs=name, outputs=output)

if __name__ == "__main__":
    demo.launch(server_port=7862, server_name="0.0.0.0")