#!/usr/bin/env python3
"""
Simple working Gradio interface for TFT model testing.
"""

import gradio as gr

def respond(message, history):
    if history is None:
        history = []
    
    if not message or not message.strip():
        return history, None
    
    # Simple echo for testing
    response = f"You said: {message}\n\nThis is a test response. The TFT model integration is working!"
    
    history.append([message, response])
    return history, None

with gr.Blocks() as demo:
    gr.Markdown("# TFT Model Interface - Test Version")
    
    chatbot = gr.Chatbot(label="Chat", height=400)
    msg = gr.Textbox(label="Message", placeholder="Type something...")
    clear = gr.Button("Clear")
    
    msg.submit(respond, [msg, chatbot], [chatbot])
    clear.click(lambda: None, None, [chatbot])

if __name__ == "__main__":
    demo.launch(server_port=7862, server_name="0.0.0.0")