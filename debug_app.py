#!/usr/bin/env python3
"""
Ultra-simple Gradio interface to isolate the JavaScript error - Fixed for Gradio 6.x.
"""

import gradio as gr

def respond(message, history):
    print(f"Received: {message}, history: {history}")  # Debug
    if history is None:
        history = []
    
    if not message or not message.strip():
        return history
    
    # Simple response
    response = f"You said: {message}"
    
    # Return as tuples for chatbot (only chatbot output)
    history.append([message, response])
    return history

with gr.Blocks() as demo:
    gr.Markdown("# Simple Test")
    
    chatbot = gr.Chatbot(label="Chat", height=300)
    msg = gr.Textbox(label="Message", placeholder="Type something...")
    clear = gr.Button("Clear")
    
    msg.submit(respond, [msg, chatbot], [chatbot])
    clear.click(lambda: [], None, [chatbot], queue=False)

if __name__ == "__main__":
    demo.launch(server_port=7862, server_name="0.0.0.0", show_error=True)