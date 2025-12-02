import gradio as gr
import os
from PIL import Image
from agent_vto import VTOAgent, build_llm

# ---------------------------------------------------
# 0) LOAD ENVIRONMENT VARIABLES
# ---------------------------------------------------
model_name = os.environ.get("VTO_LLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
token = os.environ.get("VTO_LLM_TOKEN")  # Must be set by user
lora_path = os.environ.get("VTO_LORA_PATH", "ckpts/lora.safetensors")
output_dir = os.environ.get("VTO_OUTPUT_DIR", "outputs")

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Initialize LLM + VTO Agent
llm = build_llm(model_name, token)
agent = VTOAgent(llm, lora_path)


# ---------------------------------------------------
# 1) TRY-ON FUNCTION
# ---------------------------------------------------
def tryon_api(person_img, cloth_img):
    if person_img is None or cloth_img is None:
        return "Please upload both images.", None

    user_path = os.path.join(output_dir, "user_image.png")
    cloth_path = os.path.join(output_dir, "cloth_image.png")
    final_output = os.path.join(output_dir, "final_tryon.png")

    # Clean previous outputs
    if os.path.exists(final_output):
        os.remove(final_output)

    # Save uploaded images
    person_img.save(user_path)
    cloth_img.save(cloth_path)

    try:
        response, _ = agent.handle_input(
            user_input="perform try-on",
            user_img_path=user_path,
            garment_img_path=cloth_path
        )
    except Exception as e:
        return f"Agent Error: {e}", None

    # Return final image if exists
    if os.path.exists(final_output):
        final_img = Image.open(final_output)
        return f"Done! {response}", final_img
    else:
        return f"Try-on failed. Log: {response}", None


# ---------------------------------------------------
# 2) SMART AGENT FUNCTION
# ---------------------------------------------------
def agent_api(user_text):
    if not user_text:
        return "Type something...", []

    try:
        response, images = agent.handle_input(user_input=user_text)

        # Convert returned images to gallery URLs or file paths
        gallery_urls = []
        if images:
            for img in images:
                if isinstance(img, dict) and 'url' in img:
                    gallery_urls.append(img['url'])
                elif isinstance(img, str):
                    gallery_urls.append(img)

        return response, gallery_urls

    except Exception as e:
        return f"Error: {e}", []


# ---------------------------------------------------
# 3) GRADIO UI DESIGN
# ---------------------------------------------------
my_style = """
<style>
body {background-color: #f9fafb;}
button.secondary {background-color: #ecfeff !important; color: #0e7490 !important; border: 1px solid #0e7490 !important;}
.styled-header {
    background-color: #0d9488; 
    color: white;              
    font-weight: bold;
    font-size: 1.2em;
    margin-bottom: 10px;
    text-align: center;
    padding: 10px;            
    border-radius: 8px;        
}</style>
"""

with gr.Blocks() as demo:

    gr.HTML(my_style)

    # Header
    gr.HTML("""
        <div style='text-align:center; font-family:"Poppins", sans-serif; margin-bottom: 30px; padding: 20px; background-color: white; border-radius: 10px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);'>
            <h1 style='color:#0d9488; font-size:2.5em; font-weight: bold; margin:0;'> Virtual Try-On + Smart Agent </h1>
            <p style='color:#666; font-size:1.1em; margin-top:10px;'>
                Upload your photo & a clothing item, or ask the AI Fashion Assistant!
            </p>
        </div>
    """)

    # --- Image Upload Section ---
    with gr.Row():
        with gr.Column():
            gr.HTML("<div class='styled-header'>Upload Person Image</div>")
            person_in = gr.Image(type="pil", sources=["upload"], height=350)

        with gr.Column():
            gr.HTML("<div class='styled-header'>Upload Clothing Image</div>")
            cloth_in = gr.Image(type="pil", sources=["upload"], height=350)

    # Button
    tryon_btn = gr.Button("Generate Try-On Image", variant="secondary")

    # Output Section
    with gr.Row():
        with gr.Column(): pass
        with gr.Column(scale=2):
            gr.HTML("<div class='styled-header' style='margin-top: 20px;'>Try-On Result</div>")
            result_out = gr.Image(type="pil", interactive=False)
            status_out = gr.Textbox(label="Status", lines=1)
        with gr.Column(): pass

    tryon_btn.click(
        fn=tryon_api,
        inputs=[person_in, cloth_in],
        outputs=[status_out, result_out]
    )

    gr.HTML("<hr style='margin: 40px 0; border: 0; border-top: 2px solid #e5e7eb;'>")

    gr.HTML("<div class='styled-header'>Ask the Smart Agent</div>")

    user_text = gr.Textbox(label="Enter your query", placeholder="Ask about fashion advice...")
    agent_btn = gr.Button("Ask Agent", variant="secondary")

    gr.HTML("<div class='styled-header' style='margin-top: 20px;'>Agent Response</div>")
    output_text = gr.Textbox(show_label=False, lines=4)
    image_gallery = gr.Gallery(columns=5, height="auto")

    agent_btn.click(
        fn=agent_api,
        inputs=[user_text],
        outputs=[output_text, image_gallery]
    )

demo.launch(share=True, debug=True)
