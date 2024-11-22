import torch
from diffusers import AutoencoderTiny, FluxPipeline
from PIL import Image
import time
import gradio as gr
import os 
from huggingface_hub import login



login(token = os.environ.get('HF_TOKEN'))

dtype = torch.bfloat16
device = "cuda"
base_model = "black-forest-labs/FLUX.1-dev"

taef1 = AutoencoderTiny.from_pretrained("madebyollin/taef1", torch_dtype=dtype).to(device)
pipe = FluxPipeline.from_pretrained(base_model, torch_dtype=dtype, vae=taef1).to(device)

# wget https://civitai.com/api/download/models/1026423 --content-disposition

adapter_id = "alimama-creative/FLUX.1-Turbo-Alpha"
pipe.load_lora_weights(adapter_id, adapter_name="turbo")
pipe.load_lora_weights("./lora/", weight_name="UltraRealPhoto.safetensors", adapter_name="real")





# Updated placeholder function for image generation
def t2i(prompt, seed, height, width, add_lora, lora, lora_weight, auto_seed):
    # If auto_seed is enabled, generate a random seed
    if auto_seed:
        seed = random.randint(0, 999999)
        print(f"[DEBUG] Auto-generate seed enabled. Using seed: {seed}")
    else:
        print(f"[DEBUG] Using fixed seed: {seed}")

    print(f"[DEBUG] Generating image with parameters:\n"
          f"  Prompt: {prompt}\n"
          f"  Seed: {seed}\n"
          f"  Height: {height}\n"
          f"  Width: {width}\n"
          f"  Add LoRA: {add_lora}\n"
          f"  Selected LoRA: {lora}\n"
          f"  LoRA Weight: {lora_weight}")
    try:
        torch.manual_seed(seed)
        img = Image.new("RGB", (width, height), color="gray")  # Placeholder gray image
        if add_lora:
            print(f"[DEBUG] Applying LoRA '{lora}' with weight {lora_weight}.")
        else:
            print("[DEBUG] No LoRA applied.")
        print("[DEBUG] Image generation successful (placeholder).")
        return img, seed  # Return the seed back to the UI
    except Exception as e:
        print(f"[DEBUG] Error during image generation: {e}")
        return None, seed  # Return the seed back to the UI on error

# JS function for toggling dark mode
js_function = """ () => {if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
            document.body.classList.toggle('dark');
}}
"""

title = """<h1 style="color:#f8f9fb; font-size: 3rem; line-height: 100%;" align="center"; >MARKUZ TURBO PIPELINE FORK</h1>"""

# Define Gradio interface without custom theme or CSS
flux_block = gr.Blocks(css=js_function)

with flux_block:
    gr.HTML(title, min_height=60)
    with gr.Row():
        with gr.Column():
            t2i_prompt = gr.Textbox(lines=4, placeholder="put your image description here...", label="👍 PROMPT", container=True)
            t2i_seed = gr.Number(label="\U0001F916 Seed", value=42, precision=0)
            t2i_auto_seed = gr.Checkbox(label="Auto-generate new seed", value=False)
            with gr.Group():
                t2i_height = gr.Slider(label="Image height", value=1024, minimum=768, maximum=1024, step=8)
                t2i_width = gr.Slider(label="Image width", value=256, minimum=256, maximum=1024, step=8)
            t2i_add_lora = gr.Checkbox(label="Add custom LoRa", value=False)
            t2i_lora = gr.Dropdown(choices=["LoRA 1", "LoRA 2", "LoRA 3"], label="LoRA Selection", value="LoRA 1")
            t2i_lora_weight = gr.Slider(label="LoRA Weight", minimum=-1, maximum=2, value=1.0, step=0.1)
            t2i_button = gr.Button(value="Generate Image", variant="primary")
    with gr.Row():
        t2i_output_img = gr.Image(type="pil", label="OUTPUT IMAGE", interactive=False)

    # Click event for generating image
    def debug_t2i(prompt, seed, height, width, add_lora, lora, lora_weight, auto_seed):
        # Pass all parameters to the t2i function
        img, new_seed = t2i(prompt, seed, height, width, add_lora, lora, lora_weight, auto_seed)
        return img, new_seed  # Return both image and updated seed

    t2i_button.click(
        fn=debug_t2i,
        inputs=[t2i_prompt, t2i_seed, t2i_height, t2i_width, t2i_add_lora, t2i_lora, t2i_lora_weight, t2i_auto_seed],
        outputs=[t2i_output_img, t2i_seed]  # Update the seed in the UI as well
    )

flux_block.queue()
print("[DEBUG] Launching Gradio interface...")
flux_block.launch(server_port=8898, server_name="0.0.0.0", debug=True)

