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

adapter_id = "alimama-creative/FLUX.1-Turbo-Alpha"
pipe.load_lora_weights(adapter_id, adapter_name="turbo")


# Function to handle text-to-image generation
def t2i(prompt, seed, height, width, real):
    if real:
        pipe.set_adapters(["turbo", "real"], adapter_weights=[1.0, 1.0])
        prompt = "shot on a mobile phone, " + prompt
    else:
        pipe.set_adapters(["turbo", "real"], adapter_weights=[1.0, 0.0])

    pipe.fuse_lora()

    torch.manual_seed(seed)
    out = pipe(
        prompt=prompt,
        guidance_scale=3.5,
        height=int(height),
        width=int(width),
        num_inference_steps=8,
        max_sequence_length=512,
    )[0]

    pipe.unfuse_lora()

    return out

# Function to handle file uploads and dynamic LoRA loading
def handle_file_upload(files, lora_weight):
    uploaded_files = []
    for file in files:
        uploaded_files.append(file.name)
        # Dynamically load the uploaded LoRA file into the pipeline
        pipe.load_lora_weights(file.name, adapter_name="user_uploaded_lora")
    
    # Adjust weight for the uploaded LoRA
    pipe.set_adapters(["turbo", "user_uploaded_lora"], adapter_weights=[1.0, lora_weight])
    
    return uploaded_files

# Function to update the weight of the uploaded LoRA dynamically
def update_lora_weight(weight_slider_value):
    # Update the weight of the LoRA
    pipe.set_adapters(["turbo", "user_uploaded_lora"], adapter_weights=[1.0, weight_slider_value])

# HTML and CSS for the interface
js_function = """ () => {if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
            document.body.classList.toggle('dark');
}}
        """

title = """<h1 style="color:#f8f9fb; font-size: 3rem; line-height: 100%;" align="center"; >WORKSHOP KITeGG</h1>"""

flux_block = gr.Blocks(css=css, js=js_function)

with flux_block:
    gr.HTML(title, min_height=60)
    
    # Existing content for image generation
    with gr.Row():
        with gr.Column():
            t2i_prompt = gr.Textbox(lines=4, placeholder="put your image description here...", label="👍 PROMPT", container=True)
            t2i_seed = gr.Number(label="\U0001F916 seed", value=42, precision=0)
            with gr.Group():
                t2i_height = gr.Slider(label="Image height", value=1024, minimum=768, maximum=1024, step=8)
                t2i_width = gr.Slider(label="Image width", value=768, minimum=768, maximum=1024, step=8)
            t2i_real = gr.Checkbox(label="Option for less uniform images", value=False)
            t2i_button = gr.Button(value="Generate Image", variant="primary")  
    
    with gr.Row():
        t2i_output_img = gr.Image(type="pil", label="OUTPUT IMAGE", interactive=False)    
    t2i_button.click(fn=t2i, inputs=[t2i_prompt, t2i_seed, t2i_height, t2i_width, t2i_real], outputs=[t2i_output_img])        

    # New content for file upload and listing uploaded files
    gr.HTML("<h2 style='color:#f8f9fb;'>Upload a *.safetensors LoRA File</h2>")
    file_upload = gr.File(label="Upload LoRA File", file_count="multiple", file_types=[".safetensors"])
    uploaded_files_list = gr.Textbox(label="Uploaded LoRA Files", interactive=False)
    
    # Slider for adjusting the weight of the uploaded LoRA
    weight_slider = gr.Slider(label="LoRA Weight", value=1.0, minimum=0.0, maximum=2.0, step=0.1)
    
    file_upload.change(fn=handle_file_upload, inputs=[file_upload, weight_slider], outputs=[uploaded_files_list])
    weight_slider.change(fn=update_lora_weight, inputs=[weight_slider], outputs=[])

flux_block.queue()
flux_block.launch(server_port=8898, server_name="0.0.0.0", debug=True)
