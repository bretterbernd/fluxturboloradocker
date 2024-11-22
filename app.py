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
    ).images[0]

    pipe.unfuse_lora()

    return out

js_function = """ () => {if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
            document.body.classList.toggle('dark');
}}
        """

title = """<h1 style="color:#f8f9fb; font-size: 3rem; line-height: 100%;" align="center"; >WORKSHOP KITeGG</h1>"""
 
# load kitegg-theme and init app
kitegg_theme = gr.Theme.from_hub("maidi/kitegg_llp_theme").set(
slider_color = "rgba(255, 160, 31, 0.5)",
input_background_fill_focus  = '#f8f9fb',
)

css = """
footer {visibility: hidden}
"""

flux_block = gr.Blocks(theme=kitegg_theme, css=css, js=js_function)

with flux_block:
    gr.HTML(title, min_height=60)
    with gr.Row():
        with gr.Column():
            t2i_prompt = gr.Textbox(lines=4, placeholder="put your image decription here...", label="👍 PROMPT", container=True)
            t2i_seed = gr.Number(label="\U0001F916 seed", value=42, precision=0)
            with gr.Group():
                t2i_height = gr.Slider(label="Image height", value=1024, minimum=768, maximum=1024, step=8)
                t2i_width = gr.Slider(label="Image width", value=768, minimum=768, maximum=1024, step=8)
            t2i_real = gr.Checkbox(label="Option for less uniform images", value=False)
            t2i_button = gr.Button(value="Generate Image", variant="primary")  
    with gr.Row():
        t2i_output_img = gr.Image(type="pil", label="OUTPUT IMAGE", interactive=False)    
    t2i_button.click(fn=t2i, inputs=[t2i_prompt, t2i_seed, t2i_height, t2i_width, t2i_real], outputs=[t2i_output_img])        

flux_block.queue()
flux_block.launch(server_port=8898, server_name="0.0.0.0", debug=True)




