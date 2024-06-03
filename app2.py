from flask import Flask, request, jsonify
from flask import Flask, render_template
import random
import uuid
from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler
import torch
import numpy as np
from PIL import Image

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

MAX_SEED = np.iinfo(np.int32).max

if torch.cuda.is_available():
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "sd-community/sdxl-flash",
        torch_dtype=torch.float16,
        use_safetensors=True,
        add_watermarker=False
    )
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda")

def save_image(img):
    unique_name = str(uuid.uuid4()) + ".png"
    img.save(unique_name)
    return unique_name

def randomize_seed_fn(randomize_seed: bool) -> int:
    if randomize_seed:
        return random.randint(0, MAX_SEED)
    return 1  # Valor padrão para seed se randomize_seed for False

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    prompt = data['prompt']
    seed = randomize_seed_fn(True)  # Define seed aleatoriamente
    generator = torch.Generator().manual_seed(seed)   

    negative_prompt = "(deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, (mutated hands and fingers:1.4), disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation, NSFW"

    options = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "width": 1024,
        "height": 1024,
        "guidance_scale": 3,
        "num_inference_steps": 7,
        "generator": generator,
        "use_resolution_binning": True,
        "output_type": "pil",
    }
    
    images = pipe(**options).images

    image_paths = [save_image(img) for img in images]
    return jsonify({'results': image_paths})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)