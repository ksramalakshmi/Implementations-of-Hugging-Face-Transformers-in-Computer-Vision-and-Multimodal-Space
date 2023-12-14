from transformers import DPTImageProcessor, DPTForDepthEstimation
import torch
import numpy as np
from PIL import Image
import gradio as gr

processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")

def predict(img):
    print(img.shape)
    img = torch.from_numpy(img)
    inputs = processor(images=img, return_tensors="pt")
    outputs = model(**inputs)

    predicted_depth = outputs.predicted_depth
    
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=img.shape[:2],
        mode="bicubic",
        align_corners=False,
    )

    output = prediction.squeeze().cpu().detach().numpy()
    
    formatted = (output * 255 / np.max(output)).astype("uint8")
    depth_image = Image.fromarray(formatted)
    print("depth image:", type(depth_image))

    return depth_image

iface = gr.Interface(fn=predict, inputs="image", outputs="image")
iface.launch()