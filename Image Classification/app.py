import gradio as gr
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image

processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

def predict(img):
    inputs = processor(images=img, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    return model.config.id2label[predicted_class_idx]

iface = gr.Interface(fn=predict, inputs="image", outputs="label")
iface.launch()