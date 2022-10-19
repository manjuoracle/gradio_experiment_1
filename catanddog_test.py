import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

mean = np.array([0.5, 0.5, 0.5])
std = np.array([0.25, 0.25, 0.25])

model = torch.load("/workspace/gradio_experiment_1/flagged/data_catanddog_train/model_catanddog/catdogmodel")
model.eval()
from PIL import Image
test1= Image.open("/workspace/gradio_experiment_1/flagged/data_catanddog_train/cats_and_dogs_filtered/cat_test.jpeg")
t=transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
t1 = t(test1)

t1 = t1.view(1, 3, 224 ,224)

output = model(t1)
output= torch.argmax(output)
if output.item() == 0:
  print("cat")
else:
  print("dog")

classes = ['cat','dog']

#gradio test
import gradio as gr

def catdog(image):
  if image is None:
    return None
  #config = resolve_data_config({}, model=model)
  #transform = create_transform(**config)
  image_tensor = transforms.ToTensor()(image).unsqueeze(0)

  preds = model(image_tensor)
  preds = preds[0].tolist()
  label_pred = {classes[i]: preds[i] for i in range(2)}
  return label_pred
  #torch.nn.functional.softmax(model(imagep)[0], dim=0)
    #confidences = {labels[i]: float(prediction[i]) for i in range(1)} 
    #confidences = {labels[i]: float(prediction[i]) for i in range(1)}    
  #return confidences

demo = gr.Interface(fn=catdog, inputs=gr.Image(type='pil'), outputs=gr.Label(num_top_classes=2),)
demo.launch(server_name="0.0.0.0", share=True) 
