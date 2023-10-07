import numpy as np
import onnxruntime as ort
from PIL import Image
import json
from torchvision import transforms

# Load the labels
with open('/Users/apagnoux/PycharmProjects/webapp-streamlit-upgrade/imagenet-simple-labels.json', 'r') as f:
    labels = json.load(f)

# Preprocess the image
input_image = Image.open('/Users/apagnoux/Documents/animals/puppy.jpeg')
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0).numpy()  # Convert tensor to numpy array

# Load the ONNX model
session = ort.InferenceSession("/Users/apagnoux/PycharmProjects/webapp-streamlit-upgrade/mobilenetv2-12-int8.onnx")

# Get the name of the input node
input_name = session.get_inputs()[0].name

# Run inference
outputs = session.run(None, {input_name: input_batch})

# Get the result (it's a list with one element for our case)
output = outputs[0]

# The output has unnormalized scores. To get probabilities, you can use softmax.
probabilities = np.exp(output[0]) / np.sum(np.exp(output[0]))

# Print the top-5 predicted classes
top5_idx = np.argsort(probabilities)[-5:][::-1]
for i in top5_idx:
    print(f"Label: {labels[i]}, Probability: {probabilities[i] *100 :.2f}%")
