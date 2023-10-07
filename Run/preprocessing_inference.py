import onnxruntime
import numpy as np
from PIL import Image
from scipy.special import softmax
import json
import io, base64


class OnnxInference:
    def __init__(self, model_path, class_labels_path):
        self.session = onnxruntime.InferenceSession(model_path)
        with open(class_labels_path, "r") as f:
            self.class_labels = json.load(f)

    def preprocess_image(self, image_path):
        height = 224
        width = 224
        channels = 3  # Typically RGB

        image = Image.open(image_path)
        image = image.resize((width, height), Image.ANTIALIAS)
        image_data = np.asarray(image).astype(np.float32)
        image_data = image_data.transpose([2, 0, 1])  # transpose to CHW format

        # Mean and Standard Deviation adjustments
        mean = np.array([0.079, 0.05, 0]) + 0.406
        std = np.array([0.005, 0, 0.001]) + 0.224
        for channel in range(channels):
            image_data[channel, :, :] = (image_data[channel, :, :] / 255 - mean[channel]) / std[channel]

        image_data = np.expand_dims(image_data, 0)
        return image_data

    def run_inference(self, uploaded_image):
        processed_image = self.preprocess_image(uploaded_image)
        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name
        result = self.session.run([output_name], {input_name: processed_image})

        logits = result[0][0]
        probabilities = softmax(logits)
        class_idx = np.argmax(probabilities)
        confidence_score = probabilities[class_idx] * 100
        predicted_class = self.class_labels[class_idx]

        return predicted_class, confidence_score, logits.tolist()


    def decode_image(slef,encoded_image_data):
        image_bytes = base64.b64decode(encoded_image_data)
        return Image.open(io.BytesIO(image_bytes))

    # def extract_embeddings(self, uploaded_image):
    #     processed_image = self.preprocess_image(uploaded_image)
    #     input_name = self.session.get_inputs()[0].name
    #     output_name = self.session.get_outputs()[0].name
    #     result = self.session.run([output_name], {input_name: processed_image})
    #
    #     return result[0][0]
