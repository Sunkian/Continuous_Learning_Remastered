import numpy as np
import io
from PIL import Image
import base64


class ImagePreprocessing:
    def encode_image(self, image):
        image_bytes = io.BytesIO()
        image.save(image_bytes, format="PNG")
        return base64.b64encode(image_bytes.getvalue()).decode()
