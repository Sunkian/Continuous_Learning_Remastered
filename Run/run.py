import streamlit as st
from PIL import Image
import os
import base64
import io
from API.api_helper import push_results_to_db, fetch_datasets, fetch_images
from .preprocessing_inference import OnnxInference
import streamlit.elements as ste
import tempfile

class RunInference:

    def __init__(self):
        self.model_path = os.path.join(os.path.dirname(__file__), "../mobilenetv2-12-int8.onnx")
        self.class_labels_path = os.path.join(os.path.dirname(__file__), "../imagenet-simple-labels.json")
        self.mobilenet_inference = OnnxInference(self.model_path, self.class_labels_path)

    def display_dataset_selection(self):
        datasets = fetch_datasets()
        return st.selectbox("Select a dataset:", datasets)


    def display_images_and_run_inference(self, images_data, selected_dataset):
        results = []
        images_per_row = 4
        num_images = len(images_data)
        num_rows = (num_images + images_per_row - 1) // images_per_row
        fixed_image_size = (224, 224)

        for i in range(num_rows):
            row_images = images_data[i * images_per_row: (i + 1) * images_per_row]
            cols = st.columns(images_per_row)

            for col, image_data in zip(cols, row_images):
                image = self.decode_image(image_data["data"]).resize(fixed_image_size)
                col.image(image, use_column_width=True)

                # Save the resized image temporarily and get its path
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_image:
                    image.save(temp_image)
                    temp_image_path = temp_image.name

                # Run the inference using the path of the temporary image
                predicted_class, confidence_score, embeddings = self.mobilenet_inference.run_inference(temp_image_path)
                print(embeddings)

                # Cleanup the temporary image
                os.remove(temp_image_path)

                # Remaining part is unchanged
                card_style = """
                padding: 16px;
                background-color: #f1f1f1;
                border-radius: 4px;
                width: 100%;
                text-align: center;
                height: 180px;
                overflow-y: auto;
                overflow: auto;
                margin-bottom: 20px;
                """
                with col.container():
                    col.markdown(f"""
                    <div style="{card_style}">
                        <h6 style="margin: 0;">{image_data['name']}</h6>
                        <p><strong>Predicted Class:</strong> {predicted_class}</p>
                        <p><strong>Confidence Score:</strong> {confidence_score:.2f}%</p>
                    </div>
                    """, unsafe_allow_html=True)

                results.append({
                    "name": image_data["name"],
                    "dataset_name": selected_dataset,
                    "class": predicted_class,
                    "score": confidence_score,
                    "embeddings": embeddings
                })

        return results

    @staticmethod
    def decode_image(encoded_image_data):
        image_bytes = base64.b64decode(encoded_image_data)
        return Image.open(io.BytesIO(image_bytes))

    def main(self):
        # Title and subtitle
        st.title("MobileNet ONNX Inference")
        st.info('**Inference:**\n'
                '1. Select a dataset to run the inference on\n'
                '2. Click on the "Run Inference" button to start the inference process on the selected dataset\n')

        # Dataset Selection
        st.subheader("Select Dataset")
        selected_dataset = self.display_dataset_selection()

        if selected_dataset:
            # st.success(f"✅ Dataset Selected: {selected_dataset}")

            # Fetching images from the dataset
            images_data = fetch_images(selected_dataset)

            if not images_data:
                st.warning("❗ No images found in the selected dataset.")
                return

            # Instructions and button for inference
            st.subheader("Run Inference")

            start_inference = st.button("▶️ Run Inference")

            if start_inference:
                with st.spinner("Running Inference..."):
                    results = self.display_images_and_run_inference(images_data, selected_dataset)
                if results:
                    st.markdown("<br><br>", unsafe_allow_html=True)  # This will add some space
                    st.info('The resizing is currently set to [224,224]. The normalization is on.', icon="📐️")
                    st.success("Inference completed successfully!")
                    push_results_to_db(results)

                else:
                    st.warning("No results obtained from the inference.")