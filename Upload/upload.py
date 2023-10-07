import streamlit as st
from streamlit_extras.colored_header import colored_header
from PIL import Image
from API.api_helper import upload_image_to_backend
from .preprocessing import ImagePreprocessing


class ImageUploader:
    def __init__(self):
        self.dataset_name = ""
        self.uploaded_files = []
        self.image_preprocessing = ImagePreprocessing()

    def display_uploaded_images(self):
        st.subheader("Uploaded Images:")

        # Calculating the number of rows based on the number of uploaded images
        num_rows = len(self.uploaded_files) // 4
        if len(self.uploaded_files) % 4 > 0:
            num_rows += 1

        # Iterate through the number of rows
        for row in range(num_rows):
            # Get the files for the current row
            files_for_row = self.uploaded_files[row * 4:row * 4 + 4]

            # Create columns for the current row
            cols = st.columns(4)

            # Display images in columns
            for i, uploaded_file in enumerate(files_for_row):
                cols[i].image(uploaded_file, caption=uploaded_file.name, use_column_width=True)

    def process_images(self):
        for uploaded_file in self.uploaded_files:
            image = Image.open(uploaded_file)
            encoded_image = self.image_preprocessing.encode_image(image)
            upload_image_to_backend(encoded_image, uploaded_file.name, self.dataset_name)

    def run(self):
        colored_header(
            label="Upload Images to the Database",
            description=None,
            color_name="violet-70",
        )

        st.info('**Upload:**\n'
                '1. Supported formats: JPG, PNG, JPEG\n'
                '2. You can upload multiple images\n'
                '3. The resizing is set to [224,224]. The normalization is on.')

        # Upload multiple images
        self.uploaded_files = st.file_uploader("", type=["jpg", "png", "jpeg"],
                                               accept_multiple_files=True)

        # Dataset name input
        self.dataset_name = st.text_input("Enter dataset name:", value=self.dataset_name)

        # Display uploaded images
        if self.uploaded_files:
            self.display_uploaded_images()

            if st.button("Process Images"):
                self.process_images()
                st.success("Images processed and sent to MongoDB!")
        # if st.button("Refresh Dataset List"):
        #     fetch_datasets()
