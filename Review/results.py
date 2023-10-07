import streamlit as st
import requests
from PIL import Image
import io
import base64
from API.api_helper import fetch_datasets,  fetch_all_image_data, update_selected_image_info
import tempfile
import os


class Result:

    def __init__(self):
        # Initialize the current page to 0
        if 'current_page' not in st.session_state:
            st.session_state.current_page = 0

            # Initialize selected images dict if it doesn't exist
        if 'selected_images' not in st.session_state:
            st.session_state.selected_images = {}

    def display_dataset_selection(self):
        datasets = fetch_datasets()
        return st.selectbox("Select a dataset:", datasets)

    def display_images(self, selected_dataset):
        # all_image_data = fetch_all_image_data(selected_dataset)

        with st.spinner('Loading...'):
            all_image_data = fetch_all_image_data(selected_dataset)

        # filtered_image_data = [data for data in all_image_data if data.get('class', 'Not Available') != 'Not Available' and data.get('score', 0) >= 80]
        filtered_image_data = [
            data for data in all_image_data
            if data.get('class', 'N/A') != 'Not Available' and float(data.get('score', '0')) <= 80
        ]

        images_per_row = 4
        rows_per_page = 5
        start_idx = st.session_state.current_page * images_per_row * rows_per_page
        end_idx = start_idx + images_per_row * rows_per_page

        current_batch = filtered_image_data[start_idx:end_idx]


            ## Select All
        if 'select_all' not in st.session_state or 'new_page_loaded' in st.session_state and st.session_state.new_page_loaded:
            st.session_state.select_all = False
            st.session_state.new_page_loaded = False  # Reset the new_page_loaded flag

        st.session_state.select_all = st.checkbox('Select All', value=st.session_state.select_all)

        for data in current_batch:
            image_name = data['name']
            if st.session_state.select_all:
                st.session_state.selected_images[image_name] = True
            else:
                st.session_state.selected_images[image_name] = False

        num_images = len(current_batch)
        num_rows = (num_images + images_per_row - 1) // images_per_row
        fixed_image_size = (224, 224)

        for i in range(num_rows):
            row_data = current_batch[i * images_per_row: (i + 1) * images_per_row]
            cols = st.columns(images_per_row)

            for col, data in zip(cols, row_data):
                image_name = data['name']
                is_selected = col.checkbox('Select', value=st.session_state.selected_images[image_name], key=image_name)
                st.session_state.selected_images[image_name] = is_selected

                image = self.decode_image(data["data"]).resize(fixed_image_size)
                col.image(image)

                predicted_class = data.get('class', 'N/A')
                confidence_score = data.get('score', 'N/A')
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
                                    <h6 style="margin: 0;">{data['name']}</h6>
                                    <p><strong>Predicted Class:</strong> {predicted_class}</p>
                                    <p><strong>Confidence Score:</strong> {confidence_score:.2f}%</p>
                                </div>
                                """, unsafe_allow_html=True)


        # Display navigation buttons
        total_pages = (len(filtered_image_data) + images_per_row * rows_per_page - 1) // (
                    images_per_row * rows_per_page)
        col1, col2 = st.columns(2)  # Split the buttons into two columns for better alignment

        with col1:
            if st.session_state.current_page > 0:
                if st.button('Previous', key='prev_button'):  # Unique key for Previous button
                    st.session_state.current_page -= 1
                    st.session_state.new_page_loaded = True
                    st.experimental_rerun()  # Refresh the Streamlit app after changing the page number

        with col2:
            if st.session_state.current_page < total_pages - 1:
                if st.button('Next', key='next_button'):  # Unique key for Next button
                    st.session_state.current_page += 1
                    st.session_state.new_page_loaded = True
                    st.experimental_rerun()

# @staticmethod
    def decode_image(self,encoded_image_data):
        image_bytes = base64.b64decode(encoded_image_data)
        return Image.open(io.BytesIO(image_bytes))

    def results_main(self):
        st.title("Review Results")
        st.subheader("Update Image Information")

        # Split the input fields into two columns
        col1, col2 = st.columns(2)

        # Input fields for dataset name
        with col1:
            new_dataset_name = st.text_input("Enter new dataset name:")

        # Input fields for class name
        with col2:
            new_class_name = st.text_input("Enter new class name:")

        # Continue with the previous code
        selected_dataset = self.display_dataset_selection()

        if selected_dataset:
            # st.success(f"✅ Dataset Selected: {selected_dataset}")
            self.display_images(selected_dataset)

            # Add a 'Send' button to update the database
            if st.button("Send"):
                for image_name, selected in st.session_state.selected_images.items():
                    if selected:
                        # Call the API to update the database
                        response = update_selected_image_info(
                            image_name, selected_dataset, new_dataset_name, new_class_name)
                        if response.status_code == 200:
                            st.success(f"Updated info for {image_name}!")
                        else:
                            st.error(f"Failed to update {image_name}.")