import base64
import io

import requests
import pickle

from PIL import Image
import io
from PIL import Image
from torchvision import transforms


def upload_image_to_backend(encoded_image, filename, dataset_name):
    response = requests.post("http://flask_api:5003/api/upload_image",
                             json={"image": encoded_image, "name": filename,
                                   "dataset_name": dataset_name})
    return response


def push_results_to_db(results):
    for result in results:
        requests.post("http://flask_api:5003/api/update_results", json=result)


def fetch_datasets():
    response = requests.get("http://flask_api:5003/api/get_datasets")
    datasets = response.json()
    return datasets


def fetch_images(dataset_name):
    response = requests.get("http://flask_api:5003/api/get_images", params={"dataset_name": dataset_name})
    return response.json()


def fetch_image_info(dataset_name):
    response = requests.get("http://flask_api:5003/api/get_image_info", params={"dataset_name": dataset_name})
    return response.json()


def fetch_image_data_and_embeddings(image_name, dataset_name):
    # Fetch the actual image data using a direct HTTP call
    image_data_response = requests.get("http://flask_api:5003/api/get_images", params={"dataset_name": dataset_name})
    image_data_list = image_data_response.json()

    for data in image_data_list:
        if data["name"] == image_name:
            image_data = data["data"]
            break
    else:
        return None  # Return None if no image data is found

    # Fetch the image info (including embeddings) using a direct HTTP call
    image_info_response = requests.get("http://flask_api:5003/api/get_image_info",
                                       params={"dataset_name": dataset_name})
    image_info_list = image_info_response.json()

    for info in image_info_list:
        if info["name"] == image_name:
            embeddings = info["embeddings"]
            return {
                "data": image_data,
                "embeddings": embeddings
            }

    return None





def fetch_all_image_data(dataset_name):
    """
    Fetch all information related to images in the given dataset.

    Args:
    - dataset_name (str): Name of the dataset for which to fetch the information.

    Returns:
    - list: List of dictionaries where each dictionary contains all the information for a particular image.
    """
    # Fetch raw image data
    image_data_list = fetch_images(dataset_name)

    # Fetch image info
    image_info_list = fetch_image_info(dataset_name)

    # Create a lookup for image info based on image name
    info_lookup = {info['name']: info for info in image_info_list}

    all_image_data = []
    for image_data in image_data_list:
        # Get corresponding info for the current image
        image_info = info_lookup.get(image_data["name"], {})

        # Merge image data and image info
        combined_data = {**image_data, **image_info}
        all_image_data.append(combined_data)

    return all_image_data



def update_selected_image_info(image_name, dataset_name, new_dataset_name, new_class_name):
    """
    Updates the specified image's dataset and class in the database.

    Args:
    - image_name (str): Name of the image to update.
    - dataset_name (str): Current name of the dataset containing the image.
    - new_dataset_name (str): New name of the dataset.
    - new_class_name (str): New class name for the image.

    Returns:
    - Response from the API.
    """
    data = {
        "name": image_name,
        "dataset_name": dataset_name,
        "new_dataset_name": new_dataset_name,
        "new_class_name": new_class_name,
        "confidence_score": 0
    }
    response = requests.post("http://flask_api:5003/api/update_selected_image_info", json=data)
    return response

## Push images<-> npz to db
def upload_large_npz_to_backend(npz_file_path):
    with open(npz_file_path, "rb") as file:
        response = requests.post("http://flask_api:5003/api/upload_large_npz", files={"file": file})
    return response

## Visua
# def fetch_npz_names():
#     # Implement the function to fetch available .npz file names from your database/API
#     response = requests.get("http://flask_api:5003/api/get_npz_names")
#     if response.status_code == 200:
#         return response.json()  # Assuming the server returns a list of .npz names
#     else:
#         raise Exception(f"Error {response.status_code}: {response.text}")
#
# def fetch_npz_data(file_name):
#     response = requests.get("http://flask_api:5003/api/retrieve_data/{file_name}")
#     if response.status_code == 200:
#         serialized_data = response.content
#         data = pickle.loads(serialized_data)
#         return data
#     else:
#         raise Exception(f"Error {response.status_code}: {response.text}")


