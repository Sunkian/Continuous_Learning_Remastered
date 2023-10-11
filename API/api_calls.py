from flask import Flask, request, jsonify
from pymongo import MongoClient
import streamlit as st
from bson.binary import Binary
import pickle
import numpy as np
from io import BytesIO
from flask import Response, send_file
import json
import io

app = Flask(__name__)

# MongoDB connection
client = MongoClient("mongodb://mongodb:27017/")
db = client["image_db"]
collection = db["images"]

collection2 = db["embeddings"]

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


app.json_encoder = NumpyEncoder



# Define API endpoint to receive and store image
@app.route("/api/upload_image", methods=["POST"])
def upload_image():
    data = request.json
    image_data = data["image"]
    image_name = data["name"]
    dataset_name = data["dataset_name"]
    predicted_class = "Not Available"
    confidence_score = 0.0

    # Store the image data in MongoDB with dataset name
    image_document = {
        "name": image_name,
        "dataset": dataset_name,
        "data": image_data,
        "class": predicted_class,
        "score": confidence_score,
        "embeddings": []
    }
    collection.insert_one(image_document)

    return jsonify({"message": "Image stored in MongoDB!"})

@app.route("/api/update_results", methods=["POST"])
def update_results():
    data = request.json
    image_name = data["name"]
    dataset_name = data["dataset_name"]
    predicted_class = data["class"]
    confidence_score = data["score"]
    embeddings = data["embeddings"]

    # Find the image in the database and update its class and score
    query = {"name": image_name, "dataset": dataset_name}
    # update_values = {"$set": {"class": predicted_class, "score": confidence_score}}
    update_values = {
        "$set": {
            "class": predicted_class,
            "score": confidence_score,
            "embeddings": embeddings
        }
    }
    collection.update_one(query, update_values)


    return jsonify({"message": "Review updated in MongoDB!"})

# Define API endpoint to get list of datasets
@app.route("/api/get_datasets", methods=["GET"])
def get_datasets():
    datasets = collection.distinct("dataset")
    return jsonify(datasets)


@app.route("/api/get_images", methods=["GET"])
def get_images():
    dataset_name = request.args.get("dataset_name")
    images = collection.find({"dataset": dataset_name})

    images_data = []
    for image in images:
        image_data = {
            "name": image["name"],
            "data": image["data"]
        }
        images_data.append(image_data)

    return jsonify(images_data)

@st.cache
@app.route("/api/get_image_info", methods=["GET"])
def get_image_info():
    dataset_name = request.args.get("dataset_name")
    # Exclude the "data" field from the query projection
    image_info = list(collection.find({"dataset": dataset_name}, {"_id": 0, "data": 0}))
    return jsonify(image_info)


# @app.route("/api/get_image_titles", methods=["GET"])
# def get_image_titles():
#     dataset_name = request.args.get("dataset_name")
#     image_titles = [image["name"] for image in collection.find({"dataset": dataset_name})]
#     return jsonify(image_titles)

@app.route("/api/update_selected_image_info", methods=["POST"])
def update_selected_image_info():
    data = request.json
    image_name = data["name"]
    dataset_name = data["dataset_name"]
    new_dataset_name = data["new_dataset_name"]
    new_class_name = data["new_class_name"]
    confidence_score = data["confidence_score"]

    # Find the image in the database and update its dataset and class
    query = {"name": image_name, "dataset": dataset_name}
    update_values = {
        "$set": {
            "dataset": new_dataset_name,
            "class": new_class_name,
            "score": confidence_score
        }
    }
    collection.update_one(query, update_values)

    return jsonify({"message": "Selected image info updated in MongoDB!"})


@app.route("/api/upload_large_npz", methods=["POST"])
def upload_large_npz():
    file = request.files["file"]

    # Check if the file is an .npz
    if not file.filename.endswith(".npz"):
        return jsonify({"message": "File format not supported. Only .npz files are allowed!"}), 400

    # Load the data from the .npz file
    data = np.load(file, allow_pickle=True)

    # Convert the numpy data to a serialized binary format using pickle
    serialized_data = pickle.dumps(dict(data))

    # Split data into chunks and store each chunk in a separate document
    CHUNK_SIZE = 16 * 1024 * 1024 - 1024  # 16MB minus 1KB
    total_chunks = -(-len(serialized_data) // CHUNK_SIZE)  # Ceiling division

    for chunk_number in range(total_chunks):
        start = chunk_number * CHUNK_SIZE
        end = (chunk_number + 1) * CHUNK_SIZE
        chunk_data = serialized_data[start:end]

        doc = {
            "name": file.filename,
            "chunk_number": chunk_number,
            "total_chunks": total_chunks,
            "data": Binary(chunk_data)
        }

        collection2.insert_one(doc)

    return jsonify({"message": "Data from large .npz file uploaded in chunks to MongoDB!"})


@app.route("/api/list_npz_files", methods=["GET"])
def list_npz_files():
    files = collection2.distinct("name")
    return jsonify({"npz_files": files})


@app.route("/api/retrieve_data/<file_name>", methods=["GET"])
def retrieve_data_api(file_name):
    chunks = list(collection2.find({"name": file_name}).sort("chunk_number"))
    if not chunks:
        return "No data found for the given file name", 404
    total_chunks = chunks[0]["total_chunks"]
    if len(chunks) != total_chunks:
        raise ValueError("Some chunks are missing")
    serialized_data = b"".join(chunk["data"] for chunk in chunks)
    return Response(serialized_data, content_type="application/octet-stream")



# @app.route('/api/get_npz_names', methods=['GET'])
# def get_npz_names():
#     file_names = collection2    .distinct("name")
#     return jsonify(file_names)
#
#
# @app.route('/api/retrieve_data/<file_name>', methods=['GET'])
# def retrieve_data_route(file_name):
#
#         # Query all the chunks for the file
#         chunks = list(collection.find({"name": file_name}).sort("chunk_number"))
#
#         # Verify all chunks are present
#         total_chunks = chunks[0]["total_chunks"]
#         if len(chunks) != total_chunks:
#             return "Some chunks are missing", 400
#
#         # Combine chunks to get the serialized data
#         serialized_data = b"".join(chunk["data"] for chunk in chunks)
#         data = pickle.loads(serialized_data)
#         return data

@app.route("/api/update_image_embeddings", methods=["POST"])
def update_image_embeddings():
    data = request.json
    image_name = data["name"]
    dataset_name = data["dataset_name"]
    embeddings = data["embeddings"]

    # Find the image in the database and update its embeddings
    query = {"name": image_name, "dataset": dataset_name}
    update_values = {
        "$set": {
            "embeddings": embeddings
        }
    }
    collection.update_one(query, update_values)

    return jsonify({"message": "Image embeddings updated in MongoDB!"})

# @app.route("/api/get_npz_files", methods=["GET"])
# def get_npz_files():
#     # Retrieve all distinct .npz filenames from the database
#     file_names = collection2.distinct("name")
#     return jsonify(file_names)
#
#
# def get_npz_data():
#     file_name = request.args.get('file_name')
#     chunks = list(collection2.find({"name": file_name}).sort("chunk_number"))
#
#     if not chunks:
#         return jsonify({"error": f"No data found for file {file_name}"}), 404
#
#     total_chunks = chunks[0]["total_chunks"]
#     if len(chunks) != total_chunks:
#         return jsonify({"error": "Some chunks are missing"}), 400
#
#     serialized_data = b"".join(chunk["data"] for chunk in chunks)
#     data = pickle.loads(serialized_data)
#
#     # It's better to convert the data to JSON if possible or specify the content type.
#     return Response(pickle.dumps(data), content_type="application/octet-stream"), 200
#
# @app.route("/api/download_large_npz", methods=["GET"])
# def download_large_npz():
#     file_name = request.args.get('file_name')
#
#     # Fetch the chunks for the given file name from MongoDB
#     chunks = list(collection2.find({"name": file_name}).sort("chunk_number"))
#     total_chunks = chunks[0]["total_chunks"]
#
#     # Check if all chunks are present
#     if len(chunks) != total_chunks:
#         return jsonify({"error": "Some chunks are missing"}), 400
#
#     # Assemble the serialized data from the chunks
#     serialized_data = b"".join(chunk["data"] for chunk in chunks)
#
#     # Deserialize the data to get the numpy data
#     np_data = pickle.loads(serialized_data)
#
#     # Convert the numpy data back to an .npz file format in memory
#     buffer = io.BytesIO()
#     np.savez(buffer, **np_data)
#     buffer.seek(0)
#
#     # Instead of sending the file, return the serialized data
#     return jsonify({"serialized_data": serialized_data.decode("latin1")})
#
#
# @app.route("/api/list_npz_files", methods=["GET"])
# def list_npz_files():
#     # Query distinct file names in the collection
#     file_names = collection2.distinct("name")
#     return jsonify(file_names)
#
# @app.route("/api/retrieve_npz_file/<file_name>", methods=["GET"])
# def retrieve_npz_file(file_name):
#     chunks = list(collection2.find({"name": file_name}).sort("chunk_number"))
#     total_chunks = chunks[0]["total_chunks"]
#     if len(chunks) != total_chunks:
#         return jsonify({"error": "Some chunks are missing"}), 400
#     serialized_data = b"".join(chunk["data"] for chunk in chunks)
#     data = pickle.loads(serialized_data)
#     return jsonify({"data": data.tolist()})


if __name__ == "__main__":
    app.run(host="0.0.0.0",debug=True, port=5003)

