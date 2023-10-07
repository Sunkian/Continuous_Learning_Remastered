import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from collections import Counter
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.applications.mobilenet_v2 import decode_predictions
import os

# Load the MobileNetV2 model pre-trained on ImageNet data
base_model = MobileNetV2(weights='imagenet', include_top=True)
model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)

def get_image_files_from_directory(directory, extensions=['.jpg', '.jpeg', '.png']):
    all_files = os.listdir(directory)
    image_files = [f for f in all_files if any(f.endswith(ext) for ext in extensions)]
    return [os.path.join(directory, f) for f in image_files]

def get_image_embedding(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_batch)
    embedding = model.predict(img_preprocessed)
    return embedding.squeeze()


def get_image_prediction(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_batch)
    prediction = base_model.predict(img_preprocessed)

    # Decode the prediction to get human-readable labels
    decoded = decode_predictions(prediction, top=1)  # Get top prediction
    return decoded[0][0][1]  # Return the name of the top prediction


# Example usage:
directory_path = '/Users/apagnoux/Downloads/archive/raw-img/elefante'  # Path to your image folder
image_paths = get_image_files_from_directory(directory_path)

embeddings = [get_image_embedding(img_path) for img_path in image_paths]
predicted_labels = [get_image_prediction(img_path) for img_path in image_paths]

embeddings = np.array(embeddings)

# Use t-SNE to reduce dimensionality
tsne = TSNE(n_components=2, random_state=42, perplexity=2)
embeddings_2d = tsne.fit_transform(embeddings)

# Cluster the embeddings
def get_image_label(img_path):
    """Predicts the class label for the given image."""
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_batch)
    predictions = base_model.predict(img_preprocessed)
    return np.argmax(predictions)  # Return the index of the class with highest probability

def get_image_label_and_confidence(img_path):
    """Predicts the class label and its confidence for the given image."""
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_batch)
    predictions = base_model.predict(img_preprocessed)
    decoded_predictions = decode_predictions(predictions, top=1)  # Decodes the top-1 predicted class
    label_name = decoded_predictions[0][0][1]
    label_confidence = decoded_predictions[0][0][2]
    return label_name, label_confidence

# Predict the label for each image
image_labels, image_confidences = zip(*[get_image_label_and_confidence(img_path) for img_path in image_paths])

# Cluster the 2D embeddings using KMeans
n_clusters = 10  # Change this based on how many clusters you anticipate or other criteria
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_assignments = kmeans.fit_predict(embeddings_2d)

# Plot t-SNE results with clusters circled
plt.figure(figsize=(10, 10))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=cluster_assignments, cmap='jet')  # Color based on cluster

# Add cluster circles and annotate with majority label
for cluster_id in range(n_clusters):
    cluster_points = embeddings_2d[cluster_assignments == cluster_id]
    cluster_labels = [image_labels[i] for i, assignment in enumerate(cluster_assignments) if assignment == cluster_id]

    # Majority Voting for the cluster's label
    majority_label = Counter(cluster_labels).most_common(1)[0][0]  # This will give the label with maximum occurrence

    centroid = cluster_points.mean(axis=0)
    plt.annotate(str(majority_label), (centroid[0], centroid[1]), textcoords="offset points", xytext=(0, 5),
                 ha='center', fontsize=10, color='black', weight='bold')

    radius = np.max(np.linalg.norm(cluster_points - centroid, axis=1))
    circle = plt.Circle(centroid, radius, color='gray', fill=False, linestyle='--', linewidth=1)
    plt.gca().add_patch(circle)
    majority_label_name = Counter(cluster_labels).most_common(1)[0][0]
    plt.annotate(majority_label_name, (centroid[0], centroid[1]), textcoords="offset points", xytext=(0, 5),
                 ha='center', fontsize=10, color='black', weight='bold')

plt.xlabel('t-SNE component 1')
plt.ylabel('t-SNE component 2')
plt.title('t-SNE visualization of image embeddings with clusters circled')
plt.show()
