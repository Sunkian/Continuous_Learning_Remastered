import numpy as np
import os
import json
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
# from sklearn.cluster import KMeans
from collections import Counter
import umap

from scipy.stats import mode
from sklearn.metrics import pairwise_distances_argmin_min

# Load the MobileNetV2 model pre-trained on ImageNet data
base_model = MobileNetV2(weights='imagenet', include_top=True)
model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)

def get_embedding_tf(model, image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return model.predict(x)

def get_2d_embeddings(embeddings):
    reducer = umap.UMAP()  # Create a UMAP instance
    embeddings_2d = reducer.fit_transform(embeddings)
    return embeddings_2d

def visualize_embeddings(embeddings_2d):
    plt.figure(figsize=(10, 10))
    sns.scatterplot(x=embeddings_2d[:, 0], y=embeddings_2d[:, 1], alpha=0.7)
    plt.show()


def visualize_embeddings_with_circles(embeddings_2d, labels, n_clusters=5):
    # Load ImageNet labels
    with open("/Users/apagnoux/PycharmProjects/webapp-streamlit-upgrade/imagenet-simple-labels.json", 'r') as f:
        imagenet_labels = json.load(f)

    # Convert number labels to their corresponding ImageNet labels
    label_names = [imagenet_labels[int(label)] for label in labels]


    # 1. Use KMeans to cluster the embeddings
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(embeddings_2d)
    cluster_centers = kmeans.cluster_centers_
    cluster_labels = kmeans.labels_

    plt.figure(figsize=(10, 10))

    # 2. For each cluster, find the farthest point to calculate the radius
    for cluster_center, cluster_label in zip(cluster_centers, range(n_clusters)):
        distances = np.sqrt(((embeddings_2d[cluster_labels == cluster_label] - cluster_center) ** 2).sum(axis=1))
        radius = np.max(distances)
        circle = plt.Circle(cluster_center, radius, fill=False, edgecolor='gray', linestyle='--', linewidth=1)
        plt.gca().add_patch(circle)

        # Determine the majority label for the cluster
        majority_label_count = Counter(np.array(label_names)[cluster_labels == cluster_label]).most_common(1)
        if majority_label_count:
            majority_label = majority_label_count[0][0]
            plt.text(cluster_center[0], cluster_center[1], majority_label, ha='center', va='center', fontsize=9,
                     fontweight='bold')

    sns.scatterplot(x=embeddings_2d[:, 0], y=embeddings_2d[:, 1], alpha=0.7)

    plt.show()


dataset_folder = '/Users/apagnoux/Downloads/archive/raw-img/farfalla/'
image_paths = [os.path.join(dataset_folder, fname) for fname in os.listdir(dataset_folder) if fname.endswith('.jpg')]  # assuming jpg images

all_embeddings = []
image_labels = []  # list to store predicted labels
for image_path in image_paths:
    embedding = get_embedding_tf(model, image_path)
    all_embeddings.append(embedding)

    # Preprocess the image for prediction using base_model
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # Get predicted class label for the image
    predictions = base_model.predict(x)
    predicted_label = np.argmax(predictions, axis=-1)[0]
    image_labels.append(predicted_label)

all_embeddings = np.squeeze(np.array(all_embeddings))  # C
# Reduce dimension
embeddings_2d = get_2d_embeddings(all_embeddings)

# For visualization, we can optionally use image file names as labels.
labels = [os.path.basename(p) for p in image_paths]

# visualize_embeddings(embeddings_2d)
#visualize_embeddings_with_circles(embeddings_2d, n_clusters=5)
visualize_embeddings_with_circles(embeddings_2d, image_labels, n_clusters=5)

