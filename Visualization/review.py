# import streamlit as st
# from streamlit_extras.colored_header import colored_header
# import streamlit as st
# import requests
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
# from sklearn.cluster import KMeans
# from collections import Counter
# import tensorflow as tf
# from tensorflow.keras.applications import MobileNetV2
# from tensorflow.keras.models import Model
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
# import io
# import streamlit as st
# import requests
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
# from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
# from collections import Counter
# import tensorflow as tf
# from tensorflow.keras.applications import MobileNetV2
# from tensorflow.keras.models import Model
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
#
# from API.api_helper import fetch_datasets, fetch_images, fetch_image_info
#
# # Cluster visualization
# def visualize_clusters(embeddings_2d, cluster_assignments, image_labels, n_clusters):
#     plt.figure(figsize=(10, 10))
#     plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=cluster_assignments, cmap='jet')
#
#     for cluster_id in range(n_clusters):
#         cluster_points = embeddings_2d[cluster_assignments == cluster_id]
#         cluster_labels = [image_labels[i] for i, assignment in enumerate(cluster_assignments) if assignment == cluster_id]
#
#         majority_label = Counter(cluster_labels).most_common(1)[0][0]
#         centroid = cluster_points.mean(axis=0)
#
#         radius = np.max(np.linalg.norm(cluster_points - centroid, axis=1))
#         circle = plt.Circle(centroid, radius, color='gray', fill=False, linestyle='--', linewidth=1)
#         plt.gca().add_patch(circle)
#         plt.annotate(majority_label, (centroid[0], centroid[1]), textcoords="offset points", xytext=(0, 5),
#                      ha='center', fontsize=10, color='black', weight='bold')
#
#     plt.xlabel('t-SNE component 1')
#     plt.ylabel('t-SNE component 2')
#     plt.title('t-SNE visualization of image embeddings with clusters circled')
#     st.pyplot(plt)
#
#
# # Streamlit UI
# st.title("Image Clustering App")
#
# # Get datasets
# datasets = fetch_datasets()
# selected_dataset = st.selectbox("Select a dataset", datasets)
#
# # Get images
# image_paths = fetch_images(selected_dataset)
#
# # Process and get embeddings
# embeddings = [get_image_embedding(img_path) for img_path in image_paths]
# predicted_labels = [get_image_prediction(img_path) for img_path in image_paths]
# embeddings = np.array(embeddings)
#
# # Dimensionality reduction
# tsne = TSNE(n_components=2, random_state=42, perplexity=2)
# embeddings_2d = tsne.fit_transform(embeddings)
#
# # Cluster algorithm selection
# cluster_algo = st.selectbox("Choose clustering algorithm", ["KMeans", "Agglomerative", "DBSCAN"])
# if cluster_algo == "KMeans":
#     n_clusters = st.slider("Select number of clusters", 2, 20, 10)
#     clusterer = KMeans(n_clusters=n_clusters, random_state=42)
# elif cluster_algo == "Agglomerative":
#     n_clusters = st.slider("Select number of clusters", 2, 20, 10)
#     clusterer = AgglomerativeClustering(n_clusters=n_clusters)
# else:
#     eps = st.slider("Select DBSCAN eps", 0.1, 2.0, 0.5)
#     min_samples = st.slider("Select DBSCAN min_samples", 1, 20, 5)
#     clusterer = DBSCAN(eps=eps, min_samples=min_samples)
#
# # Clustering
# cluster_assignments = clusterer.fit_predict(embeddings_2d)
#
# # Display cluster visualization
# visualize_clusters(embeddings_2d, cluster_assignments, predicted_labels, len(np.unique(cluster_assignments)))
import streamlit as st

def review():
    st.header('HELLO')

