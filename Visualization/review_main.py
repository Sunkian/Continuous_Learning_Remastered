import streamlit as st
import numpy as np
import pandas as pd
# import umap
import umap.umap_ as umap
import plotly.express as px
from API.api_helper import fetch_datasets, fetch_image_info
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances_argmin_min
# from api_helper import fetch_datasets, fetch_image_info
from collections import Counter
from scipy.spatial import ConvexHull


# Visualize clusters from embeddings
# Streamlit app
def main():
    st.title("Image Embedding Visualization")

    # Fetch and display available datasets
    datasets = fetch_datasets()
    selected_dataset = st.selectbox('Select a dataset:', datasets)

    # Fetch embeddings and labels
    data = fetch_image_info(selected_dataset)
    embeddings = np.array([item['embeddings'] for item in data])
    labels = [item['class'] for item in data]

    # Reduce dimensionality with UMAP
    reducer = umap.UMAP(n_neighbors=15, n_components=2, metric='cosine')
    embeddings_2d = reducer.fit_transform(embeddings)

    # Cluster using KMeans
    n_clusters = len(set(labels))  # Assuming number of unique labels corresponds to number of clusters
    kmeans = KMeans(n_clusters=n_clusters)
    cluster_assignments = kmeans.fit_predict(embeddings_2d)

    # Prepare data for Plotly
    df = pd.DataFrame(embeddings_2d, columns=['x', 'y'])
    df['cluster'] = cluster_assignments
    df['label'] = labels

    fig = px.scatter(df, x='x', y='y', color='cluster', hover_data=['label'])

    # Add cluster circles and annotations
    for cluster_num in range(n_clusters):
        cluster_points = embeddings_2d[cluster_assignments == cluster_num]
        cluster_labels = np.array(labels)[cluster_assignments == cluster_num]
        center = kmeans.cluster_centers_[cluster_num]

        # Determine most frequent label in the cluster
        most_common_label = Counter(cluster_labels).most_common(1)[0][0]

        # Add cluster circle
        radius = np.max(np.linalg.norm(cluster_points - center, axis=1))
        fig.add_shape(
            type="circle",
            x0=center[0] - radius,
            y0=center[1] - radius,
            x1=center[0] + radius,
            y1=center[1] + radius,
            line=dict(color="Gray", width=2, dash="dash"),
        )

        # Add annotation for most frequent label
        fig.add_annotation(
            x=center[0],
            y=center[1],
            text=most_common_label,
            showarrow=False,
            font=dict(color="Black", size=16)
        )

    # Display plotly figure in Streamlit
    st.plotly_chart(fig)


if __name__ == "__main__":
    main()