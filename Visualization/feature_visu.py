import tempfile
from io import BytesIO

import requests
import streamlit as st
import umap
import matplotlib.pyplot as plt
import numpy as np
# from API.api_helper import fetch_npz_files, fetch_npz_content
# from API.api_helper import fetch_npz_names, fetch_npz_data
from API.api_helper import retrieve_data, list_npz_files
from sklearn.manifold import TSNE

CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]

# def retrieve_data(file_name):
#     response = requests.post("http://flask_api:5003/api/get_npz_data", json={"name": file_name})
#     data = response.json()
#     return data
# def load_id_data(id_data):
#     id_feat = id_data['feat_log']
#     id_label = id_data['label_log']
#     return id_feat, id_label
#
# # def load_ood_data(ood_data):
# #     ood_feat = ood_data['ood_feat_log']
# #     ood_score = ood_data['ood_score_log'][:, 0]
# #     return ood_feat, ood_score
#
# # def load_ood_data(selected_npz):
# #     # Load OOD data
# #     data = retrieve_data(selected_npz)
# #     ood_feat = data['ood_feat_log']
# #     ood_score = data['ood_score_log'][:, 0]
# #     return ood_feat, ood_score
#
# def load_ood_data(file_name_to_retrieve):
#     # Load OOD data
#     data = retrieve_data(file_name_to_retrieve)
#     ood_feat = data['ood_feat_log']
#     ood_score = data['ood_score_log'][:, 0]  # Adjusting based on previous example
#     return ood_feat, ood_score

def load_id_data(file_name_to_retrieve):
    # Load ID data
    # to_retrieve = 'CIFAR-10_val_resnet18-supcon_in_alllayers.npz'
    data = retrieve_data(file_name_to_retrieve)

    id_feat = data['feat_log']
    id_label = data['label_log']

    return id_feat, id_label


def load_ood_data(file_name_to_retrieve):
    # Load OOD data
    # to_retrieve = "CATvsCIFAR-10_resnet18-supcon_out_alllayers.npz"
    data = retrieve_data(file_name_to_retrieve)

    ood_feat = data['ood_feat_log']
    ood_score = data['ood_score_log'][:, 0]  # Adjusting based on previous example

    return ood_feat, ood_score


def plot_umap(id_feat, id_label, ood_feat, ood_score):
    # Combine both ID and OOD features for UMAP fitting
    combined_features = np.vstack((id_feat, ood_feat))

    # Use UMAP for dimensionality reduction
    embedding = TSNE(n_components=2).fit_transform(combined_features)

    # Separate ID and OOD embeddings
    embedding_id = embedding[:len(id_feat)]
    embedding_ood = embedding[len(id_feat):]

    fig, ax = plt.subplots(figsize=(10, 8))



    # Plot ID
    scatter_id = ax.scatter(embedding_id[:, 0], embedding_id[:, 1], c=id_label, cmap='tab10', s=5, label='ID')
    # Plot OOD
    scatter_ood = ax.scatter(embedding_ood[:, 0], embedding_ood[:, 1], c=ood_score, cmap='viridis', s=5, label='OOD',
                             alpha=0.6)

    # Create a legend
    ax.legend(handles=[scatter_id, scatter_ood], loc='upper right')
    plt.colorbar(scatter_ood, ax=ax, label='OOD Score')
    ax.set_title("UMAP Visualization of ID vs OOD Features")

    for idx, class_name in enumerate(CLASS_NAMES):
        mean_x = np.mean(embedding_id[id_label == idx, 0])
        mean_y = np.mean(embedding_id[id_label == idx, 1])
        ax.text(mean_x, mean_y, class_name, fontsize=9, ha='center', va='center', backgroundcolor='white')

    return fig


# def retrieve_data(file_name):
#     return retrieve_npz_data(file_name)


def visuuu():
    st.title(f"T-SNE Visualization of ID vs TEST Embeddings on the same graph")

    available_files = list_npz_files()

    # Allow the user to select a file
    selected_id_file = st.selectbox('Select the ID data file:', available_files, index=0)
    selected_ood_file = st.selectbox('Select the OOD data file:', available_files, index=1)

    if st.button("Show Graph"):
        # Load and process the data
        id_feat, id_label = load_id_data(selected_id_file)
        ood_feat, ood_score = load_ood_data(selected_ood_file)

        # Display UMAP visualization
        st.pyplot(plot_umap(id_feat, id_label, ood_feat, ood_score))