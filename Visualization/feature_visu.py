import streamlit as st
import umap
import matplotlib.pyplot as plt
import numpy as np
# from API.api_helper import fetch_npz_files, fetch_npz_content
# from API.api_helper import fetch_npz_names, fetch_npz_data

CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]

def load_id_data(id_data):
    id_feat = id_data['feat_log']
    id_label = id_data['label_log']
    return id_feat, id_label

def load_ood_data(ood_data):
    ood_feat = ood_data['ood_feat_log']
    ood_score = ood_data['ood_score_log'][:, 0]
    return ood_feat, ood_score



def plot_umap(id_feat, id_label, ood_feat, ood_score):
    """
    Plot UMAP visualization of ID vs OOD features on the same graph.

    Parameters:
    - id_feat (numpy.array): Features of in-distribution (ID) data.
    - id_label (numpy.array): Labels of ID data.
    - ood_feat (numpy.array): Features of out-of-distribution (OOD) data.
    - ood_score (numpy.array): Scores of OOD data.

    Returns:
    - matplotlib.figure.Figure: UMAP visualization.
    """

    # Combine both ID and OOD features for UMAP fitting
    combined_features = np.vstack((id_feat, ood_feat))

    # Use UMAP for dimensionality reduction
    embedding = umap.UMAP().fit_transform(combined_features)

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


def visuuu():
    st.title("UMAP Visualization of ID vs OOD Embeddings on the same graph")

    # npz_files = fetch_npz_names()
    #
    # id_file = st.selectbox("Select ID .npz file", npz_files)
    # ood_file = st.selectbox("Select OOD .npz file", npz_files)
    #
    # if st.button("Visualize"):
    #     id_data = fetch_npz_data(id_file)
    #     ood_data = fetch_npz_data(ood_file)
    #
    #     id_feat, id_label = load_id_data(id_data)
    #     ood_feat, ood_score = load_ood_data(ood_data)
    #
    #     st.pyplot(plot_umap(id_feat, id_label, ood_feat, ood_score))