import plotly.express as px
import streamlit as st
from API.api_helper import fetch_datasets, fetch_image_info
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances_argmin_min
from API.api_helper import fetch_datasets, fetch_image_info, fetch_image_data_and_embeddings
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
import umap


def tsne_visu(embeddings):
    embeddings_array = np.array(embeddings)
    n_samples = embeddings_array.shape[0]
    perplexity_value = min(30, n_samples - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity_value, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings_array)
    return embeddings_2d

def tsne_visu_3d(embeddings):
    embeddings_array = np.array(embeddings)
    n_samples = embeddings_array.shape[0]
    perplexity_value = min(30, n_samples - 1)
    tsne = TSNE(n_components=3, perplexity=perplexity_value, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings_array)
    return embeddings_2d

def pca_visu(embeddings):
    embeddings_array = np.array(embeddings)
    pca = PCA(n_components=2, random_state=42)
    embeddings_2d = pca.fit_transform(embeddings_array)
    return embeddings_2d


def umap_visu(embeddings, n_neighbors=15, min_dist=0.1, n_components=2):
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components)
    embeddings_2d = reducer.fit_transform(embeddings)
    return embeddings_2d


def plot_explained_variance(explained_var):
    """
    Plot the explained variance ratio from PCA and display it using Streamlit.
    """
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(explained_var)), explained_var, alpha=0.5, align='center', label='individual explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.title('Explained variance ratio for each principal component')
    plt.legend(loc='best')
    plt.tight_layout()

    # Streamlit display
    st.pyplot(plt)

def visualization():
    st.title("Image and Embeddings Viewer")
    embeddings = []
    classes = []  # List to store the corresponding classes of the embeddings

    # Fetch and show datasets in a dropdown
    datasets = fetch_datasets()
    selected_dataset = st.selectbox("Choose a dataset:", datasets)

    if selected_dataset:
        image_infos = fetch_image_info(selected_dataset)

        if image_infos is not None:
            for infos in image_infos:
                image_class = infos['class']
                image_embeddings = infos['embeddings']
                if len(image_embeddings) > 0:
                    embeddings.append(image_embeddings)
                    classes.append(image_class)
                # embeddings.append(image_embeddings)
                # classes.append(image_class)  # Append the class to the classes list
            #
            # for i, emb in enumerate(embeddings):
            #     print(f"Embedding {i} shape: {np.array(emb).shape}")
            ## 3 D
            # # t-SNE Visualization
            # tsne_coords = tsne_visu(embeddings)
            # df_tsne = pd.DataFrame(tsne_coords, columns=["x", "y", "z"])
            # df_tsne["Classes"] = classes
            # fig_tsne = px.scatter_3d(df_tsne, x="x", y="y", z="z", color="Classes",
            #                          title="t-SNE Visualization per Class")
            # st.plotly_chart(fig_tsne)
            #
            # # PCA Visualization
            # pca_coords = pca_visu(embeddings)
            # df_pca = pd.DataFrame(pca_coords, columns=["x", "y", "z"])
            # df_pca["Classes"] = classes
            # fig_pca = px.scatter_3d(df_pca, x="x", y="y", z="z", color="Classes", title="PCA Visualization per Class")
            # st.plotly_chart(fig_pca)
            #
            # # UMAP Visualization
            # umap_coords = umap_visu(embeddings)
            # df_umap = pd.DataFrame(umap_coords, columns=["x", "y", "z"])
            # df_umap["Classes"] = classes
            # fig_umap = px.scatter_3d(df_umap, x="x", y="y", z="z", color="Classes",
            #                          title="UMAP Visualization per Class")
            # st.plotly_chart(fig_umap)

            # col1, col2 = st.columns(2)

            ##### 2D AND 3D Visualization
            # col1, col2 = st.columns([0.6, 0.4])
            #
            # col1.markdown('<style>margin-right: 50px;</style>', unsafe_allow_html=True)
            #
            # # For 3D plots in the first column:
            # with col1:
            #     st.header("3D Plots")
            #
            #     # t-SNE Visualization
            #     tsne_coords_3d = tsne_visu_3d(embeddings)  # Assuming tsne_visu can handle both 2D and 3D
            #     df_tsne_3d = pd.DataFrame(tsne_coords_3d, columns=["x", "y", "z"])
            #     df_tsne_3d["Classes"] = classes
            #     fig_tsne_3d = px.scatter_3d(df_tsne_3d, x="x", y="y", z="z", color="Classes",
            #                                 title="t-SNE Visualization per Class")
            #     st.plotly_chart(fig_tsne_3d)
            #
            #     # Continue with your other 3D plots...
            #     fig_tsne_3d.update_layout(margin=dict(l=300))
            # # For 2D plots in the second column:
            # with col2:
            #     st.header("2D Plots")
            #
            #     # t-SNE Visualization
            #     tsne_coords = tsne_visu(embeddings)
            #     df_tsne = pd.DataFrame(tsne_coords, columns=["comp-1", "comp-2"])
            #     df_tsne["Classes"] = classes
            #     fig_tsne = px.scatter(df_tsne, x="comp-1", y="comp-2", color="Classes",
            #                           title="t-SNE Visualization per Class")
            #     st.plotly_chart(fig_tsne)

            ## 2-D
                # t-SNE Visualization
            tsne_coords = tsne_visu(embeddings)
            df_tsne = pd.DataFrame(tsne_coords, columns=["comp-1", "comp-2"])
            df_tsne["Classes"] = classes
            fig_tsne = px.scatter(df_tsne, x="comp-1", y="comp-2", color="Classes",
                                      title="t-SNE Visualization per Class")
            st.plotly_chart(fig_tsne)

                # PCA Visualization
            pca_coords = pca_visu(embeddings)
            df_pca = pd.DataFrame(pca_coords, columns=["comp-1", "comp-2"])
            df_pca["Classes"] = classes
            fig_pca = px.scatter(df_pca, x="comp-1", y="comp-2", color="Classes",
                                     title="PCA Visualization per Class")
            st.plotly_chart(fig_pca)

                # UMAP Visualization
            umap_coords = umap_visu(embeddings)
            df_umap = pd.DataFrame(umap_coords, columns=["comp-1", "comp-2"])
            df_umap["Classes"] = classes
            fig_umap = px.scatter(df_umap, x="comp-1", y="comp-2", color="Classes",
                                      title="UMAP Visualization per Class")
            st.plotly_chart(fig_umap)

            ## PCA VARIANCE RATIO
            pca_full = PCA().fit(embeddings)
            explained_var = pca_full.explained_variance_ratio_
            plot_explained_variance(explained_var)




        else:
            st.write(f"No images found for the dataset: {selected_dataset}")

    # Alice
    # image_infos = fetch_image_info(selected_dataset)
    # embeddings = []
    # for infos in image_infos:
    #     image_name = infos['name']
    #     image_embeddings = infos['embeddings']
    #     embeddings.append(image_embeddings)
    #     # st.write(image_name,image_embeddings)
