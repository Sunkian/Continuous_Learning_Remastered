import streamlit as st
import pandas as pd
import plotly.express as px
from API.api_helper import fetch_image_info, fetch_datasets


class ImageDataApp:

    def __init__(self):
        self.selected_dataset = None
        self.datasets = fetch_datasets()

    def select_dataset(self):
        self.selected_dataset = st.selectbox("Select a dataset:", self.datasets)
        if self.selected_dataset:
            st.markdown(f"**Selected dataset:** {self.selected_dataset}")

    def display_image_info(self):
        with st.expander("Image Information for Selected Dataset"):
            image_info = fetch_image_info(self.selected_dataset)
            if not image_info:
                st.warning("No image information found for the selected dataset.")
            else:
                st.table(image_info)

    def display_class_clusters(self):
        image_info = fetch_image_info(self.selected_dataset)
        if image_info:
            class_data = [entry["class"] for entry in image_info]
            confidence_data = [entry["score"] for entry in image_info]

            df = pd.DataFrame({"Class": class_data, "Confidence": confidence_data})

            st.header("Cluster Scatter Plot of Classes")
            fig = px.scatter(df, x="Class", y="Confidence", color="Class",
                             title="Scatter Plot of Image Classes based on Confidence Scores")

            # Rotate x-axis labels for better visibility
            fig.update_layout(xaxis_tickangle=-45)

            st.plotly_chart(fig)
    def display_confidence_plot(self):
        image_info = fetch_image_info(self.selected_dataset)
        if image_info:
            class_data = [entry["class"] for entry in image_info]
            confidence_data = [entry["score"] for entry in image_info]

            df = pd.DataFrame({"Class": class_data, "Confidence": confidence_data})

            st.header("Histogram of Confidence Scores")
            fig = px.histogram(df, x="Confidence", nbins=20, marginal="box")
            st.plotly_chart(fig)

            st.header("Pie Chart for Image Classes")
            df_counts = df["Class"].value_counts().reset_index()
            df_counts.columns = ["Class", "Count"]
            fig = px.pie(df_counts, names='Class', values='Count', hole=0.3)
            st.plotly_chart(fig)

            st.header("Boxplot for Confidence Scores")
            fig = px.box(df, x="Class", y="Confidence")
            fig.update_layout(xaxis_tickangle=-45)  # for rotating x-axis labels
            st.plotly_chart(fig)

def results():
    st.title("Image Information Table and Confidence Score Per Class")
    app = ImageDataApp()
    app.select_dataset()
    app.display_image_info()
    app.display_confidence_plot()
    app.display_class_clusters()
#
# if __name__ == "__main__":
#     main()
