import argparse
from model_jingwei.utils.args_loader import get_args
import torch
from model_jingwei.exp.exp_OWL import Exp_OWL
import streamlit as st
from API.api_helper import upload_large_npz_to_backend, fetch_datasets
import os

# st.title("Sample Feature Extraction")


def test():
    datasets = fetch_datasets()

    # Let user select a dataset using Streamlit
    selected_dataset = st.selectbox("Select a dataset:", datasets)


    if st.button("Start new sample feature extraction"):
        parser = argparse.ArgumentParser()
        args = get_args()

        args.out_datasets = [selected_dataset, ]

        exp = Exp_OWL(args)



        print('>>>>>>>start feature extraction on new-coming data : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(args.out_datasets))
        # exp.ns_feature_extract('SVHN')
        # st.write("Feature extraction completed!")

        exp.ns_feature_extract(selected_dataset)
        st.write("Feature extraction completed")
        print("Feature extraction completed")
        base_path = "/app/cache/"
        # file_name = "/app/cache/SVHNvsCIFAR-10_resnet18-supcon_out_alllayers.npz"
        # original_file_name = "SVHNvsCIFAR-10_resnet18-supcon_out_alllayers.npz"
        new_file_name = f"{selected_dataset}vsCIFAR-10_resnet18-supcon_out_alllayers.npz"

        # Rename the file
        # os.rename(os.path.join(base_path, original_file_name), os.path.join(base_path, new_file_name))

        # Set the new file path
        file_name = os.path.join(base_path, new_file_name)

        response = upload_large_npz_to_backend(file_name)
        print(response.json())
        st.write("DONE")
        print("PUSHED TO DB !!!")

    if st.button('TEST'):
            args = get_args()

            args.out_datasets = [selected_dataset, ]

            exp = Exp_OWL(args)
            print('Start extracting CIFAR and put it in cache')
            exp.id_feature_extract()
            print('>>>>>>>start ood detection on new-coming data : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(
                args.out_datasets))
            unknown_idx, bool_ood, scores_conf, pred_scores, pred_labels = exp.ood_detection(selected_dataset, K=50)
            st.write(unknown_idx)
            print('UNKNOWN', unknown_idx)









