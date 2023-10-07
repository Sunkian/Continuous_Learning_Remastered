import streamlit as st
import extra_streamlit_components as stx
from intro import intro
from Upload.upload import ImageUploader
from Run.run import RunInference
# from Review.results import results
from Review.results import Result
# from Review.results import results
from streamlit_elements import elements, mui, html
# from Visualization.review import review
# from Visualization.review_main import main
from Visualization.visualization import visualization
from Review.results import Result
from Run.feature_extract_ns import test

from Run.feature_extract_ns import test
from Visualization.feature_visu import visuuu

st.set_page_config(page_title="Continuous Learning App")

val = stx.stepper_bar(steps=["Intro", "Upload", "Run/Train", "Visualization", "Review"])

if val == 0:
    intro()
elif val == 1:
    app = ImageUploader()
    app.run()
elif val == 2:
    # app = RunInference()
    # app.main()
    # main()
    test()
elif val == 3:
    visualization()
    # visuuu()
else:
    # review()
    # results()
    test = Result()
    test.results_main()


### Works
# import streamlit as st
# import extra_streamlit_components as stx
# from intro import intro
# from Upload.upload import ImageUploader
# from Run.run import RunInference
# # from Review.results import results
# from Review.results import Result
# # from Review.results import results
# from streamlit_elements import elements, mui, html
# # from Visualization.review import review
# # from Visualization.review_main import main
# from Visualization.visualization import visualization
# from Review.results import Result
#
# from Run.feature_extract_ns import test
#
# st.set_page_config(page_title="Continuous Learning App")
#
# val = stx.stepper_bar(steps=["Intro", "Upload", "Run/Train", "Visualization", "Review"])
#
# if val == 0:
#     intro()
# elif val == 1:
#     app = ImageUploader()
#     app.run()
# elif val == 2:
#     # app = RunInference()
#     # app.main()
#     # main()
# elif val == 3:
#     visualization()
# else:
#     # review()
#     # results()
#     test = Result()
#     test.results_main()
