import requests
import time
import api
from api import hello
import streamlit as st

time.sleep(5)  # wait for 5 seconds
response = hello()
st.write('HELLOOOOO')
st.write(response)

