#!/bin/bash

# Run the Streamlit frontend
streamlit run main.py &

# Run the Flask API
python3 api.py &
python3 /API/api_calls.py &


