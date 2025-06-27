#!/bin/bash

# Install numpy with specific build flags to avoid binary compatibility issues
pip uninstall -y numpy
pip install numpy==1.20.3 --no-binary numpy

# Install scipy with specific build flags
pip uninstall -y scipy
pip install scipy==1.7.3 --no-binary scipy

# Continue with normal pip install
pip install -r requirements-streamlit.txt
