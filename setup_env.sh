#!/bin/bash

# PyTorch (CUDA 11.3)
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

# PyTorch Lightning
pip install pytorch-lightning

# OpenCV (conda-forge channel)
pip install opencv-python

# scikit-learn (anaconda channel)
pip install scikit-learn

# others
pip install matplotlib
pip install scikit-image
pip install pandas
pip install openpyxl