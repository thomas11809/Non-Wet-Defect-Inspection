# 3D CT Slice Image-Based Algorithm for Non-Wet Defect Inspection in Solder Joints

[![Journal](https://img.shields.io/badge/Journal-IEEE%20Access-blue.svg)](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=6287639) [![DOI](https://img.shields.io/badge/DOI-10.1109/ACCESS.2025.3604431-brightgreen)](https://doi.org/10.1109/ACCESS.2025.3604431)


## Overview
This repository contains the code and data related to the paper **["3D CT Slice Image-Based Algorithm for Non-Wet Defect Inspection in Solder Joints"](https://doi.org/10.1109/ACCESS.2025.3604431)**. The paper explores a new method for inspecting non-wet defects in semiconductor solder joints, and is published in *IEEE Access*.

The dataset and pretrained weights can be downloaded from Zenodo: [Download link](https://zenodo.org/records/15250542)

## Datasets
This repository includes the following datasets:
- `data/3d_test_data`: Evaluation Target (Full 3D Solder Bump Volumes)
- `data/1ch_dataset`: Training Data (Discriminative 2D CT Slices)

## Installation
To run this project locally, follow these steps:

1. Clone this repository:
    ```sh
    git clone https://github.com/thomas11809/Non-Wet-Defect-Inspection.git
    ```

2. Install the required packages:
    ```sh
    conda create -n [env-name] python=3.7.11 -y
    conda activate [env-name]
    cd Non-Wet-Defect-Inspection/
    bash setup_env.sh
    ```

## Usage
To run the code:

1. Prepare the data. Place the data files in the `data/` directory.
2. Prepare the model weight. Place the ckpt file in the `weights/` directory.
3. Run the script below:
   ```sh
   # Main model
   python resnet_counting.py

   # Baseline Methods
   bash 2d-ml.sh
   bash 3d-ml.sh

   # Ablation results
   bash ablation-resnet.sh
   ```

## Directory Structure
```
your-repository/
├── data/                       # (Download required)
│   ├── 1ch_dataset/            # Training Data (2D training/validation data)
│   │   ├── 2d_train_data/
│   │   └── 2d_val_data/
│   └── 3d_test_data/           # Evaluation Target (3D test data: normal/nonwet)
│       ├── normal/
│       └── nonwet/
│
├── weights/                    # (Download required)
│   └── resnet18_model.ckpt     # Pretrained model checkpoint
│
├── resnet_counting.py          # Main model script
├── ml.py                       # Baseline methods (2d-ml & 3d-ml)
├── util.py                     # Helper functions
│
├── setup_env.sh                # Environment setup script
├── 2d-ml.sh                    # Script for running 2D ml methods
├── 3d-ml.sh                    # Script for running 3D ml methods
├── ablation-resnet.sh          # Ablation study script
└── README.md                   # Project description and usage
```

<!--  
## Citation
If you find this code useful for your research, please consider citing our paper:
```bibtex

```
-->

## Contact
For questions or inquiries, please contact [thomas11809@snu.ac.kr](mailto:thomas11809@snu.ac.kr).
