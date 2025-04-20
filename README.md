# 3D CT Slice Image-Based Algorithm for Non-Wet Defect Inspection in Solder Joints

## Overview
This repository contains the code and data related to the paper [3D CT Slice Image-Based Algorithm for Non-Wet Defect Inspection in Solder Joints]. The paper explores a new method for inspecting non-wet defects in semiconductor solder joints.

## Datasets
[Download link](https://zenodo.org/records/15250542)

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
2. Run the main script:
   ```sh
    python resnet_counting.py
    bash 2d-ml.sh
    bash 3d-ml.sh
    bash ablation-resnet.sh
    ```

## Directory Structure
```
your-repository/
├── data/
│ ├── 1ch_dataset/
│ │ ├── 2d_train_data/
│ │ └── 2d_val_data/
│ └── 3d_test_data/
│   ├── normal/
│   └── nonwet/
├── weights/
│ └── resnet18_model.ckpt
├── resnet.py
├── resnet_val.py
├── resnet_counting.py
├── ml.py
├── util.py
├── requirements.txt
└── README.md
```

## Contact
For questions or inquiries, please contact [thomas11809@snu.ac.kr](mailto:thomas11809@snu.ac.kr).
