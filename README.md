# Paper Title

## Overview
This repository contains the code and data related to the paper [3D CT Slice Image-Based Algorithm for Non-Wet Defect Inspection in Solder Joints](https://example.com). The paper explores a new method for inspecting non-wet defects in semiconductor solder joints.

## Paper Link
[Link to the paper](https://example.com)

## Datasets
This repository includes the following datasets:
- `data/1ch_dataset`: Train 2D Slice Data for Supervised Classifier
- `data/3d_test_data`: Target 3D Volume Data of Solder Joints

## Installation
To run this project locally, follow these steps:

1. Clone this repository:
    ```sh
    git clone https://github.com/thomas11809/Non-Wet-Defect-Inspection.git
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage
To run the code:

1. Prepare the data. Place the data files in the `data/` directory.
2. Run the main script:
    ```sh
    python main.py
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
