# Law Retrieval System 

This project implements a multimodal retrieval system for traffic sign regulations, combining text queries with image analysis to find relevant articles from traffic laws and regulations.

## Problem Description

### Input:
- Natural language questions about traffic signs
- Images of actual traffic signs on the street

### Output:
- Relevant articles from:
  - LAW ON ROAD TRAFFIC ORDER AND SAFETY (36/2024/QH15)
  - National Technical Regulation on Traffic Signs and Signals (QCVN 41:2024/BGTVT)

## Dataset Structure

```
data
├── VLSP2025/
│   ├── law_db/
│   │   ├── vlsp2025_law.json  # Law articles database
│   │   └── images.fld/            # Images 
│   │
│   └── train_db/
        ├── vlsp_2025_train.json  # Training data
        └── train_images/         # Training images
```

## Setup

```bash
git clone --recurse-submodules https://github.com/kuongan/VLSP2025.git
```
1. Create conda environment:
```bash
conda create -n vlsp python=3.10
conda activate vlsp
```

2. Install requirements:
```bash
pip install -r requirements.txt
```

or 

```bash
conda env create -f environment.yaml
conda env create -f environment.yaml -n vlsp
conda activate vlsp
```
