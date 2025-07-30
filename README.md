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
1. Install requirements:

```bash
conda env create -f environment.yaml -n vlsp
conda activate vlsp
```
or 
```bash
conda create -n vlsp python=3.10
conda activate vlsp
pip install -r requirements.txt
```
2. Download checkpoint
```bash
cd VLSP2025
wget https://github.com/addf400/files/releases/download/beit3/beit3.spm
mkdir -p checkpoint
mv beit3.spm checkpoint/beit3.spm
```
