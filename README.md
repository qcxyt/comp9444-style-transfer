# COMP9444 Image Style Transfer

## Download and run

Click Code > Download ZIP and extract the archive. Open a terminal in the
extracted folder containing train_baseline.py. The 440 dataset images and split
manifests are included. No manual file renaming is needed.

Create an environment with Python 3.11 and install a CUDA-compatible PyTorch and
torchvision build for your NVIDIA GPU. The tested versions are torch
2.11.0+cu128 and torchvision 0.26.0+cu128.

    conda create -n stytr2 python=3.11 -y
    conda activate stytr2
    pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
    pip install -r requirements.txt

The scripts currently require an NVIDIA CUDA GPU. Training uses BF16 and requires
a GPU supporting it. Training took approximately 24 minutes on our 8 GB RTX 5060
Laptop GPU.

## Evaluate the trained models

    python evaluate_full_grid.py --models A B --output evaluation_ab

A is baseline; B is improvement1. Required generator weights download automatically
from GitHub Releases on first use. Frozen VGG weights download from the authors'
Google Drive. First use requires internet access and approximately 365 MB of
downloads; subsequent runs use the local files.

Each split uses 32 photographs × 12 paintings = 384 pairs. Outputs include
summary.json, pairs.csv and by_style.csv. This command loads trained weights and
does not retrain. Optional control/C evaluation requires those local experiment
weights.

## Train baseline

    python train_baseline.py --crop256 --steps 5000 --content-weight 7 --output baseline_run

## Train improvement1

    python train_baseline.py --crop256 --steps 5000 --content-weight 10 --output improvement1_run

Training uses random generator initialization and frozen pretrained VGG.
The first training run automatically downloads VGG if needed.
Output folders contain trained weights, metrics and example images.
The evaluator's A/B labels refer to the published trained weights, not new
baseline_run or improvement1_run outputs. To evaluate a new run with it, edit
the corresponding model path in evaluate_full_grid.py.

## Data and results

The included subset contains COCO 2014 content images and WikiArt style images.
Content train/validation/test counts are 256/32/32. Style counts are 96/12/12.
CSV manifests identify source dataset rows and split assignments.

[Improvement1 results](experiments/improvement1/README.md) include the recorded
metrics, comparison images and limitations.

## Attribution and collaboration

Model code is adapted from https://github.com/diyiiyiii/StyTR-2
(Deng et al., CVPR 2022). The original README identifies authors and paper.
Our contributions include local training and evaluation scripts, dataset splits
and experimental loss options. The local loss is disabled in baseline training.
Use feature branches and pull requests for team changes.
