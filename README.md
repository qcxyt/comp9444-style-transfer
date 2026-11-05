# COMP9444 Image Style Transfer

Baseline A uses StyTR² with 256-pixel training crops and 5,000 updates.

## Attribution

Model implementation is adapted from https://github.com/diyiiyiii/StyTR-2
(Deng et al., CVPR 2022). See StyTR-2-main/readme.md for the original authors and paper.
Project additions include local training, fixed evaluation pairs and experimental loss options.
The local loss option is disabled for baseline A.

## Local setup

The tested environment is Anaconda stytr2, Python 3.11, PyTorch 2.11.0+cu128,
torchvision 0.26.0+cu128 on an NVIDIA GPU. Install a CUDA-compatible PyTorch build.
Other imports include Pillow, NumPy, SciPy, timm and requests.

Place the dataset under datasets/checkpoint1 with content and style folders,
each containing train, val and test. Training uses 256 content and 96 style images.
Validation and test each use 32 content images and 12 style images.
The current data manifests and setup script should be reviewed before recreating
the subset. Share the existing split manifests with teammates for identical splits.

Download the authors' vgg_normalised.pth using the link in
StyTR-2-main/readme.md and place it in StyTR-2-main/experiments/.

## Train baseline A

Run from the repository root:

    conda activate stytr2
    python run_step1.py --crop256 --steps 5000 --content-weight 7 --output baseline_run

This trains from random generator weights, with a frozen pretrained VGG.
Expected time on our RTX 5060 Laptop GPU is approximately 24 minutes.
Outputs include weights, training logs, evaluation metrics and sample images.
Training weights and datasets are excluded from Git.

For experiments requiring identical starting weights, give both runs the same
--initial-state path. The first run creates it and subsequent runs load it.

## Evaluation scope

The training script evaluates 32 fixed pairs per split. evaluate_full_grid.py
evaluates all 32 × 12 = 384 combinations per split, but currently expects all
A/B/control/C weight folders to exist locally. It is an experiment comparison
script, not a prerequisite for training the baseline.

## Collaboration

Create feature branches and open pull requests for team review.
Record the dataset split, parameters and results with each experiment.

## Dataset manifests

The committed datasets/checkpoint1/content.csv and style.csv record the exact
source rows and split assignments used in the existing experiments. Use these
assignments when preparing images. The earlier prepare_checkpoint1.py downloader
is experimental and is not included in this baseline commit.

## Scope of this baseline commit

run_step1.py defaults to the standard objective with local-style-weight 0.
The code also contains optional content-weight and local-loss switches from
subsequent experiments. They are inactive in the baseline command above.
