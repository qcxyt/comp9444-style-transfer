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

Download [baseline_dataset.zip](https://github.com/qcxyt/comp9444-style-transfer/releases/download/trained-models-v1/baseline_dataset.zip) from the release.
Extract it from the repository root:

    Expand-Archive -LiteralPath .\baseline_dataset.zip -DestinationPath .

The archive contains datasets/baseline/, including 440 images and two split manifests.
Content train/val/test counts are 256/32/32; style counts are 96/12/12.

Download the authors' vgg_normalised.pth using the link in
StyTR-2-main/readme.md and place it in StyTR-2-main/experiments/.

## Train baseline A

Run from the repository root:

    conda activate stytr2
    python train_baseline.py --crop256 --steps 5000 --content-weight 7 --output baseline_run

This trains from random generator weights, with a frozen pretrained VGG.
Expected time on our RTX 5060 Laptop GPU is approximately 24 minutes.
Outputs include weights, training logs, evaluation metrics and sample images.
Training weights and datasets are excluded from Git.

For experiments requiring identical starting weights, give both runs the same
--initial-state path. The first run creates it and subsequent runs load it.

## Evaluation scope

The training script evaluates 32 fixed pairs per split. Full-grid evaluation
uses every combination of 32 photos and 12 paintings = 384 pairs per split.
It loads trained models and computes metrics without updating weights.

After placing the downloaded model weights as described below, run:

    python evaluate_full_grid.py --models A B --output evaluation_ab

A refers to the baseline and B to improvement1. This command evaluates only A and B.
Use --models A to evaluate the baseline alone.
Results are summary.json (means), pairs.csv (every combination), and
by_style.csv (means for each reference artwork).
The default without --models includes control and C, which require additional local weights.
The environment, frozen VGG and dataset are also required.

## Collaboration

Create feature branches and open pull requests for team review.
Record the dataset split, parameters and results with each experiment.

## Dataset manifests

The committed datasets/baseline/content.csv and style.csv record the exact
source rows and split assignments used in the existing experiments. Use these
assignments when preparing images. The earlier prepare_baseline.py downloader
is experimental and is not included in this baseline commit.

## Scope of this baseline commit

train_baseline.py defaults to the standard objective with local-style-weight 0.
The code also contains optional content-weight and local-loss switches from
subsequent experiments. They are inactive in the baseline command above.

## improvement1

See [improvement1](experiments/improvement1/README.md) for reproduction
commands, full-grid results and visual comparisons.

## Trained weights

Download baseline.pth and improvement1.pth from [trained models](https://github.com/qcxyt/comp9444-style-transfer/releases/tag/trained-models-v1).
Place baseline.pth as step1_256_5000/stytr2.pth and improvement1.pth as step2_content10/stytr2.pth for the existing evaluation script.
These weights contain the trained generator. The frozen VGG weights are also required for evaluation.
