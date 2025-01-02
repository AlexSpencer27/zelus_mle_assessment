# Zelus MLE Assessment Part 2
This repository contains the solution to the Machine Learning Engineer assessment for Zelus Analytics. It includes a reproducible data pipeline, data quality checks, model training and deployment scripts, and a Dockerized solution for cross-platform execution.

## Table of Contents
1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Option 1: Use the Pre-Built Docker Image (Recommended)](#option-1-use-the-pre-built-docker-image-recommended)
    - [Options and Features](#options-and-features)
    - [Example Usage](#example-usage)
    - [Using the Shell Script (Optional)](#using-the-shell-script-optional)
4. [Option 2: Clone the Repository & Build Locally](#option-2-clone-the-repository--build-locally)
    - [Visualizing & Reproducing the Pipeline](#visualizing--reproducing-the-pipeline)

## Overview

The project follows a structured and reproducible workflow, leveraging:
- **DVC** for pipeline management and reproducibility.
- **Pytest** for comprehensive testing, including data quality checks and unit tests.
- **Docker** for cross-platform compatibility.
- **UV** for dependency management.

The primary goal is to predict expected runs per over in cricket matches, based on provided historical match data. This repository includes all steps from data parsing and preparation to model training, testing, and deployment.


## Prerequisites
- Python 3.9 or higher
- Docker
- Git (optional for cloning the repository)

## Option 1: Use the Pre-Built Docker Image (Recommended)
You can pull and use the Docker image directly from Docker Hub

1. Pull the Docker Image
```bash
docker pull schnoodfam/zelus_mle_assessment:latest
```

2. Run the Image
```bash
docker run --rm model_package \
  --model "expected_runs_model.pkl" \
  --data "data.parquet" \
  --batting-team "India" \
  --bowling-team "England" \
  --start-over 1 \
  --end-over 5 \
  --num-matches 1 \
  --match-order "newest"
```

#### Options and Features:
``--model``: Path to the trained model file inside the container. Default: expected_runs_model.pkl.

``--data``: Path to the input data file inside the container. Default: data.parquet.

``--batting-team``: The team batting in the analysis (e.g., India). Required.

``--bowling-team``: The team bowling in the analysis (e.g., England). Optional. If omitted, the most recent matches against all teams are considered.

``--start-over`` and ``--end-over``: The range of overs (inclusive) to analyze. Default: 1 to 5.

``--num-matches``: The number of most recent matches to include. Use -1 to include all matches. Default: 1.

``--match-order``: Specify whether to retrieve matches starting from the newest or oldest. Default: oldest.

#### Example Usage
Predictions for India's 5 most recent matches:

```bash
docker run --rm schnoodfam/zelus_mle_assessment:latest \
  --batting-team "India" \
  --start-over 1 \
  --end-over 5 \
  --num-matches 5 \
  --match-order "newest"
```

Predictions for India's 5 oldest matches against England:
```bash
docker run --rm schnoodfam/zelus_mle_assessment:latest \
  --batting-team "India" \
  --bowling-team "England" \
  --start-over 1 \
  --end-over 5 \
  --num-matches 5 \
  --match-order "oldest"
```

#### Using the Shell Script (Optional)
I've provided a shell script (run_model.sh) that simplifies running the exemplar query from the prompt using the Docker image. The script can be used as follows:

1. Ensure Docker is installed & the Docker daemon is running
2. Make the script executable (if not already):
```bash
chmod +x run_model.sh
```
3. Run the shell script directly
```bash
./run_model.sh
```

## Option 2: Clone the Repository & Build Locally
If you'd like to reproduce the entire pipeline locally:

1. Clone the repository Ensure you have Git LFS installed before cloning:
```bash
   git lfs install
   git clone git@github.com:<your-username>/zelus_mle_assessment.git
   cd zelus_mle_assessment
```

2. Download large files The data/provided_json folder contains large files managed with Git LFS. After cloning the repository, run:
```bash
   git lfs pull
```

3. Install dependencies
```bash
   pip install uv
   uv sync
```

Then, simply activate the resulting .venv

### Visualizing & Reproducing the Pipeline
This project uses DVC (Data Version Control) to manage and orchestrate the entire pipeline.

**Visualizing the Pipeline**

To understand the structure and flow of the pipeline, you can generate a DVC DAG visualization:
```bash
dvc dag
```
This will display a compact view of all stages and dependencies in the pipeline.

**Reproducing the Pipeline**

To execute the full pipeline and reproduce the results:
```bash
dvc repro
```

This command will:

- Parse match and innings results
- Filter and prepare training data
- Test data quality
- Train the model
- Run all tests (data quality, training, and model interaction)
- Build the Docker image for deployment

The pipeline is modular, so you can rerun specific stages if needed by specifying the target stage. DVC has a lot of documentation on these kinds of things!

I did not optimize for runtime, so it will take a few minutes from start->finish.
