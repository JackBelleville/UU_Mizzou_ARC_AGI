# UU_Mizzou_ARC_AGI

This repository contains our work for the Mizzou ARC AGI Challenge, where we explore multiple approaches for solving ARC (Abstraction and Reasoning Corpus) tasks.

🚀 Project Overview

The ARC challenge evaluates an AI system’s ability to perform abstract reasoning using very small, grid-based tasks. Each task consists of:

A set of input–output training examples

One or more test inputs for which the model must predict the correct output transformation

This repository includes:

A custom SmallUNet convolutional model for early experiments

An LLM-based solver capable of inferencing transformation rules

Utility scripts for visualization and evaluation

Metric tracking for correctness, based on comparison between predicted assumptions and ground-truth transformations

🧠 LLM-Based Solver

The LLM approach attempts to infer transformation rules per task using structured prompting.
It generates:

A set of assumptions describing the pattern or rule

A predicted output grid

A comparison against ground truth

A calculated correctness score

Correctness is measured as:

(# of correct assumptions) / (total assumptions)


At the end of the run, the script prints the overall percentage accuracy across all tasks.

🎨 Visualizer

The prediction visualizer displays:

Input grid

LLM-generated predictions

Ground truth target

Correctness count per task

Final summary accuracy across the entire dataset

This is used to diagnose reasoning errors and refine the prompting strategy.

🧪 Classical Model Experiments

Early experiments used a simple UNet-style CNN.
Scripts included:

Training

Loading saved checkpoints

Running predictions

Troubleshooting PyTorch state_dict mismatches

These models were eventually replaced by the LLM-based approach due to limited performance on ARC’s abstract reasoning patterns.


📊 Evaluation

Each task is evaluated using:

Assumption correctness

Output grid match

Final correctness % across tasks

This allows us to measure how well the LLM generalizes abstract rules from only a handful of examples.

🔧 Setup & Usage

Install dependencies:

pip install -r requirements.txt

Run LLM solver with visualization:

python src/prediction_enabled_visualizer.py

🏆 Goal

The ultimate goal of this repository is to:

Explore whether large language models can abstract rules from minimal context

Compare classical and reasoning-based approaches

Contribute to research on general reasoning and AGI-related task solving