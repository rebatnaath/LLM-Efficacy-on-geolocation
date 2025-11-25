# Evaluating LLMs for Geolocation

This repository contains the research, code, and data for our study: **"Evaluating the Efficacy of Large Language Models for Geolocation Identification"**.

## Overview

We investigate whether "everyday" AI tools—specifically **Gemini 2.5 Pro** and **GPT-5**—can effectively determine the geographic location of an image without specialized training. Unlike previous research that focuses on custom-built models, we evaluate the out-of-the-box capabilities of the assistants millions of people use daily.

## Repository Structure

*   **`dataset/`**: Contains the test images and the ground truth mapping (`map.md`).
*   **`latex/`**: The LaTeX source code for the research paper, including diagrams and bibliographies.
*   **`scripts/`**: Python scripts used to automate the evaluation pipeline (API interaction, coordinate extraction, and error calculation).

## Getting Started

### Prerequisites
*   Python 3.x
*   LaTeX (for compiling the paper)

### Running the Evaluation
To run the automated geolocation pipeline:
```bash
python3 scripts/geolocation_pipeline.py
```
*Note: The script currently uses mock API calls for demonstration. You will need to configure your own API keys in the script to run live queries.*

### Compiling the Paper
To generate the PDF of the research paper:
```bash
cd latex
./compile.sh
```

## Key Findings
*   **GPT-5** exhibits a "Bounded Error" profile, with 100% of predictions falling within 250 km.
*   **Gemini 2.5 Pro** shows higher variance but maintains strong regional understanding.
*   Both models achieved **29.1%** accuracy at the street level ($< 1$ km).

## Authors
*   Rohan Poudel
*   Om Goswami
