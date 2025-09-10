# Weld Defect Dataset Preprocessor

A Python tool to clean, normalize, and prepare weld defect image datasets in Pascal VOC format for machine learning.

## Features

- **Class Normalization:** Automatically maps inconsistent class labels to a standardized naming convention
- **Data Validation:** Checks for and reports corrupt image files and missing XML annotation files
- **Stratified Splitting:** Creates balanced train, validation, and test splits
- **Comprehensive Reporting:** Generates detailed reports on class distribution and file issues
- **Highly Configurable:** Easy-to-use YAML/JSON configuration files

## Installation

1. Clone this repository:
```bash
git clone https://github.com/your-username/weld-defect-preprocessor.git
cd weld-defect-preprocessor
