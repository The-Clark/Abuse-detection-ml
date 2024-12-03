# Project Structure

The abuse detection project follows a structured organization that separates different components for clarity and maintainability. Here's a detailed overview of how the project is organized:

## Directory Details

### src/
The source directory contains our core implementation files:
- `data_generation.py`: Creates synthetic sports commentary data with balanced abuse examples
- `model_training.py`: Implements and trains our BERT-based classification model

### data/
This directory maintains our data pipeline:
- `raw/`: Stores our original synthetic dataset
- `processed/`: Contains preprocessed data ready for model training

### models/
Stores our trained models and checkpoints:
- `saved_models/`: Contains trained model weights and configurations

### notebooks/
Jupyter notebooks for analysis and visualization:
- `analysis.ipynb`: Contains detailed performance analysis and result visualization

### docs/
Project documentation and guides:
- `DOCUMENTATION.md`: Comprehensive project documentation
- `PROJECT_STRUCTURE.md`: Project organization guide (this file)

## Navigation Guide

1. Start with `README.md` for a project overview
2. Review `DOCUMENTATION.md` for detailed implementation information
3. Examine the source code in `src/` directory
4. Look at `notebooks/` for detailed analysis

## Future Structure Additions

As the project grows, we plan to add:
- `tests/`: Directory for unit tests
- `config/`: Configuration files
- `scripts/`: Utility scripts
- `examples/`: Usage examples
