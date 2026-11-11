# Wine Quality Prediction - ML Zoomcamp Project

## 📌 Project Overview

This project is the final assignment for the **Machine Learning Zoomcamp**. The goal is to predict whether a wine (red or white) is of "high quality" (score $\ge$ 7) based on its physicochemical properties, such as acidity, residual sugar level, pH, and alcohol content.

This is a **binary classification** problem. The model was developed following a rigorous process of experimentation, hyperparameter tuning (to prevent overfitting), and validation. This culminated in a **Random Forest Classifier** that was packaged and deployed via a REST API using Flask and Gunicorn, entirely containerized with Docker.

## 📊 Dataset

The data is sourced from the famous [Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality) from the UCI Machine Learning Repository.

- `winequality-red.csv`: Red wine samples.
- `winequality-white.csv`: White wine samples.

## 📂 Repository Structure

```text
.
├── data/
│   ├── winequality-red.csv
│   └── winequality-white.csv
├── notebook.ipynb         # EDA, feature engineering, and model selection
├── train.py               # Script to train and export the final model
├── predict.py             # Flask script to serve the model
├── test_predict.py        # Script to send requests to the local API
├── Pipfile                # Environment dependencies
├── Pipfile.lock           # Locked dependency versions
├── Dockerfile             # Docker containerization configuration
└── README.md              # This document
```

## ⚙️ 1. Environment Setup (Pipenv)

This project uses `pipenv` for dependency management and requires **Python 3.13.5**.

1. Clone the repository and navigate to the project directory:

   ```bash
   git clone https://github.com/rodric-ds-dev/wine-quality-ml
   ```

2. Install the dependencies and activate the virtual environment:
   ```bash
   pipenv install
   pipenv shell
   ```

## 🧠 2. Model Training

If you wish to retrain the model from scratch and generate the binary artifact (`model_rf.bin`), run the training script:

```bash
python train.py
```

_This will read the data, apply the unified preprocessing steps, train the champion Random Forest model, and export the serialized artifact._

## 🐳 3. Local Deployment with Docker

To ensure reproducibility and avoid compatibility issues across different OS, the predictive API is containerized.

1. Build the Docker image:

   ```bash
   docker build -t wine-quality-api .
   ```

2. Run the container, exposing port `9696`:
   ```bash
   docker run -it --rm -p 9696:9696 wine-quality-api
   ```
   _The Gunicorn server will start and be ready to receive HTTP requests._

## 🧪 4. API Testing

With the container running, open a new terminal window and run the test script to send a JSON payload with a wine's features and get the prediction:

```bash
python test_predict.py
```

**Expected output:**

```text
High quality probability: 0.0199
Is high quality?: False
```
