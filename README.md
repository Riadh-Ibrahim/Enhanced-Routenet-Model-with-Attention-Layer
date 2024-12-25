# Enhanced-Routenet-Model-with-Attention-Layer

This repository contains the implementation of an enhanced version of the **RouteNet** model for network performance prediction. By integrating **attention mechanisms**, the model improves predictive accuracy and converges faster than the baseline model. The project includes hyperparameter optimization, data handling for the **NSFNET dataset**, and comprehensive evaluations of model performance using metrics such as **R²** and **MAPE**.

## Features
- **Attention Mechanism**: Integrates attention layers to improve model performance by focusing on relevant parts of the input.
- **Hyperparameter Optimization**: Uses Optuna for optimizing hyperparameters to achieve the best performance.
- **NSFNET Dataset**: The model is trained and evaluated on the **NSFNET** dataset.
- **Model Performance Evaluation**: Evaluates model performance using metrics like **R²** and **MAPE**.
- **Efficient Training**: Improved training process for faster convergence.

## Dataset
The dataset used in this project is the **NSFNET** dataset, which is publicly available and used to simulate network performance. You can download it using the following commands:

```bash
wget "http://knowledgedefinednetworking.org/data/datasets_v0/nsfnet.tar.gz"
tar -xvzf nsfnet.tar.gz
