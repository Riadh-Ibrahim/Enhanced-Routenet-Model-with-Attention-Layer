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
```
# Model Training

## Install Dependencies
To install the required dependencies, run:

```bash
pip install -r requirements.txt
```
## Download the Dataset
To download the NSFNET dataset, use the following commands:

```bash
wget "http://knowledgedefinednetworking.org/data/datasets_v0/nsfnet.tar.gz"
tar -xvzf nsfnet.tar.gz
```

## Set Dataset Path
Once downloaded, set the path to the dataset directory by defining an environment variable:

```bash
export PATH_TO_DATASET='/path/to/your/dataset'
```
## Train the Model
To train the model, you can use the provided training script. For instance, to train the model using the NSFNET dataset:

```bash
./run_routenet.sh train nsfnetbw 50000
```

## Monitor Training
Use TensorBoard to monitor the training process:

```bash
tensorboard --logdir <path_to_logs>
```
Then open [http://localhost:6006/](http://localhost:6006/) in your web browser to view the training logs.

## Usage
Once the model is trained, you can use it to make predictions on arbitrary samples from the dataset. We provide a Jupyter notebook in the `demo_notebooks` directory that demonstrates how to load the trained model and make predictions.

## Example Commands

- To train the model:

```bash
./run_routenet.sh train nsfnetbw 50000
```
- To evaluate the model on a different topology:

```bash
./run_routenet.sh train_multiple nsfnetbw synth50bw output_logdir 100000
```
## To visualize training progress with TensorBoard:

```bash
tensorboard --logdir <path_to_logs>
```
## Conclusion
This enhanced **RouteNet** model with attention mechanisms improves the accuracy and efficiency of network performance predictions. The model is trained on the NSFNET dataset and can be easily modified to support different datasets. This implementation includes all necessary tools for training, evaluation, and visualization of model performance.

