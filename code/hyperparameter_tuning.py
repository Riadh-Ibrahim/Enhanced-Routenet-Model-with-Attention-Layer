import numpy as np
import tensorflow as tf
from tensorflow import keras
import random
import optuna
import os
import sys

sys.path.append('../code/')
import routenet_with_link_cap as rout


# --- 1. Fonction de traitement des données ---
def parse(serialized, target='delay'):
    """
    Target est la variable prédite (exemple: 'delay').
    """
    with tf.device("/cpu:0"):
        with tf.name_scope('parse'):
            features = tf.parse_single_example(
                serialized,
                features={
                    'traffic': tf.VarLenFeature(tf.float32),
                    target: tf.VarLenFeature(tf.float32),
                    'link_capacity': tf.VarLenFeature(tf.float32),
                    'links': tf.VarLenFeature(tf.int64),
                    'paths': tf.VarLenFeature(tf.int64),
                    'sequences': tf.VarLenFeature(tf.int64),
                    'n_links': tf.FixedLenFeature([], tf.int64),
                    'n_paths': tf.FixedLenFeature([], tf.int64),
                    'n_total': tf.FixedLenFeature([], tf.int64)
                })
            for k in ['traffic', target, 'link_capacity', 'links', 'paths', 'sequences']:
                features[k] = tf.sparse_tensor_to_dense(features[k])
                if k == 'traffic':
                    features[k] = (features[k] - 0.17) / 0.13
                if k == 'link_capacity':
                    features[k] = (features[k] - 25.0) / 40.0

    return {k: v for k, v in features.items() if k != target}, features[target]


def read_dataset(sample_file):
    """
    Lecture et parsing des fichiers TFRecord.
    """
    ds = tf.data.TFRecordDataset(sample_file)
    ds = ds.map(lambda buf: parse(buf))
    ds = ds.batch(1)
    it = ds.make_initializable_iterator()
    return it


# --- 2. Entraînement et Évaluation ---
def train_and_evaluate(hparams, train_path, val_path):
    """
    Entraîne le modèle avec des hyperparamètres donnés et retourne la performance.
    """
    parsed_hparams = rout.hparams.parse(
        f"l2={hparams['l2']},dropout_rate={hparams['dropout_rate']},"
        f"link_state_dim={hparams['link_state_dim']},path_state_dim={hparams['path_state_dim']},"
        f"readout_units={hparams['readout_units']},learning_rate={hparams['learning_rate']},T={hparams['T']}"
    )
    
    # Charger le modèle RouteNet
    model = rout.ComnetModel(parsed_hparams)
    model.build()

    # Chargement des ensembles d'entraînement et validation
    train_it = read_dataset(random.choice(os.listdir(train_path)))
    val_it = read_dataset(random.choice(os.listdir(val_path)))

    # Session TensorFlow
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.local_variables_initializer())
        sess.run(tf.compat.v1.global_variables_initializer())

        # Entraînement
        for epoch in range(10):
            sess.run(train_it.initializer)
            try:
                while True:
                    features, label = sess.run(train_it.get_next())
                    sess.run(model.train_op, feed_dict={model.inputs: features, model.labels: label})
            except tf.errors.OutOfRangeError:
                pass

        # Validation
        sess.run(val_it.initializer)
        val_losses = []
        try:
            while True:
                features, label = sess.run(val_it.get_next())
                val_loss = sess.run(model.loss, feed_dict={model.inputs: features, model.labels: label})
                val_losses.append(val_loss)
        except tf.errors.OutOfRangeError:
            pass

    return np.mean(val_losses)

# --- 3. Fonction d'objectif pour Optuna ---
def objective(trial):
    """
    Fonction utilisée par Optuna pour rechercher les meilleurs hyperparamètres.
    """
    hparams = {
        "l2": trial.suggest_loguniform("l2", 0.01, 1.0),
        "dropout_rate": trial.suggest_uniform("dropout_rate", 0.3, 0.7),
        "link_state_dim": trial.suggest_categorical("link_state_dim", [16, 32, 64]),
        "path_state_dim": trial.suggest_categorical("path_state_dim", [16, 32, 64]),
        "readout_units": trial.suggest_categorical("readout_units", [128, 256, 512]),
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.0001, 0.01),
        "T": trial.suggest_int("T", 4, 16)
    }
    train_path = '/nsfnet/train/dataset/'
    val_path = '/nsfnet/val/dataset/'

    # Évaluer le modèle avec ces hyperparamètres
    val_loss = train_and_evaluate(hparams, train_path, val_path)
    return val_loss  # Minimiser la perte


# --- 4. Optimisation avec Optuna ---
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)

# Affichage des meilleurs hyperparamètres
print("Best hyperparameters:", study.best_params)

# --- 5. Entraînement Final avec les Meilleurs Hyperparamètres ---
best_hparams = study.best_params
parsed_hparams = rout.hparams.parse(
    f"l2={best_hparams['l2']},dropout_rate={best_hparams['dropout_rate']},"
    f"link_state_dim={best_hparams['link_state_dim']},path_state_dim={best_hparams['path_state_dim']},"
    f"readout_units={best_hparams['readout_units']},learning_rate={best_hparams['learning_rate']},T={best_hparams['T']}"
)

model = rout.ComnetModel(parsed_hparams)
model.build()

# Réentraînement avec les meilleurs hyperparamètres
train_dataset = '/path/to/full/train/dataset/'
model.fit(train_dataset, epochs=50)  
