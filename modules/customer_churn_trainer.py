import os

import tensorflow as tf
from keras.utils.vis_utils import plot_model
import tensorflow_transform as tft

from customer_churn_transform import (
    CATEGORICAL_FEATURES,
    LABEL_KEY,
    NUMERICAL_FEATURES,
    transformed_name
)

def get_model(show_summary=True):
    """
    This function defines a Keras model and returns the model as a 
    Keras object
    """

    # one-hot categorical features
    input_features = []
    for key, dim in CATEGORICAL_FEATURES.items():
        input_features.append(
            tf.keras.Input(shape=(dim + 1), name=transformed_name(key))
        )

    for feature in NUMERICAL_FEATURES: 
        input_features.append(
            tf.keras.Input(shape=(1,), name=transformed_name(feature))
        )

    concatenate = tf.keras.layers.concatenate(input_features)
    deep = tf.keras.layers.Dense(256, activation='relu')(concatenate)
    deep = tf.keras.layers.Dense(64, activation='relu')(deep)
    deep = tf.keras.layers.Dense(16, activation='relu')(deep)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid') (deep)

    model = tf.keras.models.Model(input=input_features,outputs=outputs)
    model.compile(
        optimizers=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.BinaryAccuracy()]
    )

    if show_summary:
        model.summmary()
    
    return model

def gzip_reader_fn(filenames):
    """Load compressed data"""
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')
