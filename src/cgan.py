import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

latent_dim = 100
num_classes = 2

def build_generator(num_features):
    noise = Input(shape=(latent_dim,))
    label = Input(shape=(1,), dtype='int32')

    label_embedding = layers.Embedding(num_classes, latent_dim)(label)
    label_embedding = layers.Flatten()(label_embedding)

    model_input = layers.multiply([noise, label_embedding])
    x = Dense(128, activation='relu')(model_input)
    output = Dense(num_features, activation='sigmoid')(x)

    return models.Model([noise, label], output)

def build_discriminator(num_features):
    data_input = Input(shape=(num_features,))
    label = Input(shape=(1,), dtype='int32')

    label_embedding = layers.Embedding(num_classes, num_features)(label)
    label_embedding = layers.Flatten()(label_embedding)

    combined = layers.multiply([data_input, label_embedding])
    x = Dense(128, activation='relu')(combined)
    output = Dense(1, activation='sigmoid')(x)

    return models.Model([data_input, label], output)