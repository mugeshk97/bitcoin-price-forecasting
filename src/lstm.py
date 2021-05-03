import tensorflow as tf
gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)
from utils import load_config

config = load_config()
model_config = config['hyp']

def model(X):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(model_config['lstm_1'], activation='relu', input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
    model.add(tf.keras.layers.LSTM(model_config['lstm_2'], activation='relu', return_sequences=False))
    model.add(tf.keras.layers.Dropout(model_config['drpout']))
    model.add(tf.keras.layers.Dense(1))

    model.compile(optimizer='adam', loss=['mse'])
    return model