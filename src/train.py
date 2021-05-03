import tensorflow as tf
gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)
from sklearn.preprocessing import MinMaxScaler
from sequence import create_sequence
import pandas as pd
from lstm import model
from utils import save_pkl,load_config,write_json

config = load_config()
model_config = config['hyp']


train_data = pd.read_csv('data/feature/train.csv', parse_dates=['date'])
train = train_data.drop('date', axis=1)
test_data = pd.read_csv('data/feature/test.csv', parse_dates= ['date'])
test = test_data.drop('date', axis=1)


scaler = MinMaxScaler(feature_range=(-1,1))
scaler.fit(train)
train_data_scaled = scaler.transform(train)
test_data_scaled = scaler.transform(test)

save_pkl('scaler/scaler.pkl', scaler)

X, y = create_sequence(train_data_scaled, 60, 7)
testX, testy = create_sequence(test_data_scaled, 60, 7)

model = model(X)


history = model.fit(X, y, epochs=model_config['epochs'],
             batch_size= model_config['batch_size'],
             validation_data =(testX, testy),  verbose=1)

loss  = model.evaluate(testX, testy)
score = {'val_loss' : loss}

report = {'loss' : history.history['loss'],' val_loss' : history.history['val_loss']}

write_json('report/plot.json', report)
write_json('report/score.json', score)

model.save('model/model.h5')
