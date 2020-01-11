import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from keras.layers import Dense
from sklearn.metrics import classification_report, confusion_matrix, log_loss
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras import optimizers as opt
from keras.wrappers.scikit_learn import KerasClassifier

#prepping Data
data = pd.read_csv('clean_data.csv').reset_index()
data = data[['slope_of_peak_exercise_st_segment','resting_blood_pressure',
      'chest_pain_type','num_major_vessels','fasting_blood_sugar_gt_120_mg_per_dl',
     'resting_ekg_results','serum_cholesterol_mg_per_dl','oldpeak_eq_st_depression','sex'
     ,'age','max_heart_rate_achieved', 'exercise_induced_angina', 'heart_disease_present',
     'quant_thal']]


#separating training Values and labels
vals = data.drop(columns = 'heart_disease_present')
labs = data['heart_disease_present']

#standardizing data
scaler = StandardScaler()
scaler.fit(X = vals)
scaled_vals = scaler.transform(vals)


#splitting data into training and testing set
X = scaled_vals
y = labs
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
X_train, y_train = np.asarray(X_train), np.asarray(y_train)

'''
#defining initial AI model
inputs = tf.keras.Input(shape = (13,))
x = layers.Dense(64, activation = 'relu')(inputs)
#x = layers.BatchNormalization(axis = -1)(x)
x = layers.Dropout(rate = 0.1)(x)
#x = layers.BatchNormalization(axis = 1)(x)
#x = layers.Dense(64, activation = 'relu')(x)

predictions = layers.Dense(2, activation='softmax')(x)
model = tf.keras.Model(inputs=inputs, outputs=predictions)


model.compile( optimizer = tf.keras.optimizers.RMSprop(),
             loss = keras.losses.SparseCategoricalCrossentropy(),
             metrics = [tf.keras.metrics.SparseCategoricalAccuracy()])

'''

#model.fit(X_train, y_train, batch_size = 32, epochs = 100, verbose = 0)


#defining function that converts preditions into 1 or 0 1D matrix
def convert(preds):
    new_preds = []
    for pred in preds:
        if pred[0] > pred[1]:
            new_preds.append(0)
        else:
            new_preds.append(1)
    return new_preds

def write_results_to_file(grid_result, fname):
    pass






def grid_search2():

    def create_seq_model(num_neurons, activation):

        model = Sequential()

        model.add(Dense(num_neurons, input_dim = 13, activation = activation))
        model.add(Dense(2, activation = 'softmax'))
        model.compile(optimizer= 'adam', loss = keras.losses.SparseCategoricalCrossentropy(),
                      metrics = ['accuracy'])

        return model

    def create_seq_model_fast():

        model = Sequential()

        model.add(Dense(16, input_dim = 13, activation = 'relu'))
        model.add(Dense(2, activation = 'softmax'))
        model.compile(optimizer= 'adam', loss = keras.losses.SparseCategoricalCrossentropy(),
                      metrics = ['accuracy'])

        return model

    batch_size = [50,60,80,100]
    epochs = [10,50,100]
    num_neurons = [8,16,32,64,128]
    activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']

    model = KerasClassifier(build_fn=create_seq_model_fast, verbose=0)
    param_grid = {'batch_size' : batch_size, 'epochs' : epochs, 'num_neurons':num_neurons, 'activation': activation}
    param_grid_fast = {'batch_size' : batch_size, 'epochs' : epochs}


    grid = GridSearchCV(estimator=model, param_grid=param_grid_fast, n_jobs=-1, cv = 3, scoring= 'neg_log_loss')
    grid_result = grid.fit(X_train, y_train)

    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

    df = pd.DataFrame(grid_result.cv_results_)
    print(df.head())


grid_search2()














