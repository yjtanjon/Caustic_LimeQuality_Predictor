
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import datetime
import os

from scripts import test_model_performance, scale_data
import csv
import pickle
import tensorflow as tf
from tensorflow import keras
from keras import layers, regularizers
from keras.layers import Layer
from keras import backend as K

from keras.models import Sequential
from keras.layers import LSTM

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn import svm, tree
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.gaussian_process import kernels
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.naive_bayes import GaussianNB
import joblib

from smt.surrogate_models import RMTB, RMTC, RBF, KRG, KPLS, KPLSK
from smt.applications.mfk import MFK, NestedLHS

from sklearn.metrics import r2_score
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from scipy.integrate import simpson
from statsmodels.nonparametric.smoothers_lowess import lowess
from numba import jit






caustic = pd.read_pickle('JanJune_CausticData.pkl')
# print(caustic)
# caustic.index = caustic.index.apply(lambda x: datetime.strptime(x, "%d/%m/%Y %H:%M:%S"))

caustic.index = pd.to_datetime(caustic.index)

# raw_data = raw_data.drop(index=raw_data[raw_data['Elevator_Load'] < min_Elevator_Load].index)
caustic= caustic[caustic['GL_TTA_auto'] >= 115]
caustic= caustic[caustic['screw_speed'] >= 2]
print("loaded .pkl file")



# caustic = pd.read_pickle('caustic.pk1')






# @jit(nopython=True)
def relativevar(x):
    mu = np.mean(x)
    if mu < 1e-8:
        return np.nan
    else:
        return np.std(x) / np.mean(x)


# def hampel(df, var, rolling_width, threshold):
#     df['rel'] = df[var].rolling(rolling_width, center=True).apply(relativevar,
#                                                                   engine='numba', raw=True,
#                                                                   engine_kwargs=dict(nopython=True))
#
#     idx = (np.linspace(0, len(df) - 1, len(df)).astype('int')[(df['rel'] > threshold).to_numpy()])
#     df.iloc[idx, df.columns.get_loc(var)] = np.nan * np.ones(len(idx))
#     df = df.interpolate(method='index')
#     return df

def hampel(df, var, rolling_width, threshold):
    error_rows = []  # List to store the indices of rows causing errors

    try:
        # Calculate the relative variance using the rolling window and custom function
        df['rel'] = df[var].rolling(rolling_width, center=True).apply(relativevar)

        # Find the indices where the relative variance exceeds the threshold
        idx = df.index[df['rel'] > threshold].tolist()

        # Set the corresponding 'var' column values to NaN
        # df.loc[idx, var] = np.nan

        # Drop the rows where the relative variance exceeds the threshold
        df.drop(idx, inplace=True)

        # Interpolate missing values using the index
        df = df.interpolate(method='index')

    except Exception as e:
        print(f"An error occurred: {e}")
        error_rows = idx  # Store the index where the error occurred

    return df, error_rows

# caustic = hampel(caustic, 'lime_pct', 5, 0.01)
# caustic = hampel(caustic, 'screw_speed', 20, 0.01)
# caustic = hampel(caustic, 'lime_T', 5, 0.004)
# caustic = hampel(caustic, 'slaker_T', 10, 0.002)
# caustic = hampel(caustic, 'classifier_T', 20, 0.005)
# caustic = hampel(caustic, 'GL_T', 10, 0.005)

columns_to_process = [
    ('lime_pct', 5, 0.01),
    ('screw_speed', 20, 0.01),
    ('lime_T', 5, 0.004),
    ('slaker_T', 10, 0.002),
    ('classifier_T', 20, 0.005),
    ('GL_T', 10, 0.005)
]
error_log = []
for col, rolling_width, threshold in columns_to_process:
    caustic, error_rows = hampel(caustic, col, rolling_width, threshold)

    if error_rows:
        # error_msg = f"Errors occurred for column '{col}' in rows: {error_rows}\n"
        # error_log.append(error_msg)
        for row_index in error_rows:
            # value = caustic.loc[row_index, col]
            # value_msg = f"Value causing issue at row {row_index}: {value}\n"
            value_msg = f"Value causing issue at row {row_index}\n"
            error_log.append(value_msg)
    else:
        success_msg = f"No errors detected for column '{col}'.\n"
        error_log.append(success_msg)

# Export error log to a text file
with open("error.txt", "w") as f:
    f.writelines(error_log)

caustic.to_csv("hampel.csv")
print("Hampel Function Succeeded")

slaker_volume = 43  # [m³]
c1_volume = 138  # [m³]
c2_volume = 138  # [m³]
c3_volume = 46  # [m³]
c4_volume = 46  # [m³]
c5_volume = 46  # [m³]

caustic['lime_flow_T'] = caustic['GL_flow'] * (caustic['classifier_T'] - caustic['slaker_T']) / 11465  # [m3/h]
caustic['lime_flow_rpm'] = caustic['screw_speed'] / 2.53  # [m3/h]
slaker_flow = (caustic['GL_flow'] / 1000 / 60) + 9.75 / 1000 + caustic['lime_flow_T'] / 3600
flow = (caustic['GL_flow'] / 1000 / 60) + 9.75 / 1000 #in m3/s

caustic['slaker_residence_time'] = slaker_volume / slaker_flow
caustic['c1_residence_time'] = (slaker_volume + c1_volume) / flow
caustic['c5_residence_time'] = (slaker_volume + c1_volume + c2_volume + c3_volume + c4_volume + c5_volume) / flow

caustic["lime_mass_flow"] = caustic['screw_speed'] * 14 + 75.333  # mass flow in kg/min from RPM
caustic["lime_dosage"] = 100 * caustic["lime_mass_flow"] \
                         / ((caustic['GL_flow']/1000) * 0.905 *(100 - caustic['GL_S'])/100*caustic['GL_TTA_auto'])



caustic.to_csv("Caustic.CSV")

caustic5 = caustic[['5caust_EA', '5caust_TTA', '5caust_CE', '5caust_AA', '5caust_sulp', 'c5_residence_time']]
caustic1 = caustic[
    ['1caust_EA', '1caust_AA', '1caust_TTA', '1caust_CE', '1caust_sulp', '1caust_EA_auto', '1caust_TTA_auto',
     '1caust_CE_auto', 'c1_residence_time']]
slaker4 = caustic[['classifier_T', 'slaker_residence_time']]
caustic = caustic.drop(
    ['5caust_EA', '5caust_TTA', '5caust_CE', '5caust_AA', '5caust_sulp', 'c5_residence_time', '1caust_EA', '1caust_AA',
     '1caust_TTA', '1caust_CE', '1caust_sulp', '1caust_EA_auto', '1caust_TTA_auto', '1caust_CE_auto',
     'c1_residence_time','classifier_T', 'slaker_residence_time'], axis=1)

caustic1.to_csv("Caustic1.CSV")
caustic5.to_csv("Caustic5.CSV")

t_res_shift5 = pd.to_timedelta(caustic5['c5_residence_time'] * 1e9)
# t_res_shift5 = t_res_shift5 * caustic5.index.freq  # Ensure units match
caustic5.index -= t_res_shift5
caustic5 = caustic5.resample('1T').mean()

t_res_shift1 = pd.to_timedelta(caustic1['c1_residence_time'] * 1e9)
# t_res_shift1 = t_res_shift1 * caustic1.index.freq  # Ensure units match
caustic1.index -= t_res_shift1
caustic1 = caustic1.resample('1T').mean()

t_res_shift_slaker = pd.to_timedelta(slaker4['slaker_residence_time'] * 1e9)
# t_res_shift1 = t_res_shift1 * caustic1.index.freq  # Ensure units match
slaker4 .index -= t_res_shift_slaker
slaker4  = slaker4.resample('1T').mean()


inx = pd.Index.intersection(caustic.index, caustic5.index)
caustic = caustic.reindex(inx)
caustic1 = caustic1.reindex(inx)
caustic5 = caustic5.reindex(inx)
slaker4 = slaker4.reindex(inx)
caustic_shifted = pd.concat([caustic, slaker4, caustic1, caustic5], axis=1)



caustic_shifted.to_pickle('caustic_shifted.pk1')

caustic = pd.read_pickle('caustic_shifted.pk1')  # .resample('1T').mean()
# inputs = caustic[['lime_pct', 'screw_speed', 'lime_T', 'slaker_T', 'classifier_T', 'GL_T',
#        'GL_flow', 'GL_T_in', 'GL_TTA', 'GL_EA', 'GL_TTA_auto','deadload',
#        'carbonate_deadload', 'slaker_residence_time', 'c1_residence_time',
#        'c5_residence_time','lime_flow_T', 'lime_flow_rpm']]

caustic.to_csv("whyisslakernothere.csv")

caustic["slaker_diff_temp"] = caustic["classifier_T"] - caustic["slaker_T"]


caustic.to_csv("caustic_after_operations.csv")


inputs = caustic[['classifier_T', 'slaker_T','screw_speed','GL_flow',"slaker_diff_temp"]]
# , 'deadload',
#    'carbonate_deadload', 'slaker_residence_time', 'c1_residence_time',
#    'c5_residence_time','lime_flow_T', 'lime_flow_rpm','1caust_EA', '1caust_AA', '1caust_TTA', '1caust_CE',
#    '1caust_sulp', '1caust_EA_auto', '1caust_TTA_auto', '1caust_CE_auto']]

outputs = caustic[["lime_dosage"]]
# outputs = caustic[['5caust_CE']]
# outputs = caustic[['5caust_TTA']]

print("shifted function succeeded")

inputs.to_pickle('inputs.pk1')
outputs.to_pickle('outputs.pk1')

inputs.to_csv("inputs.CSV")
outputs.to_csv("outputs.CSV")

# inputs.to_csv('inputs.csv')
inputs = pd.read_pickle('inputs.pk1')
outputs = pd.read_pickle('outputs.pk1')

na_inx = np.array(~outputs.isna().any(axis=1))

inputs = inputs[na_inx]
outputs = outputs[na_inx]

print_list = outputs.columns

for i in inputs.columns:
    inputs[i + '_mean'] = inputs[i].copy()
    inputs[i + '_std'] = inputs[i].std()
    inputs = inputs.drop(columns=i, axis=0)

for i in outputs.columns:
    outputs[i + '_mean'] = outputs[i].copy()
    outputs[i + '_std'] = outputs[i].std()
    outputs = outputs.drop(columns=i, axis=0)

print("mean and std succeeded")


def load_pickled_data(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data


def convert_to_csv(data, csv_file_path):
    with open(csv_file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Assuming 'data' is a list of dictionaries
        if isinstance(data, list) and all(isinstance(item, dict) for item in data):
            # Write the header row
            writer.writerow(data[0].keys())
            # Write each data row
            for row in data:
                writer.writerow(row.values())

def train_model(model, train_features, train_labels, verbose=True):
    history = model.fit(
        train_features,
        train_labels,
        validation_split=0.2,
        verbose=1, epochs=300)

    if verbose:
        plt.figure()
        plt.semilogy(history.history['loss'], label='loss')
        plt.semilogy(history.history['val_loss'], label='val_loss')
        # plt.ylim([0, 10])
        plt.xlabel('Epoch')
        plt.ylabel('Error')
        plt.legend()
        plt.grid(True)

        print("Should print train_model here")
        plt.show()


    return model


def test_model(model, test_features, output_cols):
    model_output = model.predict(test_features)
    model_output = pd.DataFrame(data=model_output, columns=output_cols)
    return model_output


class RBFLayer(Layer):
    def __init__(self, units, gamma, **kwargs):
        super(RBFLayer, self).__init__(**kwargs)
        self.units = units
        self.gamma = K.cast_to_floatx(gamma)

    def build(self, input_shape):
        self.mu = self.add_weight(name='mu',
                                  shape=(int(input_shape[1]), self.units),
                                  initializer='uniform',
                                  trainable=True)
        super(RBFLayer, self).build(input_shape)

    def call(self, inputs):
        diff = K.expand_dims(inputs) - self.mu
        l2 = K.sum(K.pow(diff, 2), axis=1)
        res = K.exp(-1 * self.gamma * l2)
        return res

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)


def build_model(N_in, N_out, NN):
    inputs = keras.Input(shape=(N_in,))

    N_layer = int(NN * 0.5)

    # dense = RBFLayer(10, 0.5)(inputs) # dropout layer?
    # dense2 = RBFLayer(10, 0.5)(dense) # dropout layer?

    # dense = layers.Dense(N_layer, activation="relu")(inputs) # dropout layer?
    # dense2 = layers.Dense(N_layer, activation="relu")(dense) # dropout layer?

    dense = layers.Dense(N_layer, activation="relu", activity_regularizer=regularizers.L2(1e-2))(
        inputs)  # dropout layer?
    dense2 = layers.Dense(N_layer, activation="relu", activity_regularizer=regularizers.L2(1e-2))(
        dense)  # dropout layer?
    # dense3 = layers.Dense(20, activation="relu")(dense2) # dropout layer?
    # dense4 = layers.Dense(20, activation="relu")(dense3) # dropout layer?
    # dense5 = layers.Dense(20, activation="relu")(dense4) # dropout layer?
    outputs = layers.Dense(N_out, activation="linear")(dense2)

    model = keras.Model(inputs=inputs, outputs=outputs, name="model")

    # model.compile(loss='mean_absolute_error',
    #               optimizer=tf.keras.optimizers.Adam(0.001))

    model.compile(loss='mean_squared_error',
                  optimizer=tf.keras.optimizers.Adam(0.005))
    keras.utils.plot_model(model, show_shapes=True)
    model.summary()
    # save_model(model, 'lime_screw_347.joblib')
    return model


# model = Sequential()
# model.add(LSTM(64,return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[-1])))
# # model.add(Dropout(0.5))
# model.add(LSTM(20,return_sequences=False))
# # model.add(Dropout(0.5))
# model.add(Dense(1))
# model.compile(loss='mse', optimizer='rmsprop')



def train_model(model, train_features, train_labels, verbose=True):
    model.set_training_values(train_features.to_numpy(), train_labels.to_numpy())
    model.train()

    return model


def test_model(model, test_features, output_cols):
    model_output = model.predict_values(test_features.to_numpy())
    model_output = pd.DataFrame(data=model_output, columns=output_cols)
    return model_output


def build_model(N_in, N_out):
    model = KRG(theta0=[1e-2] * N_out, poly='quadratic', n_start=100, print_prediction=False)
    # save_model(model, 'lime_screw_373.joblib')
    return model



def train_model(model, train_features, train_labels, verbose=True):
    model.set_training_values(train_features.to_numpy(), train_labels.to_numpy())
    model.train()

    return model


def test_model(model, test_features, output_cols):
    model_output = model.predict_values(test_features.to_numpy())
    model_output = pd.DataFrame(data=model_output, columns=output_cols)
    return model_output


def build_model(N_in, N_out, NN):
    model = RBF(d0=0.5, poly_degree=0, reg=10)
    # save_model(model, 'lime_screw_392.joblib')
    return model





def train_model(model, train_features, train_labels, verbose=True):
    model.set_training_values(train_features.to_numpy(), train_labels.to_numpy())
    model.train()

    return model


def test_model(model, test_features, output_cols):
    model_output = model.predict_values(test_features.to_numpy())
    model_output = pd.DataFrame(data=model_output, columns=output_cols)
    return model_output


def build_model(N_in, N_out, NN):
    model = KPLS(theta0=[1e-2], poly='quadratic', n_start=10, print_prediction=False)
    # save_model(model, 'lime_screw_413.joblib')
    return model




def train_model(model, train_features, train_labels, verbose=True):
    model.set_training_values(train_features.to_numpy(), train_labels.to_numpy())

    #   for i in range(2):
    #     sm.set_training_derivatives(xt, yt[:, 1 + i].reshape((yt.shape[0], 1)), i)

    model.train()

    return model


def test_model(model, test_features, output_cols):
    model_output = model.predict_values(test_features.to_numpy())
    model_output = pd.DataFrame(data=model_output, columns=output_cols)
    return model_output


def build_model(N_in, N_out):
    sm = GEKPLS(theta0=[1e-2] * N_out, xlimits=fun.xlimits, extra_points=1, print_prediction=False, n_comp=n_comp, )
    return model




# def train_model(model, train_features, train_labels, verbose = True):
#   N_out = train_labels.to_numpy().shape[1]
#   for i in range(N_out):
#     model[i].fit(train_features.to_numpy(),train_labels.to_numpy()[:,i])
#   return model

# def test_model(model, test_features, output_cols):
#   N_out = len(output_cols)
#   model_output = np.array([model[i].predict(test_features.to_numpy()) for i in range(N_out)]).T
#   model_output = pd.DataFrame(data = model_output, columns = output_cols)
#   return model_output

# def build_model():
#   model = [svm.SVR(),svm.SVR(),svm.SVR()]
#   return model


def train_model(model, train_features, train_labels, verbose=True):
    model.fit(train_features.to_numpy(), train_labels.to_numpy())
    return model


def test_model(model, test_features, output_cols):
    model_output = model.predict(test_features.to_numpy())
    model_output = pd.DataFrame(data=model_output, columns=output_cols)
    return model_output


def build_model(N_in, N_out, NN):
    model = MultiOutputRegressor(svm.SVR(kernel='rbf', degree=3), n_jobs=N_out)
    # save_model(model, 'lime_screw_472.joblib')
    return model




def train_model(model, train_features, train_labels, verbose=True):
    model.fit(train_features.to_numpy(), train_labels.to_numpy())
    return model


def test_model(model, test_features, output_cols):
    model_output = model.predict(test_features.to_numpy())
    model_output = pd.DataFrame(data=model_output, columns=output_cols)
    return model_output


def build_model(N_in, N_out):
    model = MultiOutputRegressor(GaussianNB(), n_jobs=N_out)
    # save_model(model, 'lime_screw_490.joblib')
    return model




def train_model(model, train_features, train_labels, verbose=True):
    model.fit(train_features.to_numpy(), train_labels.to_numpy())
    return model


def test_model(model, test_features, output_cols):
    model_output = model.predict(test_features.to_numpy())
    model_output = pd.DataFrame(data=model_output, columns=output_cols)
    return model_output


def build_model(N_in, N_out):
    kernel = 1.0 * kernels.RBF(1.0)
    model = MultiOutputRegressor(GaussianProcessRegressor(kernel), n_jobs=N_out)
    # save_model(model, 'lime_screw_508.joblib')
    return model





def train_model(model, train_features, train_labels, verbose=True):
    model.fit(train_features.to_numpy(), train_labels.to_numpy())
    return model


def test_model(model, test_features, output_cols):
    model_output = model.predict(test_features.to_numpy())
    model_output = pd.DataFrame(data=model_output, columns=output_cols)
    return model_output


def build_model(N_in, N_out):
    model = MultiOutputRegressor(tree.DecisionTreeRegressor(), n_jobs=N_out)
    # save_model(model, 'lime_screw_528.joblib')
    return model





def train_model(model, train_features, train_labels, verbose=True):
    model.fit(train_features.to_numpy(), train_labels.to_numpy())
    return model


def test_model(model, test_features, output_cols):
    model_output = model.predict(test_features.to_numpy())
    model_output = pd.DataFrame(data=model_output, columns=output_cols)
    return model_output


def build_model(N_in, N_out):
    model = MultiOutputRegressor(RandomForestRegressor(1000), n_jobs=N_out)
    save_model(model, 'lime_screw_547.joblib')
    return model





def train_model(model, train_features, train_labels, verbose=True):
    model.fit(train_features.to_numpy(), train_labels.to_numpy())
    return model


def test_model(model, test_features, output_cols):
    model_output = model.predict(test_features.to_numpy())
    model_output = pd.DataFrame(data=model_output, columns=output_cols)
    return model_output


def build_model(N_in, N_out, NN):
    model = MultiOutputRegressor(AdaBoostRegressor(n_estimators=10, estimator=
    tree.DecisionTreeRegressor(max_depth=NN, criterion='squared_error'),
                                                   learning_rate=0.3), n_jobs=N_out)

    save_model(model, 'lime_screw_571.joblib')
    return model

def save_model(model, file_path):
    # Save the trained model to the specified file path
    joblib.dump(model, file_path)


NN = 10
(test_model_all, test_data_all, rarity_all, model) = test_model_performance(inputs, outputs, build_model, train_model,
                                                                            test_model, NN)



def evaluate_performance(test_model_all, test_data_all, rarity_all, model, print_list, verbose=False):
    MSE = [mean_squared_error(test_data_all[j + '_mean'], test_model_all[j + '_mean']) for j in print_list]
    R2 = [r2_score(test_data_all[j + '_mean'], test_model_all[j + '_mean']) for j in print_list]
    MAE = [mean_absolute_error(test_data_all[j + '_mean'], test_model_all[j + '_mean']) for j in print_list]
    MAPE = [mean_absolute_percentage_error(test_data_all[j + '_mean'], test_model_all[j + '_mean']) for j in print_list]

    if verbose:

        print('variables', print_list)
        print('MSE:', MSE)
        print('R2:', R2)
        print('MAE:', MAE)
        print('MAPE:', MAPE)

        outputs = print_list

        for j in outputs:
            j = j + '_mean'
            plt.figure()
            plt.title(j)
            plt.plot(test_data_all[j], test_model_all[j], 'x')
            minn = np.min([np.min(test_data_all[j]), np.min(test_model_all[j])])
            maxx = np.max([np.max(test_data_all[j]), np.max(test_model_all[j])])
            plt.plot([minn, maxx], [minn, maxx])
            ax = plt.gca()
            ax.set_aspect('equal')
            plt.ylabel('Data')
            plt.xlabel('Model')
            print("Should print evaluate performance here")
            plt.show()

        for j in outputs:
            j = j + '_mean'
            plt.figure()
            plt.title(j)
            plt.plot(test_data_all[j])
            plt.plot(test_model_all[j])
            plt.ylabel('Data')
            plt.xlabel('Model')
            plt.xlim([0, 100])
            print("Should print evaluate performance 2 here")
            plt.show()

        # print(test_data_all+)
        # plt.figure()
        # plt.title("Predict vs Actual CE")
        # plt.plot(test_data_all["time2"], test_data_all["5caust_CE"], label="Actual")
        # plt.plot(test_model_all["time2"], test_model_all["5caust_CE"], label="predicted")
        # plt.ylabel('CE')
        # plt.xlabel('Date')
        # plt.show(block=False)

        test_data_all.to_csv("test_data_all.csv")
        test_model_all.to_csv("test_model_all.csv")


    # return {'MSE':MSE,'R2':R2,'MAE':MAE,'MAPE':MAPE}
    return [np.average(MSE), np.average(R2), np.average(MAE), np.average(MAPE)]



evaluate_performance(test_model_all, test_data_all, rarity_all, model, print_list, True)



def evaluate_performance(test_model_all, test_data_all, rarity_all, model, print_list, verbose=False):
    MSE = [mean_squared_error(test_data_all[j + '_mean'], test_model_all[j + '_mean']) for j in print_list]
    R2 = [r2_score(test_data_all[j + '_mean'], test_model_all[j + '_mean']) for j in print_list]
    MAE = [mean_absolute_error(test_data_all[j + '_mean'], test_model_all[j + '_mean']) for j in print_list]
    MAPE = [mean_absolute_percentage_error(test_data_all[j + '_mean'], test_model_all[j + '_mean']) for j in print_list]

    if verbose:

        print('variables', print_list)
        print('MSE:', MSE)
        print('R2:', R2)
        print('MAE:', MAE)
        print('MAPE:', MAPE)

        outputs = print_list

        # Call the function to write to the file
        write_metrics_to_file(print_list, MSE, R2, MAE, MAPE)

        for j in outputs:
            j = j + '_mean'
            plt.figure()
            plt.title(j)
            plt.plot(test_data_all[j], test_model_all[j], 'x')
            minn = np.min([np.min(test_data_all[j]), np.min(test_model_all[j])])
            maxx = np.max([np.max(test_data_all[j]), np.max(test_model_all[j])])
            plt.plot([minn, maxx], [minn, maxx])
            ax = plt.gca()
            ax.set_aspect('equal')
            plt.ylabel('Data')
            plt.xlabel('Model')
            print("Should print evaluate performance 3 here")
            plt.show()

        for j in outputs:
            j = j + '_mean'
            plt.figure()
            plt.title(j)
            plt.plot(test_data_all[j])
            plt.plot(test_model_all[j])
            plt.ylabel('Data')
            plt.xlabel('Model')
            plt.xlim([0, 100])
            print("Should print evaluate performance 4 here")
            plt.show()



    # return {'MSE':MSE,'R2':R2,'MAE':MAE,'MAPE':MAPE}
    return [np.average(MSE), np.average(R2), np.average(MAE), np.average(MAPE)]


def hyperparameter_optimizer(start, end):
    opt_list = np.linspace(start, end, (end - start) + 1).astype('int')
    df = pd.DataFrame(columns=['MSE', 'R2', 'MAE', 'MAPE'], index=opt_list, data=np.zeros((len(opt_list), 4)))

    for NN in opt_list:
        (test_model_all, test_data_all, rarity_all, model) = test_model_performance(inputs, outputs, build_model,
                                                                                    train_model, test_model, int(NN))
        stats_dict = evaluate_performance(test_model_all, test_data_all, rarity_all, model, print_list)
        try:
            df.loc[int(NN)] = stats_dict
        except:
            return df

    return df

def write_metrics_to_file(print_list, MSE, R2, MAE, MAPE):
    # Get the project's directory (the directory containing this script)
    project_directory = os.path.dirname(os.path.abspath(__file__))

    # Define the directory where you want to save the file
    results_directory = os.path.join(project_directory, 'results')

    # Create the results directory if it doesn't exist
    os.makedirs(results_directory, exist_ok=True)

    # Create the filename with the path to the results directory
    filename = os.path.join(results_directory, f'model_summary_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.txt')

    # Open the file for writing
    with open(filename, 'w') as file:
        file.write('variables: ' + ', '.join(print_list) + '\n')
        file.write(f'MSE: {MSE}\n')
        file.write(f'R2: {R2}\n')
        file.write(f'MAE: {MAE}\n')
        file.write(f'MAPE: {MAPE}\n')

    print(f'Data has been saved to {filename}')



df = hyperparameter_optimizer(20, 40)
plt.plot(df['MAE'])
plt.show()
print("Should print MAE")
NN = 12
(test_model_all, test_data_all, rarity_all, model) = test_model_performance(inputs, outputs, build_model, train_model,
                                                                            test_model, NN)

test_model_all.to_csv('debug.csv')

# save_model(model, "lime_screw_729.pkl")
# pickle.dump(model, open("lime_screw_729.pkl", "wb"))
joblib.dump(model, "lime_model_shifted.pkl")

stats_dict = evaluate_performance(test_model_all, test_data_all, rarity_all, model, print_list, verbose=True)

#load model
loaded_model = joblib.load("lime_screw_729.pkl")
# slaker_data = pd.read_csv("inputs.csv", index_col = [0])
# slaker_data.index = pd.to_datetime(slaker_data.index)
# print(slaker_data)
# Make predictions

# slaker_data['lime_dosing_factor_reloaded'] = loaded_model.predict(slaker_data)

caustic.to_csv("final.csv")





