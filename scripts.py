import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error, r2_score
from scipy.interpolate import interp1d
from scipy.spatial import distance
import tensorflow as tf
from tensorflow import keras
from keras import layers
import joblib
import numpy as np
import csv


def scale_data(df, *scaler_data):
    '''
    Designed to scale and unscale data using a range scaling method
    The optional argument is for unscaling as the original values are required.
    '''
    df.to_csv("input_2_scale.csv")
    if scaler_data:  # unscale
        scaler_save_divide = scaler_data[0][0]
        scaler_save_subtract = scaler_data[0][1]
        for var in df.columns:
            if '_mean' in var:
                df.loc[:, var] = df.loc[:, var] * scaler_save_divide[var] + scaler_save_subtract[var]
            elif '_std' in var:
                df.loc[:, var] = df.loc[:, var] * scaler_save_divide[var[:-4] + '_mean']

        # pd.DataFrame([scaler_data]).to_csv("scaler_data.csv")

        # Specify the file path where you want to save the data
        file_path = 'scaler_data.txt'

        # Open the file in write mode
        with open(file_path, 'w') as file:
            for dictionary in scaler_data:
                # Write each dictionary as a string in the file
                file.write(str(dictionary) + '\n')
        return df

    else:  # scale
        mean_list = [c for c in df.columns if 'mean' in c]
        scaler_save_subtract = dict(zip(mean_list, df[mean_list].min().to_numpy()))
        scaler_save_divide = dict(zip(mean_list, (df[mean_list].max() - df[mean_list].min()).to_numpy()))

        # if np.isnan(scaler_save_subtract) or np.isnan(scaler_save_subtract):
        #     print(scaler_save_subtract)

        for var in df.columns:
            if '_mean' in var:
                df.loc[:, var] = (df.loc[:, var] - scaler_save_subtract[var]) / scaler_save_divide[var]
            elif '_std' in var:
                df.loc[:, var] = (df.loc[:, var]) / scaler_save_divide[var[:-4] + '_mean']
        return df, (scaler_save_divide, scaler_save_subtract)


def splitter(df, thresh, inputs):
    df_train = df.copy()
    df_test = df.copy()

    for col in inputs:
        rangee = df[col].max() - df[col].min()

        df_train = df_train[(df_train[col] < (df[col].max() - (1 - thresh) * rangee))]
        df_train = df_train[(df_train[col] > (df[col].min() + (1 - thresh) * rangee))]

    df_inx = pd.DataFrame(np.array(range(df.shape[0])))
    test_inx = df_inx.index.difference(df_train.index)
    df_test = df.iloc[test_inx]

    return df_train, df_test


def split_dataset(df, split=0.8, method='random', inputs=[], rarity=[]):
    if method == 'random':
        df_train = df.sample(frac=0.8, random_state=0)
        df_test = df.drop(df_train.index)

    elif method == 'spherical':
        N_rare = int((1 - split) * len(df))
        N_core = int((split) * len(df))
        rare_inx = np.argpartition(rarity, -N_rare)[-N_rare:]
        core_inx = np.argpartition(rarity, N_core)[:N_core]

        df_train = df.iloc[core_inx]
        df_test = df.iloc[rare_inx]

    elif method == 'geographic':
        thresh = split ** (1 / len(inputs))

        def split_wrap(thresh):
            print('thresh', thresh[0])
            df_train, df_test = splitter(df, thresh[0], inputs)
            test_split = len(df_train) / len(df)

            diff = (split - test_split) ** 2
            print('diff', diff)
            return np.array(diff)

        out = minimize(split_wrap, thresh, method='Nelder-Mead')
        thresh = out.x[0]
        df_train, df_test = splitter(df, thresh, inputs)
    return df_train, df_test


def combiner(A, B):
    listt = np.concatenate([A.to_numpy(), B.to_numpy()], axis=1)
    cols = list(A.columns) + list(B.columns)
    C = pd.DataFrame(columns=cols, data=listt)
    return C


def compute_rarity(inputs, method='kde'):
    if method == 'kde':
        kernel = gaussian_kde(inputs.T, bw_method='scott')
        kde = kernel.evaluate(inputs.T)
        rarity_arr = 1 - ((kde - kde.min()) / (kde.max() - kde.min()))

    elif method == 'euclidean':
        AV = np.average(inputs, axis=0)
        eu = np.array([distance.euclidean(inputs[i, :], AV) for i in range(len(inputs))])
        rarity_arr = eu
        # rarity_arr = ((eu - np.min(eu))/(np.max(eu) - np.min(eu)))
    return rarity_arr


def split_io(df, input_list, output_list):
    features = df[input_list]
    labels = df[output_list]

    return features, labels


def calculate_aic(model, features, data):
    output = model.predict(features)
    num_params = model.count_params()
    mse = mean_squared_error(data, output)
    N = len(data)
    aic = N * np.log(mse) + 2 * num_params
    bic = N * np.log(mse) + num_params * np.log(N)
    return aic, bic


def test_surrogate_performance(input_data, output_data, build_model, train_model, test_model):
    input_list = [col for col in input_data.columns if '_mean' in col]
    output_list = [col for col in output_data.columns if '_mean' in col]

    # Scaling

    input_scaled, input_scaler_data = scale_data(input_data.copy())
    output_scaled, output_scaler_data = scale_data(output_data.copy())

    datax = np.concatenate([input_scaled.to_numpy(), output_scaled.to_numpy()], axis=1)
    cols = list(input_scaled.columns) + list(output_data.columns)
    input_dataset = pd.DataFrame(columns=cols, data=datax)

    input_dataset = input_dataset.dropna()

    rarity = compute_rarity(input_dataset[input_list].to_numpy(), method='euclidean')
    input_dataset['rarity'] = rarity

    input_dataset = input_dataset.copy().sort_values('rarity')
    input_dataset_r = input_dataset.copy().sort_values('rarity', ascending=False)

    train_dataset_all, test_dataset_all = split_dataset(input_dataset, split=0.8, method='random', inputs=input_list)

    train_dataset_core1, test_dataset_fringe1 = split_dataset(input_dataset, split=0.8, method='spherical',
                                                              inputs=input_list,
                                                              rarity=input_dataset['rarity'].to_numpy())
    test_dataset_core1 = train_dataset_core1.copy()

    train_dataset_fringe2, test_dataset_core2 = split_dataset(input_dataset_r, split=0.8, method='spherical',
                                                              inputs=input_list,
                                                              rarity=input_dataset['rarity'].to_numpy())

    train_dataset_core1 = train_dataset_core1.copy().sort_values('rarity', ascending=False)

    train_features_all, train_labels_all = split_io(train_dataset_all, input_list, output_list)
    test_features_all, test_labels_all = split_io(test_dataset_all, input_list, output_list)

    train_features_core1, train_labels_core1 = split_io(train_dataset_core1, input_list, output_list)
    test_features_core1, test_labels_core1 = split_io(test_dataset_core1, input_list, output_list)
    test_features_fringe1, test_labels_fringe1 = split_io(test_dataset_fringe1, input_list, output_list)

    train_features_fringe2, train_labels_fringe2 = split_io(train_dataset_fringe2, input_list, output_list)
    test_features_core2, test_labels_core2 = split_io(test_dataset_core2, input_list, output_list)

    rarity_all = test_dataset_all['rarity'].to_numpy()
    rarity_fringe1 = test_dataset_fringe1['rarity'].to_numpy()
    rarity_core1 = test_dataset_core1['rarity'].to_numpy()
    rarity_fringe2 = train_dataset_fringe2['rarity'].to_numpy()
    rarity_core2 = test_dataset_core2['rarity'].to_numpy()

    # 1.0 Train on all ---------------------------------------------------------------------------------------------

    model = build_model()
    model = train_model(model, train_features_all, train_labels_all)

    # model_aic = calculate_aic(model,train_features_all, train_labels_all)
    # print('model AIC', model_aic)

    # 1.1 test on all
    model_output_all = test_model(model, test_features_all, output_list)
    # 1.2 test on core
    model_output_all_core = test_model(model, test_features_core1, output_list)
    # 1.3 test on fringe
    model_output_all_fringe = test_model(model, test_features_fringe1, output_list)
    # 1.4 test on fringe
    model_output_all_core2 = test_model(model, test_features_core2, output_list)

    test_outputs_unscaled_all = scale_data(model_output_all.copy(), output_scaler_data)
    test_outputs_unscaled_all_core = scale_data(model_output_all_core.copy(), output_scaler_data)
    test_outputs_unscaled_all_fringe = scale_data(model_output_all_fringe.copy(), output_scaler_data)
    test_outputs_unscaled_all_core2 = scale_data(model_output_all_core2.copy(), output_scaler_data)

    test_data_unscaled_all = scale_data(test_labels_all.copy(), output_scaler_data)
    test_data_unscaled_all_core = scale_data(test_labels_core1.copy(), output_scaler_data)
    test_data_unscaled_all_fringe = scale_data(test_labels_fringe1.copy(), output_scaler_data)
    test_data_unscaled_all_core2 = scale_data(test_labels_core2.copy(), output_scaler_data)

    test_inputs_unscaled_all = scale_data(test_features_all.copy(), input_scaler_data)
    test_inputs_unscaled_all_core = scale_data(test_features_core1.copy(), input_scaler_data)
    test_inputs_unscaled_all_fringe = scale_data(test_features_fringe1.copy(), input_scaler_data)
    test_inputs_unscaled_all_core2 = scale_data(test_features_core2.copy(), input_scaler_data)

    # 2.0 Train on core (80%) ---------------------------------------------------------------------------------------------

    model1 = build_model()
    model1 = train_model(model1, train_features_core1, train_labels_core1)

    # model1_aic = calculate_aic(model1,train_features_core1, train_labels_core1)
    # print('model1 AIC', model1_aic)

    # 2.1 test on all
    model_output_core1_all = test_model(model1, test_features_all, output_list)
    # # 2.2 test on core
    # model_output_core1_core = test_model(model, test_features_core1, output_list)
    # 2.3 test on fringe
    model_output_core1_fringe = test_model(model1, test_features_fringe1, output_list)

    test_outputs_unscaled_core1_all = scale_data(model_output_core1_all.copy(), output_scaler_data)
    test_outputs_unscaled_core1_fringe = scale_data(model_output_core1_fringe.copy(), output_scaler_data)

    test_data_unscaled_core1_all = scale_data(test_labels_all.copy(), output_scaler_data)
    test_data_unscaled_core1_fringe = scale_data(test_labels_fringe1.copy(), output_scaler_data)

    test_inputs_unscaled_core1_all = scale_data(test_features_all.copy(), input_scaler_data)
    test_inputs_unscaled_core1_fringe = scale_data(test_features_fringe1.copy(), input_scaler_data)

    # 3.0 Train on fringe (80%) ---------------------------------------------------------------------------------------------

    model2 = build_model()
    model2 = train_model(model2, train_features_fringe2, train_labels_fringe2)

    # model2_aic = calculate_aic(model2, train_features_fringe2, train_labels_fringe2)
    # print('model2 AIC', model2_aic)

    # 3.1 test on all
    model_output_fringe2_all = test_model(model2, test_features_all, output_list)
    # 3.2 test on core
    model_output_fringe2_core = test_model(model2, test_features_core2, output_list)

    test_outputs_unscaled_fringe2_all = scale_data(model_output_fringe2_all.copy(), output_scaler_data)
    test_outputs_unscaled_fringe2_core = scale_data(model_output_fringe2_core.copy(), output_scaler_data)

    test_data_unscaled_fringe2_all = scale_data(test_labels_all.copy(), output_scaler_data)
    test_data_unscaled_fringe2_core = scale_data(test_labels_core2.copy(), output_scaler_data)

    test_inputs_unscaled_fringe2_all = scale_data(test_features_all.copy(), input_scaler_data)
    test_inputs_unscaled_fringe2_core = scale_data(test_features_core2.copy(), input_scaler_data)

    # ---------------------------------------------------------------------------------------------

    test_model_all = combiner(test_inputs_unscaled_all, test_outputs_unscaled_all)
    test_model_all_core = combiner(test_inputs_unscaled_all_core, test_outputs_unscaled_all_core)
    test_model_all_fringe = combiner(test_inputs_unscaled_all_fringe, test_outputs_unscaled_all_fringe)
    test_model_all_core2 = combiner(test_inputs_unscaled_all_core2, test_outputs_unscaled_all_core2)

    test_model_core1_all = combiner(test_inputs_unscaled_core1_all, test_outputs_unscaled_core1_all)
    test_model_core1_fringe = combiner(test_inputs_unscaled_core1_fringe, test_outputs_unscaled_core1_fringe)

    test_model_fringe2_all = combiner(test_inputs_unscaled_fringe2_all, test_outputs_unscaled_fringe2_all)
    test_model_fringe2_core = combiner(test_inputs_unscaled_fringe2_core, test_outputs_unscaled_fringe2_core)

    test_data_all = combiner(test_inputs_unscaled_all, test_data_unscaled_all)
    test_data_all_core = combiner(test_inputs_unscaled_all_core, test_data_unscaled_all_core)
    test_data_all_fringe = combiner(test_inputs_unscaled_all_fringe, test_data_unscaled_all_fringe)
    test_data_all_core2 = combiner(test_inputs_unscaled_all_core2, test_data_unscaled_all_core2)

    test_data_core1_all = combiner(test_inputs_unscaled_core1_all, test_data_unscaled_core1_all)
    test_data_core1_fringe = combiner(test_inputs_unscaled_core1_fringe, test_data_unscaled_core1_fringe)

    test_data_fringe2_all = combiner(test_inputs_unscaled_fringe2_all, test_data_unscaled_fringe2_all)
    test_data_fringe2_core = combiner(test_inputs_unscaled_fringe2_core, test_data_unscaled_fringe2_core)

    return (test_model_all, test_model_all_core, test_model_all_fringe, test_model_all_core2, test_model_core1_all,
            test_model_core1_fringe, test_model_fringe2_all, test_model_fringe2_core, test_data_all,
            test_data_all_core, test_data_all_fringe, test_data_all_core2, test_data_core1_all, test_data_core1_fringe,
            test_data_fringe2_all, test_data_fringe2_core,
            rarity_all, rarity_fringe1, rarity_core1, rarity_fringe2, rarity_core2)


def test_model_performance(input_data, output_data, build_model, train_model, test_model, NN):
    input_list = [col for col in input_data.columns if '_mean' in col]
    output_list = [col for col in output_data.columns if '_mean' in col]

    N_in = len(input_list)
    N_out = len(output_list)


    # Scaling
    input_scaled, input_scaler_data = scale_data(input_data.copy())
    output_scaled, output_scaler_data = scale_data(output_data.copy())
    input_scaled.to_csv('input_scaled.csv')

    print(f'input Scaler Data: {input_scaler_data}')
    print(f'output_scaler_data: { output_scaler_data}')

    export_tuple_to_txt(input_scaler_data, "input_scaler_data.txt")
    export_tuple_to_txt(output_scaler_data, "output_scaler_data.txt")

    datax = np.concatenate([input_scaled.to_numpy(), output_scaled.to_numpy()], axis=1)
    cols = list(input_scaled.columns) + list(output_data.columns)
    input_dataset = pd.DataFrame(columns=cols, data=datax)

    input_dataset = input_dataset.dropna()

    rarity = compute_rarity(input_dataset[input_list].to_numpy(), method='euclidean')
    input_dataset['rarity'] = rarity

    input_dataset = input_dataset.copy().sort_values('rarity')
    input_dataset_r = input_dataset.copy().sort_values('rarity', ascending=False)

    train_dataset_all, test_dataset_all = split_dataset(input_dataset, split=0.8, method='random', inputs=input_list)

    train_features_all, train_labels_all = split_io(train_dataset_all, input_list, output_list)
    test_features_all, test_labels_all = split_io(test_dataset_all, input_list, output_list)

    rarity_all = test_dataset_all['rarity'].to_numpy()

    # 1.0 Train on all ---------------------------------------------------------------------------------------------

    model = build_model(N_in, N_out, NN)
    model = train_model(model, train_features_all, train_labels_all)

    # model_aic = calculate_aic(model,train_features_all, train_labels_all)
    # print('model AIC', model_aic)

    # 1.1 test on all
    model_output_all = test_model(model, test_features_all, output_list)
    test_features_all.to_csv(' test_features_all.csv')
    model_output_all.to_csv( 'model_output_all.csv')


    test_outputs_unscaled_all = scale_data(model_output_all.copy(), output_scaler_data)


    # pd.DataFrame([scaler_data]).to_csv("scaler_data.csv")

    test_data_unscaled_all = scale_data(test_labels_all.copy(), output_scaler_data)

    test_inputs_unscaled_all = scale_data(test_features_all.copy(), input_scaler_data)

    test_inputs_unscaled_all.to_csv( 'test_inputs_unscaled_all.csv')

    # ---------------------------------------------------------------------------------------------

    test_model_all = combiner(test_inputs_unscaled_all, test_outputs_unscaled_all)

    test_data_all = combiner(test_inputs_unscaled_all, test_data_unscaled_all)

    # joblib.dump(model, "lime_screw_scripts.pkl")

    return (test_model_all, test_data_all, rarity_all, model)



def evaluate_performance(test_model_all, test_model_all_core, test_model_all_fringe, test_model_all_core2,
                         test_model_core1_all,
                         test_model_core1_fringe, test_model_fringe2_all, test_model_fringe2_core, test_data_all,
                         test_data_all_core, test_data_all_fringe, test_data_all_core2, test_data_core1_all,
                         test_data_core1_fringe, test_data_fringe2_all, test_data_fringe2_core,
                         rarity_all, rarity_fringe1, rarity_core1, rarity_fringe2, rarity_core2):
    av_all = np.average([np.average(abs(test_data_all[j] - test_model_all[j])) / np.average(test_model_all[j]) for j in
                         test_model_all.columns])
    av_all_core = np.average(
        [np.average(abs(test_data_all_core[j] - test_model_all_core[j])) / np.average(test_model_all_core[j]) for j in
         test_model_all_core.columns])
    av_all_fringe = np.average(
        [np.average(abs(test_data_all_fringe[j] - test_model_all_fringe[j])) / np.average(test_model_all_fringe[j]) for
         j in test_model_all_fringe.columns])
    av_all_core2 = np.average(
        [np.average(abs(test_data_all_core2[j] - test_model_all_core2[j])) / np.average(test_model_all_core2[j]) for j
         in test_model_all_core2.columns])

    av_core1_all = np.average(
        [np.average(abs(test_data_core1_all[j] - test_model_core1_all[j])) / np.average(test_model_core1_all[j]) for j
         in test_model_core1_all.columns])
    av_core1_fringe = np.average([np.average(abs(test_data_core1_fringe[j] - test_model_core1_fringe[j])) / np.average(
        test_model_core1_fringe[j]) for j in test_model_core1_fringe.columns])

    av_fringe2_all = np.average(
        [np.average(abs(test_data_fringe2_all[j] - test_model_fringe2_all[j])) / np.average(test_model_fringe2_all[j])
         for j in test_model_fringe2_all.columns])
    av_fringe2_core = np.average([np.average(abs(test_data_fringe2_core[j] - test_model_fringe2_core[j])) / np.average(
        test_model_fringe2_core[j]) for j in test_model_fringe2_core.columns])

    r2_all = np.average([r2_score(test_data_all[j], test_model_all[j]) for j in test_model_all.columns])
    r2_all_core = np.average(
        [r2_score(test_data_all_core[j], test_model_all_core[j]) for j in test_model_all_core.columns])
    r2_all_fringe = np.average(
        [r2_score(test_data_all_fringe[j], test_model_all_fringe[j]) for j in test_model_all_fringe.columns])
    r2_all_core2 = np.average(
        [r2_score(test_data_all_core2[j], test_model_all_core2[j]) for j in test_model_all_core2.columns])

    r2_core1_all = np.average(
        [r2_score(test_data_core1_all[j], test_model_core1_all[j]) for j in test_model_core1_all.columns])
    r2_core1_fringe = np.average(
        [r2_score(test_data_core1_fringe[j], test_model_core1_fringe[j]) for j in test_model_core1_fringe.columns])

    r2_fringe2_all = np.average(
        [r2_score(test_data_fringe2_all[j], test_model_fringe2_all[j]) for j in test_model_fringe2_all.columns])
    r2_fringe2_core = np.average(
        [r2_score(test_data_fringe2_core[j], test_model_fringe2_core[j]) for j in test_model_fringe2_core.columns])

    print('r2 scores')
    print(r2_all, r2_all_core, r2_all_fringe, r2_all_core2, r2_core1_all, r2_core1_fringe, r2_fringe2_all,
          r2_fringe2_core)

    fringe_correction = (av_all_fringe / av_all_core)
    extrapolation_accuracy = av_core1_fringe / (fringe_correction)

    print('fringe_correction', fringe_correction)

    print('overall_accuracy', av_all)
    print('extrapolation_accuracy', extrapolation_accuracy)

    av_all = (abs(test_data_all - test_model_all) / (test_model_all.mean(axis=0)))  # .mean(axis=1)
    av_all_core = (abs(test_data_all_core - test_model_all_core) / (test_model_all_core.mean(axis=0)))  # .mean(axis=1)
    av_all_fringe = (abs(test_data_all_fringe - test_model_all_fringe) / (
        test_model_all_fringe.mean(axis=0)))  # .mean(axis=1)
    av_all_core2 = (
                abs(test_data_all_core2 - test_model_all_core2) / (test_model_all_core2.mean(axis=0)))  # .mean(axis=1)

    av_core1_all = (
                abs(test_data_core1_all - test_model_core1_all) / (test_model_core1_all.mean(axis=0)))  # .mean(axis=1)
    av_core1_fringe = (abs(test_data_core1_fringe - test_model_core1_fringe) / (
        test_model_core1_fringe.mean(axis=0)))  # .mean(axis=1)

    av_fringe2_all = (abs(test_data_fringe2_all - test_model_fringe2_all) / (
        test_model_fringe2_all.mean(axis=0)))  # .mean(axis=1)
    av_fringe2_core = (abs(test_data_fringe2_core - test_model_fringe2_core) / (
        test_model_fringe2_core.mean(axis=0)))  # .mean(axis=1)

    av_all = ((av_all.loc[:, (av_all != 0).any(axis=0)]).mean(axis=1)).to_frame()
    av_all_core = ((av_all_core.loc[:, (av_all_core != 0).any(axis=0)]).mean(axis=1)).to_frame()
    av_all_fringe = ((av_all_fringe.loc[:, (av_all_fringe != 0).any(axis=0)]).mean(axis=1)).to_frame()
    av_all_core2 = ((av_all_core2.loc[:, (av_all_core2 != 0).any(axis=0)]).mean(axis=1)).to_frame()

    av_core1_all = ((av_core1_all.loc[:, (av_core1_all != 0).any(axis=0)]).mean(axis=1)).to_frame()
    av_core1_fringe = ((av_core1_fringe.loc[:, (av_core1_fringe != 0).any(axis=0)]).mean(axis=1)).to_frame()

    av_fringe2_all = ((av_fringe2_all.loc[:, (av_fringe2_all != 0).any(axis=0)]).mean(axis=1)).to_frame()
    av_fringe2_core = ((av_fringe2_core.loc[:, (av_fringe2_core != 0).any(axis=0)]).mean(axis=1)).to_frame()

    av_all['rarity'] = rarity_all
    av_all_core['rarity'] = rarity_core1
    av_all_fringe['rarity'] = rarity_fringe1
    av_all_core2['rarity'] = rarity_core2

    av_core1_all['rarity'] = rarity_core1
    av_core1_fringe['rarity'] = rarity_fringe1

    av_fringe2_all['rarity'] = rarity_core2
    av_fringe2_core['rarity'] = rarity_fringe2

    # Smoothing

    def resampler(df):
        df = df.set_index('rarity').sort_index()
        df_resampled = df.rolling(200, center=True, min_periods=1).mean().ewm(alpha=0.05, adjust=True).mean()
        df_resampled = df_resampled.dropna()
        return df_resampled

    al = 0.02
    adj = True

    av_all = resampler(av_all)
    av_all_core = resampler(av_all_core)
    av_all_fringe = resampler(av_all_fringe)
    av_all_core2 = resampler(av_all_core2)

    av_core1_all = resampler(av_core1_all)
    av_core1_fringe = resampler(av_core1_fringe)

    av_fringe2_all = resampler(av_fringe2_all)
    av_fringe2_core = resampler(av_fringe2_core)

    plt.figure()
    plt.plot(av_core1_fringe / av_all_fringe, 'x')
    print('average extrapolation disadvantage', (np.average(av_core1_fringe / av_all_fringe)))

    # av_all_arr = av_all_arr.ewm(alpha=al, adjust =adj).mean()
    # av_core_arr = av_core_arr.ewm(alpha=al, adjust =adj).mean()
    # av_fringe_arr = av_fringe_arr.ewm(alpha=al, adjust =adj).mean()
    # av_core_fringe_arr = av_core_fringe_arr.ewm(alpha=al, adjust =adj).mean()

    plt.figure()
    plt.plot(av_all)
    plt.plot(av_all_core)
    plt.plot(av_all_fringe)
    plt.plot(av_all_core2)

    plt.plot(av_core1_all)
    plt.plot(av_core1_fringe)

    plt.plot(av_fringe2_all)
    plt.plot(av_fringe2_core)

    plt.figure()
    plt.plot(av_all_fringe, av_core1_fringe, 'x')

    minn = np.min([np.min(av_all_fringe[0].to_numpy()), np.min(av_core1_fringe[0].to_numpy())])
    maxx = np.max([np.max(av_all_fringe[0].to_numpy()), np.max(av_core1_fringe[0].to_numpy())])

    plt.plot([minn, maxx], [minn, maxx])
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.ylabel('extrapolate accuracy')
    plt.xlabel('interpolate accuracy')

    for j in ['WF_T_out_mean', 'air_T_out_mean', 'WF_subcool_mean']:
        print(j)
        plt.figure()
        # plt.plot(test_data['WF_flow'], test_data[j],'x')
        # plt.plot(test_data['WF_flow'], test_model[j],'x')
        plt.plot(test_data_all[j], test_model_all[j], 'x')

    for j in ['WF_T_out_mean', 'air_T_out_mean', 'WF_subcool_mean']:
        print(j)
        plt.figure()
        # plt.plot(test_data['WF_flow'], test_data[j],'x')
        # plt.plot(test_data['WF_flow'], test_model[j],'x')
        plt.plot(test_data_core1_fringe[j], test_model_core1_fringe[j], 'x')

    # plt.figure()

    # Its very unlikely that extrapolation accuracy will meet overall accuracy
    # but if theyre close we can infer if the underlying trend has been captured.

    return

def export_tuple_to_txt(data, txt_file):
    try:
        with open(txt_file, 'w') as file:
            for item in data:
                file.write(str(item) + '\n')
        print(f'Data has been exported to {txt_file}')
    except Exception as e:
        print(f'Error: {e}')

