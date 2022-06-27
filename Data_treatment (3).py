import pandas as pd
import numpy as np
from tqdm import tqdm

train_set = pd.read_csv('/Users/lucasmichaud/Desktop/MLTask2/train_features.csv')
test_set = pd.read_csv('/Users/lucasmichaud/Desktop/MLTask2/test_features.csv')


def data_normalization_train(df_train_data):
    df_train_data = df_train_data.drop(["pid", "Age", "Time"], axis=1)
    mean_train_data = df_train_data.mean(axis=0)  # Attention ce code est faux
    std_train_data = df_train_data.std(axis=0)
    normalized_train_data = (df_train_data - mean_train_data) / std_train_data

    return normalized_train_data, mean_train_data, std_train_data


# def preprocessing_train_features(train_features):


def main_preprocessing_train(data_frame_train_features):
    normalized_train_data, mean_train_data, std_train_data = data_normalization_train(data_frame_train_features)
    DATA_for_NN = preprocess_features(normalized_train_data, data_frame_train_features["pid"],
                                      data_frame_train_features[
                                          "Age"])  # ATTENTION A LA FORME #tuple, extract while keeping pd
    DATA_for_NN.to_csv('/Users/lucasmichaud/Desktop/MLTask2/DATA_for_NN.csv')
    return 0


def main_preprocessing_test(data_frame_test_features, train_set1):
    mean_train_data = data_normalization_train(train_set1)[1]
    std_train_data = data_normalization_train(train_set1)[2]
    test_data = data_frame_test_features.drop(["pid", "Age", "Time"], axis=1)
    normalized_test_data = (test_data - mean_train_data) / std_train_data
    DATA_for_NN_test = preprocess_features(normalized_test_data, data_frame_test_features["pid"],
                                           data_frame_test_features["Age"])
    DATA_for_NN_test.to_csv('/Users/lucasmichaud/Desktop/MLTask2/DATA_for_NN_test.csv')
    return 0


def preprocess_features(normalized_data, pid_column, age_column):
    normalized_data.insert(0, "pid", pid_column)
    normalized_data.insert(0, "Age", age_column)

    DATA = []

    for patient_id, normalized_features in tqdm(normalized_data.groupby("pid")):
        features = normalized_features
        age = features['Age'].mean()  # get the value to inout in STD
        pid = features['pid'].mean()  # same

        # number_of_nans_patient = features.isna().sum()
        features.loc[:, features.isna().all()] = 0
        patient_mean = features.mean(axis=0)
        patient_min = features.min(axis=0)
        patient_max = features.max(axis=0)
        patient_std = features.interpolate(method='linear', axis=0).diff().mean()

        # number_of_nans_patient = features.isna().sum()
        # tot_nans = number_of_nans_patient.sum()
        # nans.append(tot_nans)
        # patient_features = extract_features_each_patients(features)

        patient_features = pd.concat([patient_mean, patient_min, patient_max, patient_std], axis=1)
        patient_features = patient_features.transpose()
        patient_features.rename(index={0: 'mean', 1: 'min', 2: 'max', 3: 'STD'}, inplace=True)
        patient_features.at['STD', 'pid'] = pid
        patient_features.at['STD', 'Age'] = age
        DATA.append(patient_features)

    df_DATA = pd.concat(DATA, axis=0)
    return df_DATA  # what type do we find


#main = main_preprocessing_train(train_set)
test = main_preprocessing_test(test_set, train_set)
