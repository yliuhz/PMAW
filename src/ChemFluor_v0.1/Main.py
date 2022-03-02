import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
import csv
# from sklearn.externals import joblib
import joblib
import lightgbm as lgb
from scipy import stats
import warnings
import os
import time

#get all *.csv files in given path
def get_all_csv_name(path):
    filename_list = []
    for folderName, subfolders, filenames in os.walk(path):
        for file_name in filenames:
            if '.csv' in file_name:
                filename_list.append(file_name)
    return filename_list

#read all test file, returning molecular name array y and fingerprint array X
def read_test_csv(filename_list):
    X_temp = []
    for filename in filename_list:
        with open('./put_your_predict_file_here/' + filename, 'r') as f:
            f_csv = csv.reader(f)
            for row in f_csv:
                X_temp.append(row)
    y = [example[0] for example in X_temp]
    X_np_temp = np.array(X_temp)
    X = X_np_temp[:, 1:]
    X.astype('float64')
    return X, y

#read all train file, returning label array y and fingerprint array X
def read_train_csv_EM(filename_list):
    X_temp = []
    #read every newly added training dataset
    for filename in filename_list:
        with open('./put_your_train_file_here/' + filename, 'r') as f:
            f_csv = csv.reader(f)
            for row in f_csv:
                X_temp.append(row)
    #read default training dataset
    with open('./model/Emission_Database.csv', 'r') as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            X_temp.append(row)
    for i in range(0, len(X_temp)):
        X_temp[i] = [float(j) for j in X_temp[i]]
    X_np_temp = np.array(X_temp)
    y = X_np_temp[:, 0]
    X = X_np_temp[:, 1:]
    return X, y

#read all train file, returning label array y and fingerprint array X
def read_train_csv_ABS(filename_list):
    X_temp = []
    #read every newly added training dataset
    for filename in filename_list:
        with open('./put_your_train_file_here/' + filename, 'r') as f:
            f_csv = csv.reader(f)
            for row in f_csv:
                X_temp.append(row)
    # #read default training dataset
    # with open('./model/Absorption_Database.csv', 'r') as f:
    #     f_csv = csv.reader(f)
    #     for row in f_csv:
    #         X_temp.append(row)
    for i in range(0, len(X_temp)):
        X_temp[i] = [float(j) for j in X_temp[i]]
    X_np_temp = np.array(X_temp)
    y = X_np_temp[:, 0]
    X = X_np_temp[:, 1:]
    return X, y

#read all train file, returning label array y and fingerprint array X
def read_train_csv_QY(filename_list, thereshold=0.25):
    X_temp = []
    #read every newly added training dataset
    for filename in filename_list:
        with open('./put_your_train_file_here/' + filename, 'r') as f:
            f_csv = csv.reader(f)
            for row in f_csv:
                X_temp.append(row)
    #read default training dataset
    with open('./model/PLQY_Classification_Database.csv', 'r') as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            X_temp.append(row)

    for i in range(0, len(X_temp)):
        X_temp[i] = [float(j) for j in X_temp[i]]
    X_np_temp = np.array(X_temp)
    y = X_np_temp[:, 0]
    X = X_np_temp[:, 1:]
    #classify label based on QY value
    for u in range(0, np.size(y)):
        if y[u] < thereshold:
            y[u] = 0
        else:
            y[u] = 1
    return X, y

def read_train_csv_QY_reg_no_oversampling(filename_list):
    X_temp = []
    #read every newly added training dataset
    for filename in filename_list:
        with open('./put_your_train_file_here/' + filename, 'r') as f:
            f_csv = csv.reader(f)
            for row in f_csv:
                X_temp.append(row)
    #read default training dataset
    with open('./model/PLQY_Regression_Database_high.csv', 'r') as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            X_temp.append(row)
    with open('./model/PLQY_Regression_Database_low.csv', 'r') as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            X_temp.append(row)

    X_np_temp = np.array(X_temp)
    y = X_np_temp[:, 0]
    X = X_np_temp[:, 1:]
    y = y.astype(np.float)
    X = X.astype(np.float)
    return X, y

def read_train_csv_QY_reg_with_oversampling(filename_list):
    X_temp = []
    #read every newly added training dataset
    for filename in filename_list:
        with open('./put_your_train_file_here/' + filename, 'r') as f:
            f_csv = csv.reader(f)
            for row in f_csv:
                X_temp.append(row)
    #read default training dataset
    with open('./model/PLQY_Regression_Database_high.csv', 'r') as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            X_temp.append(row)

    whole_high = np.array(X_temp)

    X_temp = []
    with open('./model/PLQY_Regression_Database_low.csv', 'r') as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            X_temp.append(row)

    whole_low = np.array(X_temp)
    whole = np.concatenate((whole_low,whole_high))
    whole = np.concatenate((whole,whole_high))
    whole = np.concatenate((whole,whole_high))
    y = whole[:, 0]
    X = whole[:, 1:]
    y = y.astype(np.float)
    X = X.astype(np.float)
    return X, y

#print result
def output_result(mol_result, mol_name, mol_type):
    if mol_type == 'QY Classification':
        #QY result is the probability of being one, so thereshold is set to 0.5
        mol_result_new=[]
        for i in range(0, len(mol_name)):
            if mol_result[i] < 0.5:
                mol_result_new.append("0 PLQY<0.25") 
            else:
                mol_result_new.append("1 PLQY>0.25") 
        mol_result = mol_result_new

    long_string = str(mol_type) + ' results:\n'
    for i in range(0, len(mol_name)):
        long_string = long_string + str(
            mol_name[i]) + ' ' + str(mol_type) + ': ' + str(
                mol_result[i]) + '\n'
    print(long_string)
    with open(
            str(mol_type) + 'results' + time.strftime("%H%M%S") + '.txt',
            'w') as file_handle:
        file_handle.write(long_string)
        file_handle.close()


warnings.filterwarnings("ignore")

while True:

    #search for every *.csv file in train folder and predict folder
    train_files = get_all_csv_name('./put_your_train_file_here/')
    test_files = get_all_csv_name('./put_your_predict_file_here/')
    pre_type = input(
        'please input the job type you want to do.\n1: predict EM;\n2: predict ABS;\n3: classify QY;\n4: regress QY;\n0: quit\n '
    )
    #if enter 0 then quit
    pre_type = int(pre_type)
    if pre_type == 0:
        break

    train_type = input('New models? 0: No; 1:Yes\n ')
    train_type = int(train_type)

    if train_type == 0:
        #load model
        if pre_type == 1:
            clf = joblib.load('./model/Emsision_Model_for_Predict.m')
        elif pre_type == 2:
            clf = joblib.load('./model/Absorption_Model_for_Predict.m')
        elif pre_type == 3:
            clf = joblib.load('./model/PLQY_Model_for_Classification.m')
        elif pre_type == 4:
            oversampling_type = input('Use oversampled model? 0: No; 1:Yes\n ')
            oversampling_type = int(oversampling_type)
            if oversampling_type == 0:
                clf = joblib.load('./model/PLQY_Model_for_Regression_no_Oversample.pkl')
            else:
                clf = joblib.load('./model/PLQY_Model_for_Regression_with_Oversample.pkl')

        #read fingerprint and predict result
        X_pre, X_name = read_test_csv(test_files)
        y_pre = clf.predict(X_pre)
        #show result
        if pre_type == 1:
            output_result(y_pre, X_name, 'EM')
        elif pre_type == 2:
            output_result(y_pre, X_name, 'ABS')
        elif pre_type == 3:
            output_result(y_pre, X_name, 'QY Classification')
        elif pre_type == 4:
            output_result(y_pre, X_name, 'QY Regression')

    elif train_type == 1:
        #train model with new database and predict
        if pre_type == 1:
            clf = GradientBoostingRegressor(learning_rate=0.05,
                                            max_depth=31,
                                            max_features=300,
                                            min_samples_leaf=20,
                                            n_estimators=1000)
            #read fingerprint and train model
            X_train, y_train = read_train_csv_EM(train_files)
            print('training')
            clf.fit(X_train, y_train)
            X_pre, X_name = read_test_csv(test_files)
            #predict and show result
            y_pre = clf.predict(X_pre)
            output_result(y_pre, X_name, 'EM')
        elif pre_type == 2:
            clf = GradientBoostingRegressor(learning_rate=0.05,
                                            max_depth=31,
                                            max_features=300,
                                            min_samples_leaf=20,
                                            n_estimators=1000)
            #read fingerprint and train model
            X_train, y_train = read_train_csv_ABS(train_files)
            print('training')
            clf.fit(X_train, y_train)
            X_pre, X_name = read_test_csv(test_files)
            #predict and show result
            y_pre = clf.predict(X_pre)
            output_result(y_pre, X_name, 'ABS')
        elif pre_type == 3:
            clf = lgb.LGBMRegressor(n_estimators=600,
                                    learning_rate=0.1,
                                    max_depth=70,
                                    num_leaves=45,
                                    objective='binary')
            #read fingerprint and train model, and setting thereshold
            X_train, y_train = read_train_csv_QY(train_files, thereshold = 0.25)
            print('training')
            clf.fit(X_train, y_train)
            X_pre, X_name = read_test_csv(test_files)
            #predict and show result
            y_pre = clf.predict(X_pre)
            output_result(y_pre, X_name, 'QY Classification')
        elif pre_type == 4:
            clf = lgb.LGBMRegressor(learning_rate=0.1, 
                                    max_depth=20, 
                                    num_leaves=20,
                                    n_estimators=1000)
            oversampling_type = input('Oversampling? 0: No; 1:Yes\n ')
            oversampling_type = int(oversampling_type)
            if oversampling_type == 0:
                X_train, y_train = read_train_csv_QY_reg_no_oversampling(train_files)
            else:
                X_train, y_train = read_train_csv_QY_reg_with_oversampling(train_files)
            print('training')
            clf.fit(X_train, y_train)
            X_pre, X_name = read_test_csv(test_files)
            #predict and show result
            y_pre = clf.predict(X_pre)
            output_result(y_pre, X_name, 'QY Regression')
            
            
            