
#################################
### YOU WON'T USE THIS FILE!! ###
#################################

import librosa
import librosa.display
import numpy as np
import os
import random
from scipy.fft import fft
from tqdm import tqdm 
from sklearn.utils import shuffle


''' ### Preparing model input for each part ###
# We tried to make sure data in different distances could be evenly devided to training and testing set, 
# so the code may be a little complicated.

# At first, we transfered the sound file to the frequency spectrum by using FFT.
# Next, we selected specific frequency band and combined the real and imaginary part of the frequency response 
# to an image as the model input.
# Thirdly, evenly devided data to training and testing set.
# Finally, labeled all data in the desire form. (depends on classification or regression)
'''
#%%
def data_preprocessing_1st(sec, shift, ratio, TYPE):
    # sec: the time length of the target sound file
    # shift: the time shift to retrieve next time window data
    # ratio : The ratio of training data in total data
    # TYPE : 
    #    0 -> classification (label = 0 or 1)
    #    1 -> regression (label = distance)
    # valid_ratio = (1 - ratio)*0.2
    num_eachfile_5s = int((5-sec)/shift + 1)
    num_eachfile_3s = int((3-sec)/shift + 1)

    # For sound with distance
    distances = ["0_0", "0_3", "0_5", "0_6", "1_0", "1_5", "2_0", "2_5", "3_0", "5_0", "6_0", "8_0"]
    number = [71, 15, 66, 22, 58, 41, 21, 32, 20, 5, 20, 15]
    label = [0, 0.3, 0.5, 0.6, 1, 1.5, 2, 2.5, 3, 5, 6, 8]
    training_datalist1 = []
    training_label = []
    testing_datalist1 = []
    testing_label = []

    ### Use your address
    addr = "./音檔/20220607_cut_data/shift=" + str(shift) + "/20200514/"
    ###
    
    for i in tqdm(range(len(distances))):
        addr_ducument = addr + distances[i] + "/"
        files = os.listdir(addr_ducument)

        a = [c for c in range(number[i])]
        ran = random.sample(a, number[i])
        limit = int(number[i]*ratio)
        for j in ran:
            for k in range(num_eachfile_5s):
                filename = files[j*num_eachfile_5s+k]
                
                filename = "./音檔/20220607_cut_data/shift=" + str(shift) + "/20200514/" + distances[i] + "/" + filename

                x, fs = librosa.load(filename, sr=None)
                X = fft(x)[1:3201] / len(x) * 2
                data1 = np.array([X.real, X.imag]).reshape((80,80,1))
                
                if limit > 0:
                    training_datalist1.append(data1)
                    training_label.append(label[i])
                    
                else:
                    testing_datalist1.append(data1)
                    testing_label.append(label[i])
            limit = limit-1
    
    ### Use your address
    addr = "./音檔/20220607_cut_data/shift=" + str(shift) + "/env/env/"
    ###
    
    # For env sound 
    files = os.listdir(addr)
    pre_5 = [c for c in range(169)]
    pre_5_ran = random.sample(pre_5, 169)
    limit = int(169*ratio)
    for j in tqdm(pre_5_ran):
        for k in range(num_eachfile_5s):
            filename = files[j*num_eachfile_5s+k]
            
            filename = "./音檔/20220607_cut_data/shift=" + str(shift) + "/env/env/" + filename

            x, fs = librosa.load(filename, sr=None)
            X = fft(x)[1:3201] / len(x) * 2
            data1 = np.array([X.real, X.imag]).reshape((80,80,1))

            
            if limit > 0:
                training_datalist1.append(data1)
                training_label.append(10)
            else:
                testing_datalist1.append(data1)
                testing_label.append(10)
        limit = limit-1
        
    post_3 = [c for c in range(83)]
    pre_3_ran = random.sample(post_3, 83)
    limit = int(83*ratio)
    for j in tqdm(pre_3_ran):
        for k in range(num_eachfile_3s):
            filename = files[j*num_eachfile_3s+k]

            filename = "./音檔/20220607_cut_data/shift=" + str(shift) + "/env/env/" + filename

            x, fs = librosa.load(filename, sr=None)
            X = fft(x)[1:3201] / len(x) * 2
            data1 = np.array([X.real, X.imag]).reshape((80,80,1))
            
            
            if limit > 0:
                training_datalist1.append(data1)
                training_label.append(10)
            else:
                testing_datalist1.append(data1)
                testing_label.append(10)
        limit = limit-1
        
    # container
    temp1 = training_datalist1[0]
    data_train1 = np.zeros((len(training_datalist1), temp1.shape[0], temp1.shape[1], 1))
    data_test1 = np.zeros((len(testing_datalist1), temp1.shape[0], temp1.shape[1], 1))
    
    label_train = np.zeros((len(training_label)))
    label_test = np.zeros((len(testing_label)))
    
    for i in range(len(training_datalist1)):
        data_train1[i,:] = training_datalist1[i]
        # data_train2[i,:] = training_datalist2[i]
        label_train[i] = training_label[i]
    for i in range(len(testing_datalist1)):
        data_test1[i,:] = testing_datalist1[i]
        # data_test2[i,:] = testing_datalist2[i]
        label_test[i] = testing_label[i]
    
    if TYPE == 0:
        for i in range(len(label_train)):
            if label_train[i] > 3 :
                label_train[i] = 0
            else:
                label_train[i] = 1
        for i in range(len(label_test)):
            if label_test[i] > 3 :
                label_test[i] = 0
            else:
                label_test[i] = 1
    
    data_train1, label_train, training_label = shuffle(data_train1, label_train, training_label)
    
    return data_train1, label_train, data_test1, label_test, training_label, testing_label
#%%
def preprocessing_2nd(x_train, train_prediction, training_label, x_test, testing_label, prediction):
    x_new_train = np.zeros(x_train.shape)
    new_training_label = np.zeros(len(training_label))
    x_new_test = np.zeros(x_test.shape)
    new_testing_label = np.zeros(len(testing_label))
    
    count_train = 0
    for i in range(len(x_train)):
        if train_prediction[i] == 1:
            x_new_train[count_train] = x_train[i]
            new_training_label[count_train] = training_label[i]
            count_train+=1
    x_new_train = x_new_train[:count_train]
    new_training_label = new_training_label[:count_train]
    
    
    y_new_train = np.zeros(len(new_training_label))
    for i in range(len(y_new_train)):
        if new_training_label[i] > 1.5 :
            y_new_train[i] = 0
        else:
            y_new_train[i] = 1
            
    
    count_test = 0
    for i in range(len(prediction)):
        if prediction[i] == 1:
            x_new_test[count_test] = x_test[i]
            new_testing_label[count_test] = testing_label[i]
            count_test+=1
    x_new_test = x_new_test[:count_test]
    new_testing_label = new_testing_label[:count_test]
    
    y_new_test = np.zeros(len(new_testing_label))
    for i in range(len(y_new_test)):
        if new_testing_label[i] > 1.5 :
            y_new_test[i] = 0
        else:
            y_new_test[i] = 1
    
    return x_new_train, y_new_train, x_new_test, y_new_test, new_training_label, new_testing_label
#%%x_new_train, y_new_train, x_new_test, y_new_test, new_training_label, new_testing_label
def preprocessing_3rd(x_new_train, train_prediction_2nd, new_training_label, x_new_test, new_testing_label, prediction_2nd):
    x_regression_train1_5 = np.zeros(x_new_train.shape)
    y_regression_train1_5 = np.zeros(len(new_training_label))
    x_regression_train3 = np.zeros(x_new_train.shape)
    y_regression_train3 = np.zeros(len(new_training_label))
    
    count_train1_5 = 0
    count_train3 = 0
    for i in range(len(x_new_train)):
        if train_prediction_2nd[i] == 1:
            x_regression_train1_5[count_train1_5] = x_new_train[i]
            y_regression_train1_5[count_train1_5] = new_training_label[i]
            count_train1_5+=1
        else:
            x_regression_train3[count_train3] = x_new_train[i]
            y_regression_train3[count_train3] = new_training_label[i]
            count_train3+=1
            
    x_regression_train1_5 = x_regression_train1_5[:count_train1_5]
    y_regression_train1_5 = y_regression_train1_5[:count_train1_5]
    x_regression_train3 = x_regression_train3[:count_train3]
    y_regression_train3 = y_regression_train3[:count_train3]
    
    
            
    x_regression_test1_5 = np.zeros(x_new_test.shape)
    y_regression_test1_5 = np.zeros(len(new_testing_label))
    x_regression_test3 = np.zeros(x_new_test.shape)
    y_regression_test3 = np.zeros(len(new_testing_label))
    
    count_test1_5 = 0
    count_test3 = 0    
    for i in range(len(prediction_2nd)):
        if prediction_2nd[i] == 1:
            x_regression_test1_5[count_test1_5] = x_new_test[i]
            y_regression_test1_5[count_test1_5] = new_testing_label[i]
            count_test1_5+=1
        else:
            x_regression_test3[count_test3] = x_new_test[i]
            y_regression_test3[count_test3] = new_testing_label[i]
            count_test3+=1
    x_regression_test1_5 = x_regression_test1_5[:count_test1_5]
    y_regression_test1_5 = y_regression_test1_5[:count_test1_5]
    x_regression_test3 = x_regression_test3[:count_test3]
    y_regression_test3 = y_regression_test3[:count_test3]
    
    regression_1_5 = [x_regression_train1_5, y_regression_train1_5, x_regression_test1_5, y_regression_test1_5]
    regression_3 = [x_regression_train3, y_regression_train3, x_regression_test3, y_regression_test3]
    
    return regression_1_5, regression_3
    
    




    