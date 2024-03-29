
#################################
### YOU WON'T USE THIS FILE!! ###
#################################

import numpy as np
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.models import load_model, Model
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, Input, Activation
from data_preprocessing import data_preprocessing_1st, preprocessing_2nd, preprocessing_3rd

'''### Step-by-step training models for each part ###
    
    We have total 4 model here:
1. CNN classification model (<3m or not)
2. CNN classification model (<1.5m or not)
3. CNN regression model (actual distance in 1.5m)
4. CNN regression model (actual distance in 3m)  <- Redundant, just testing

'''
#%% Split original data to training and testing set at first (labeled by distance in 3m or not)
x_train, y_train, x_test, y_test, training_label, testing_label = data_preprocessing_1st(2, 0.2, 0.8, 0)

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print('data complete')

#%% 1. First CNN classification model (to decide whether the distance is in 3m or not)
Img_1 = Input(shape=(80,80,1))

x1 = Conv2D(filters=36., kernel_size=(5,5), padding='same')(Img_1)
x1 = MaxPooling2D(pool_size=(2,2),padding='same')(x1) 
x1 = Activation('relu')(x1)
x1 = Conv2D(filters=36, kernel_size=(5,5), padding='same')(x1)
x1 = MaxPooling2D(pool_size=(2,2),padding='same')(x1) 
x1 = Activation('relu')(x1)
x1 = Dropout(0.25)(x1)
x1 = Flatten()(x1)
x = Dense(units=128,activation='relu')(x1)
x = Dense(units=128,activation='relu')(x)
x = Dense(units=2,activation='softmax')(x)
model1=Model([Img_1], x)
model1.summary()

#%%
savebestmodel = ModelCheckpoint('./first_Cmodel_new.h5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto')
model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
train_history = model1.fit(x_train, y_train, batch_size=50, epochs=50, callbacks=[savebestmodel],
                            validation_split=0.2, verbose=1)
#%%
model1 = load_model('./model/first_Cmodel.h5')
score = model1.evaluate(x_test, y_test)
prediction = np.argmax(model1.predict(x_test),axis=1)
print("Testing data :\n", y_test[:,1].astype(int))
print("Prediction :\n", prediction)
print("Accuracy of testing data = {:2.1f}%".format(score[1]*100.0))
print('First Classification done')

labels = [0, 0.3, 0.5, 0.6, 1, 1.5, 2, 2.5, 3, 5, 6, 8, 10]
classdict = {key:[] for key in labels}
for label in labels:
    target = []
    for i in range(len(testing_label)):
        if testing_label[i] == label:
            target.append(prediction[i])
    accuracy = round(np.sum(target)/len(target),4)*100
    classdict[label].append((accuracy, len(target)))
train_prediction = np.argmax(model1.predict(x_train),axis=1)
#%%
score_env = model1.evaluate(x_test, y_test)
prediction_env = np.argmax(model1.predict(x_test),axis=1)
print("Testing data :\n", y_test[:,1].astype(int))
print("Prediction :\n", prediction_env)
print("Accuracy of testing data = {:2.1f}%".format(score_env[1]*100.0))
print('First Classification done')


#%% Relabeling the training and testing set by distance in 1.5m or not)
x_new_train, y_new_train, x_new_test, y_new_test, new_training_label, new_testing_label = preprocessing_2nd(x_train, train_prediction, training_label, x_test, testing_label, prediction)
y_new_train = np_utils.to_categorical(y_new_train)
y_new_test = np_utils.to_categorical(y_new_test)

#%% 2. Second CNN classification model (to decide whether the distance is in 1.5m or not)
Img_1 = Input(shape=(80,80,1))
x1 = Conv2D(filters=36, kernel_size=(5,5), padding='same')(Img_1)
x1 = MaxPooling2D(pool_size=(2,2),padding='same')(x1) 
x1 = Activation('relu')(x1)
x1 = Conv2D(filters=36, kernel_size=(5,5), padding='same')(x1)
x1 = MaxPooling2D(pool_size=(2,2),padding='same')(x1) 
x1 = Activation('relu')(x1)
x1 = Dropout(0.25)(x1)
x1 = Flatten()(x1)
x = Dense(units=512,activation='relu')(x1)
x = Dense(units=128,activation='relu')(x)
x = Dense(units=2,activation='softmax')(x)
model2=Model([Img_1], x)
model2.summary()


#%% 
savebestmodel2 = ModelCheckpoint('./first_Cmodel2_new.h5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto')
model2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
train_history2 = model2.fit(x_new_train, y_new_train, batch_size=50, epochs=50, callbacks=[savebestmodel2],
                          validation_split=0.2, verbose=1)
#%%
model2 = load_model('./model/first_Cmodel2.h5')
score_2nd = model2.evaluate(x_new_test, y_new_test)
prediction_2nd = np.argmax(model2.predict(x_new_test),axis=1)
print("Testing data :\n", y_new_test[:,1].astype(int))
print("Prediction :\n", prediction_2nd)
print("Accuracy of testing data = {:2.1f}%".format(score_2nd[1]*100.0))

classdict_2 = {key:[] for key in labels}
for label in labels:
    target = []
    for i in range(len(new_testing_label)):
        if new_testing_label[i] == label:
            target.append(prediction_2nd[i])
    accuracy = round(np.sum(target)/len(target),4)*100
    classdict_2[label].append((accuracy, len(target)))

train_prediction_2nd = np.argmax(model2.predict(x_new_train),axis=1)



#%% Relabeling the training and testing set from (0/1) to the actual distance value
regression_1_5, regression_3 = preprocessing_3rd(x_new_train, train_prediction_2nd, new_training_label, x_new_test, new_testing_label, prediction_2nd)
#%%
x_regression_train1_5, y_regression_train1_5, x_regression_test1_5, y_regression_test1_5 = regression_1_5
x_regression_train3, y_regression_train3, x_regression_test3, y_regression_test3 = regression_3
#%% 3. First CNN regression model (to predict the actual distance in 1.5m)
Img_1 = Input(shape=(80,80,1))

x1 = Conv2D(filters=36., kernel_size=(5,5), padding='same')(Img_1)
x1 = MaxPooling2D(pool_size=(2,2),padding='same')(x1) 
x1 = Activation('relu')(x1)
x1 = Conv2D(filters=36, kernel_size=(5,5), padding='same')(x1)
x1 = MaxPooling2D(pool_size=(2,2),padding='same')(x1) 
x1 = Activation('relu')(x1)
x1 = Dropout(0.25)(x1)
x1 = Flatten()(x1)
x = Dense(units=128,activation='relu')(x1)
x = Dense(units=128,activation='relu')(x)
x = Dense(units=1,activation='linear')(x)
model3=Model([Img_1], x)
model3.summary()

#%% 
savebestmodel3 = ModelCheckpoint('./first_Cmodel3_new.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
model3.compile(loss='mae', optimizer='adam', metrics=['accuracy'])
train_history = model3.fit(x_regression_train1_5, y_regression_train1_5, batch_size=50, epochs=100, callbacks=[savebestmodel3], 
                          validation_split=0.2, verbose=1)
#%%
from sklearn.metrics import mean_absolute_error

model3 = load_model('./model/first_Cmodel3.h5')
score_1_5 = model3.evaluate(x_regression_test1_5, y_regression_test1_5)
prediction_1_5 = model3.predict(x_regression_test1_5)
MAE = mean_absolute_error(y_regression_test1_5, prediction_1_5)
print("Testing data :\n", y_regression_test1_5.astype(float))
print("Prediction :\n", prediction_1_5)
print("MAE of testing data = {:2.4f}".format(MAE))
#%%
Y_regression_test1_5 = y_regression_test1_5.reshape(len(y_regression_test1_5),1)
result1_5 = np.concatenate((Y_regression_test1_5, prediction_1_5, abs(Y_regression_test1_5 - prediction_1_5)), axis=1)
index = np.where(result1_5[:,0] == 2)[0][0]
#%%
d_min = 0
d_max = 10
plt.figure(figsize=(10,6))
plt.plot(np.linspace(d_min,d_max), np.linspace(d_min,d_max), lw=2, c='k', linestyle='--', label='ideal')
plt.scatter(result1_5[:index,0], result1_5[:index,1], s=16, c='b', label='test point')
plt.scatter(result1_5[index:,0], result1_5[index:,1], s=16, c='r', label='wrong point')
plt.xticks(np.arange(d_min,d_max+0.5,0.5), np.array(np.arange(d_min,d_max+0.5,0.5), dtype='str'))
plt.yticks(np.arange(d_min,d_max+0.5,0.5), np.array(np.arange(d_min,d_max+0.5,0.5), dtype='str'))
plt.title(f'Prediction result\nMAE={round(MAE,4)}')
plt.xlabel('Actual distance (m)')
plt.ylabel('Predicted distance (m)')
plt.legend()
plt.grid()


#%% 4. Second CNN regression model (to predict the actual distance in 3m)
Img_1 = Input(shape=(80,80,1))

x1 = Conv2D(filters=36., kernel_size=(5,5), padding='same')(Img_1)
x1 = MaxPooling2D(pool_size=(2,2),padding='same')(x1) 
x1 = Activation('relu')(x1)
x1 = Conv2D(filters=36, kernel_size=(5,5), padding='same')(x1)
x1 = MaxPooling2D(pool_size=(2,2),padding='same')(x1) 
x1 = Activation('relu')(x1)
x1 = Dropout(0.25)(x1)
x1 = Flatten()(x1)
x = Dense(units=128,activation='relu')(x1)
x = Dense(units=128,activation='relu')(x)
x = Dense(units=1,activation='linear')(x)
model4=Model([Img_1], x)
model4.summary()
#%%
savebestmodel4 = ModelCheckpoint('./first_Cmodel4_new.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
model4.compile(loss='mae', optimizer='adam', metrics=['accuracy'])
train_history4 = model4.fit(x_regression_train3, y_regression_train3, batch_size=50, epochs=100, callbacks=[savebestmodel4],
                          validation_split=0.2, verbose=1)
#%%
model4 = load_model('./model/first_Cmodel4.h5')
score_3 = model4.evaluate(x_regression_test3, y_regression_test3)
prediction_3 = model4.predict(x_regression_test3)
MAE_3 = mean_absolute_error(y_regression_test3, prediction_3)
print("Testing data :\n", y_regression_test3.astype(float))
print("Prediction :\n", prediction_3)
print("MAE of testing data = {:2.4f}".format(MAE_3))
#%%
Y_regression_test3 = y_regression_test3.reshape(len(y_regression_test3),1)
result_3 = np.concatenate((Y_regression_test3, prediction_3, abs(Y_regression_test3 - prediction_3)), axis=1)
index_3 = np.where(result_3[:,0] == 2)[0][0]
index_env = np.where(result_3[:,0] == 10)[0][0]

#%%  Visualize the error of regression result
d_min = 0
d_max = 10
plt.figure(figsize=(10,6))
plt.plot(np.linspace(d_min,d_max), np.linspace(d_min,d_max), lw=2, c='k', linestyle='--', label='ideal')
plt.scatter(result_3[index_3:,0], result_3[index_3:,1], s=16, c='b', label='test point')
plt.scatter(result_3[:index_3,0], result_3[:index_3,1], s=16, c='r', label='wrong point')
plt.scatter(result_3[index_env:,0], result_3[index_env:,1], s=16, c='r')
plt.xticks(np.arange(d_min,d_max+0.5,0.5), np.array(np.arange(d_min,d_max+0.5,0.5), dtype='str'))
plt.yticks(np.arange(d_min,d_max+0.5,0.5), np.array(np.arange(d_min,d_max+0.5,0.5), dtype='str'))
plt.title(f'Prediction result\nMAE={round(MAE_3,4)}')
plt.xlabel('Actual distance (m)')
plt.ylabel('Predicted distance (m)')
plt.legend()
plt.grid()