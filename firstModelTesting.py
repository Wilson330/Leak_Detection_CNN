
import os
import numpy as np
import librosa
from scipy.fft import fft
from keras.models import load_model
from Cut_Audio import cut_and_save

PATH = "./testFiles/"
SEC = 2
SHIFT = 0.2

cut_and_save(PATH, sec=SEC, shift=SHIFT)

filePath = PATH + "shift=" + str(SHIFT)
testingData = []
testingLabel = []

for fileName in os.listdir(filePath):
    x, fs = librosa.load(fileName, sr=None)
    X = fft(x)[1:3201] / len(x) * 2
    data = np.array([X.real, X.imag]).reshape((80,80,1))
    testingData.append(data)
    
    name = fileName.split("_")
    
    if(float(name[1]) > 3.0):
        testingLabel.append(0)
    else:
        testingLabel.append(1)

testingData = np.array(testingData)
testingLabel = np.array(testingLabel)
    
# Testing the training model

model1 = load_model('./first_Cmodel.h5')
score = model1.evaluate(testingData, testingLabel)
prediction = np.argmax(model1.predict(testingData),axis=1)
print("Testing data :\n", testingLabel[:,1].astype(int))
print("Prediction :\n", prediction)
print("Accuracy of testing data = {:2.1f}%".format(score[1]*100.0))
print('First Classification done')