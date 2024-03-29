
import os
import librosa
import soundfile as sf

path = "./testFiles/"

# Cut the raw data to be as desire time length and save all output files
def cut_and_save (filePath, sec=2, shift=0.2):
    # filePath: the path of sound files 
    # sec: the time length of the target sound file (default=2)
    # shift: the time shift to retrieve next time window data (default=0.2)
    
    count = 0
    
    for fileName in os.listdir(filePath):
        data, fs = librosa.load(filePath + fileName, sr=None)
        for split in range(int((len(data)/fs - sec) / shift + 1)):
            begin = int(shift * fs * split)
            end = int(begin + sec * fs)
            data_new = data[begin:end]
            
            saveDirPath = filePath + "shift=" + str(shift) + "/"
            if not os.path.exists(saveDirPath):
                os.makedirs(saveDirPath)
            savefile = saveDirPath + fileName + ' - ' + str(split+1) + '.wav'
            sf.write(savefile, data_new, fs)
            
            count = count + 1

    print('Total file number =', count)


#%%
if __name__ == "__main__":
    cut_and_save(path, sec=2, shift=0.2)


