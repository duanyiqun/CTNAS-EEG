import numpy as np
import pickle
import mne

# check the version of these modules
print(np.__version__)
print(pickle.format_version)

# load DE features named '1_123.npz'

file_path = "/home/xxx/Data/bci/data/seed_v/EEG_DE_features/1_123.npz"
data_npz = np.load(file_path)
print(data_npz.files)

# get data and label
# ** both 'data' and 'label' are pickled dict **

data = pickle.loads(data_npz['data'])
label = pickle.loads(data_npz['label'])

print(data.keys())
print(label.keys())

# As we can see, there are 45 keys in both 'data' and 'label'.
# Each participant toaok part in our experiments for 3 sessions, and he/she watched 15 movie clips (i.e. 15 trials) during each session.
# Therefore, we could extract 3 * 15 = 45 DE feature matrices.

# The key indexes [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14] belong to Session 1.
# The key indexes [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29] belong to Session 2.
# The key indexes [30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44] belong to Session 3.

# We will print the emotion labels for each trial.
label_dict = {0:'Disgust', 1:'Fear', 2:'Sad', 3:'Neutral', 4:'Happy'}

for i in range(45):
    print('Session {} -- Trial {} -- EmotionLabel : {}'.format(i//15+1, i%15+1, label_dict[label[i][0]]))
    print("The datashape of DE is {}".format(data[i].shape))
    print("The datashape of label is {}".format(label[i].shape))
    print("The datashape of ")


raw_signal_path = "/home/xxx/Data/bci/data/seed_v/EEG_raw/6_3_20180802.cnt"

# use mne to load the file "6_3_20180802.cnt"
eeg_raw = mne.io.read_raw_cnt(raw_signal_path)
print(eeg_raw)
print('\n')
print(eeg_raw.info)

# check channel names
ch_names = eeg_raw.ch_names
print(ch_names)
print(len(ch_names))
print('\n')

# drop non-used channels
useless_ch = ['M1', 'M2', 'VEO', 'HEO']
eeg_raw.drop_channels(useless_ch)
new_ch = eeg_raw.ch_names
print(new_ch)
print(len(new_ch))
print('\n')

# see raw data wave
eeg_raw.plot()

# get the data matrix
data_matrix = eeg_raw.get_data()
print(data_matrix.shape)

final_seccond = data_matrix.shape[1]/200
print(final_seccond/60)