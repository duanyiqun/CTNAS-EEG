import numpy as np
import pickle
import mne
import scipy.io as sio
import torch
import torch.nn.functional as F
import os
from torchaudio import transforms as freq_transforms

s1_start_second = [30, 132, 287, 555, 773, 982, 1271, 1628, 1730, 2025, 2227, 2435, 2667, 2932, 3204]
s1_end_second = [102, 228, 524, 742, 920, 1240, 1568, 1697, 1994, 2166, 2401, 2607, 2901, 3172, 3359]

s2_start_second = [30, 299, 548, 646, 836, 1000, 1091, 1392, 1657, 1809, 1966, 2186, 2333, 2490, 2741]
s2_end_second = [267, 488, 614, 773, 967, 1059, 1331, 1622, 1777, 1908, 2153, 2302, 2428, 2709, 2817]

s3_start_second = [30, 353, 478, 674, 825, 908, 1200, 1346, 1451, 1711, 2055, 2307, 2457, 2726, 2888]
s3_end_second = [321, 418, 643, 764, 877, 1147, 1284, 1418, 1679, 1996, 2275, 2425, 2664, 2857, 3066]

Disgust = 0
Fear = 1
Sad = 2
Neutral = 3
Happy = 4

s1_emotion_label = [Happy, Fear, Neutral, Sad, Disgust, Happy, Fear, Neutral, Sad, Disgust, Happy, Fear, Neutral, Sad, Disgust]
s2_emotion_label = [Sad, Fear, Neutral, Disgust, Happy, Happy, Disgust, Neutral, Sad, Fear, Neutral, Happy, Fear, Sad, Disgust]
s3_emotion_label = [Sad, Fear, Neutral, Disgust, Happy, Happy, Disgust, Neutral, Sad, Fear, Neutral, Happy, Fear, Sad, Disgust]


session_dict = {'1': [s1_start_second, s1_end_second, s1_emotion_label],
                '2': [s2_start_second, s2_end_second, s2_emotion_label],
                '3': [s3_start_second, s3_end_second, s3_emotion_label]}
sample_freq = 1000


def subject_process_stamp(data_path=None, subject=6, session=3, trails_len=15, sample_freq=sample_freq,
                          target_freq=250, target_seg=7*400):
    # use mne to load the file "6_3_20180802.cnt"
    eeg_raw = mne.io.read_raw_cnt(data_path)
    """
    ch_names = eeg_raw.ch_names
    print(ch_names)
    print(len(ch_names))
    print('\n')
    # drop non-used channels
    useless_ch = ['M1', 'M2', 'VEO', 'HEO']
    eeg_raw.drop_channels(useless_ch)
    new_ch = eeg_raw.ch_names
    """
    useless_ch = ['M1', 'M2', 'VEO', 'HEO']
    eeg_raw.drop_channels(useless_ch)
    data_matrix = eeg_raw.get_data()
    print("The datamatrix shape without Eye is {}".format(data_matrix.shape))
    data_trails = []
    data_trail_labels = []
    resampler = freq_transforms.Resample(orig_freq= 16000, new_freq = 16000, resampling_method = "sinc_interpolation")
    for i in range(trails_len):
        session_start_stamp = session_dict[session][0]
        session_end_stamp = session_dict[session][1]
        data_trial_cr = data_matrix[:, session_start_stamp[i] * sample_freq: session_end_stamp[i] * sample_freq]
        # print("original datashape is {}".format(data_trial_cr.shape))
        # data_trial_cr = F.avg_pool1d(torch.tensor(data_trial_cr), kernel_size=8, stride=4)
        data_trial_cr = resampler(torch.tensor(data_trial_cr))
        data_trial_cr = data_trial_cr.numpy()
        # print("datashape after 1000hz to 250hz projection is {}".format(data_trial_cr.shape))
        current_label = session_dict[session][2][i]
        current_label = np.expand_dims(np.asarray(current_label), axis=0)
        for seg_num in range(data_trial_cr.shape[1]//target_seg):
            data_trail_seg = data_trial_cr[:, seg_num*target_seg:(seg_num+1)*target_seg]
            data_trail_seg = data_trail_seg[np.newaxis, :]
            print("datashape after slicing segments is {}".format(data_trail_seg.shape))
            print("data label after seg is {}".format(current_label))
            data_trails.append(data_trail_seg)
            data_trail_labels.append(current_label)
    data_trails = np.concatenate(tuple(data_trails), axis=0)
    data_trail_labels = np.concatenate(tuple(data_trail_labels), axis=0)
    print(data_trails.shape)
    print(data_trail_labels.shape)
    return data_trails, data_trail_labels


# 

def mix_subjects_mix_session(data_folder_path = "/Users/xxx/Downloads/BCI data/SEED-V-multimodal/EEG_raw/", output_dir="", test_ratio = 0.3):
    whole_data_trails, whole_data_labels = [],[]
    
    for path,dir_list,file_list in os.walk(data_folder_path):  
        for file_name in file_list:  
            file_abs_path = os.path.join(path, file_name)
            print(file_abs_path)
            if not file_name.endswith(".cnt"):
                continue
            else:
                # print(file_name.split("_"))
                subject, session, _ = file_name.split("_")
                data_trails, data_trail_labels = subject_process_stamp(data_path=file_abs_path, subject=subject, session=session, trails_len=15)
                whole_data_trails.append(data_trails)
                whole_data_labels.append(data_trail_labels)
    whole_data_trails = np.concatenate(whole_data_trails, axis=0)
    whole_data_labels = np.concatenate(whole_data_labels, axis=0)
    print("whole dataset samples lenth is {}".format(whole_data_trails.shape))
    print("whole dataset labels lenth is {}".format(whole_data_labels.shape))
    idx = list(range(whole_data_labels.shape[0]))
    np.random.shuffle(idx)
    samples_x = whole_data_trails[idx]
    samples_y = whole_data_labels[idx]
    
    isExist = os.path.exists(output_dir)
    if not isExist:
        # Create a new directory because it does not exist 
        os.makedirs(output_dir)
        print("The new output dir is created!")
    
    train_X = F.normalize(torch.tensor(samples_x[:int(len(samples_x)*(1-test_ratio)), :, :]), p=2.0, dim=2).numpy()
    train_y = samples_y[:int(len(samples_y)*(1-test_ratio))]
    test_X = F.normalize(torch.tensor(samples_x[int(len(samples_x)*(1-test_ratio)):, :, :]), p=2.0, dim=2).numpy()
    test_y = samples_y[int(len(samples_y)*(1-test_ratio)):]
    
    unique_name = "mix_subject_mix_session_all_in_one.mat"
    output_file = os.path.join(output_dir, unique_name)
    # sio.savemat(output_file, {"train_x": train_X, "train_y": train_y, "test_x": test_X, "test_y": test_y})
    dataset = {"train_x": train_X, "train_y": train_y, "test_x": test_X, "test_y": test_y}
    # scipy.io.matlab.miobase.MatWriteError: Matrix too large to save with Matlab 5 format so use pickle
    with open(output_file, "wb") as writer:
        pickle.dump(dataset, writer, protocol = 4)

        
    
def mix_subjects_cross_session(data_folder_path = "/Users/xxx/Downloads/BCI data/SEED-V-multimodal/EEG_raw/", output_dir="", test_ratio = 0.3):
    whole_data_trails_train, whole_data_labels_train = [],[]
    whole_data_trails_test, whole_data_labels_test = [],[]
    for path,dir_list,file_list in os.walk(data_folder_path):  
        for file_name in file_list:  
            file_abs_path = os.path.join(path, file_name)
            print(file_abs_path)
            if not file_name.endswith(".cnt"):
                continue
            else:
                # print(file_name.split("_"))
                subject, session, _ = file_name.split("_")
                data_trails, data_trail_labels = subject_process_stamp(data_path=file_abs_path, subject=subject, session=session, trails_len=15)
                if session == '3':
                    whole_data_trails_test.append(data_trails)
                    whole_data_labels_test.append(data_trail_labels)
                else:
                    whole_data_trails_train.append(data_trails)
                    whole_data_labels_train.append(data_trail_labels)
    whole_data_trails_train = np.concatenate(whole_data_trails_train, axis=0)
    whole_data_labels_train = np.concatenate(whole_data_labels_train, axis=0)
    print("whole dataset samples lenth for train is {}".format(whole_data_trails_train.shape))
    print("whole dataset labels lenth for train is {}".format(whole_data_labels_train.shape))

    whole_data_trails_test = np.concatenate(whole_data_trails_test, axis=0)
    whole_data_labels_test = np.concatenate(whole_data_labels_test, axis=0)
    print("whole dataset samples lenth for test is {}".format(whole_data_trails_test.shape))
    print("whole dataset labels lenth for test is {}".format(whole_data_labels_test.shape))

    idx = list(range(whole_data_trails_train.shape[0]))
    np.random.shuffle(idx)
    train_X = F.normalize(torch.tensor(whole_data_trails_train[idx]), p=2.0, dim=2).numpy()
    train_y = whole_data_labels_train[idx]

    idx = list(range(whole_data_trails_test.shape[0]))
    np.random.shuffle(idx)
    test_X = F.normalize(torch.tensor(whole_data_trails_test[idx]), p=2.0, dim=2).numpy()
    test_y = whole_data_labels_test[idx]

    isExist = os.path.exists(output_dir)
    if not isExist:
        # Create a new directory because it does not exist 
        os.makedirs(output_dir)
        print("The new output dir is created!")
    
    unique_name = "mix_subject_cross_session_all_in_one.mat"
    output_file = os.path.join(output_dir, unique_name)
    # sio.savemat(output_file, {"train_x": train_X, "train_y": train_y, "test_x": test_X, "test_y": test_y})
    dataset = {"train_x": train_X, "train_y": train_y, "test_x": test_X, "test_y": test_y}
    # scipy.io.matlab.miobase.MatWriteError: Matrix too large to save with Matlab 5 format so use pickle
    with open(output_file, "wb") as writer:
        pickle.dump(dataset, writer, protocol = 4)
    
                
                
def sep_subjects_mix_session(data_folder_path = "/Users/xxx/Downloads/BCI data/SEED-V-multimodal/EEG_raw/", output_dir="", test_ratio = 0.3, single_subject_range=15):
    
    # path,dir_list,file_list = os.walk(data_folder_path)
    file_name_list = []
    for path,dir_list,file_list in os.walk(data_folder_path):  
        file_name_list.append(file_list)
        print(file_list)
   
    
    for current_subject in range(single_subject_range):
        whole_data_trails, whole_data_labels = [],[]
        for file_name in file_list:  
            file_abs_path = os.path.join(path, file_name)
            if not file_name.endswith(".cnt"):
                continue
            else:
                # print(file_name.split("_"))
                subject, session, _ = file_name.split("_")
                if subject == str(current_subject+1):
                    print(file_abs_path)
                    data_trails, data_trail_labels = subject_process_stamp(data_path=file_abs_path, subject=subject, session=session, trails_len=15)
                    whole_data_trails.append(data_trails)
                    whole_data_labels.append(data_trail_labels)
                
        whole_data_trails = np.concatenate(whole_data_trails, axis=0)
        whole_data_labels = np.concatenate(whole_data_labels, axis=0)
        print("whole dataset samples lenth is {}".format(whole_data_trails.shape))
        print("whole dataset labels lenth is {}".format(whole_data_labels.shape))
        idx = list(range(whole_data_labels.shape[0]))
        np.random.shuffle(idx)
        samples_x = whole_data_trails[idx]
        samples_y = whole_data_labels[idx]

        isExist = os.path.exists(output_dir)
        if not isExist:
            # Create a new directory because it does not exist 
            os.makedirs(output_dir)
            print("The new output dir is created!")
        
        train_X = F.normalize(torch.tensor(samples_x[:int(len(samples_x)*(1-test_ratio)), :, :]), p=2.0, dim=2).numpy()
        train_y = samples_y[:int(len(samples_y)*(1-test_ratio))]
        test_X = F.normalize(torch.tensor(samples_x[int(len(samples_x)*(1-test_ratio)):, :, :]), p=2.0, dim=2).numpy()
        test_y = samples_y[int(len(samples_y)*(1-test_ratio)):]

        unique_name = "seed_subject_{}_session_all_in_one.mat".format(current_subject)
        output_file = os.path.join(output_dir, unique_name)
        # sio.savemat(output_file, {"train_x": train_X, "train_y": train_y, "test_x": test_X, "test_y": test_y})
        dataset = {"train_x": train_X, "train_y": train_y, "test_x": test_X, "test_y": test_y}
        # scipy.io.matlab.miobase.MatWriteError: Matrix too large to save with Matlab 5 format so use pickle
        with open(output_file, "wb") as writer:
            pickle.dump(dataset, writer, protocol = 4)


def sep_subjects_cross_session(data_folder_path = "/Users/xxx/Downloads/BCI data/SEED-V-multimodal/EEG_raw/", output_dir="", test_ratio = 0.3, single_subject_range=15):
    
    file_name_list = []
    for path,dir_list,file_list in os.walk(data_folder_path):  
        file_name_list.append(file_list)
        print(file_list)
   
    for current_subject in range(single_subject_range):
        whole_data_trails_train, whole_data_labels_train = [],[]
        whole_data_trails_test, whole_data_labels_test = [],[]
        for file_name in file_list:  
            file_abs_path = os.path.join(path, file_name)

            if not file_name.endswith(".cnt"):
                continue
            else:
                # print(file_name.split("_"))
                subject, session, _ = file_name.split("_")
                if subject == str(current_subject+1):
                    print(file_abs_path)
                    data_trails, data_trail_labels = subject_process_stamp(data_path=file_abs_path, subject=subject, session=session, trails_len=15)
                    if session == '3':
                        whole_data_trails_test.append(data_trails)
                        whole_data_labels_test.append(data_trail_labels)
                    else:
                        whole_data_trails_train.append(data_trails)
                        whole_data_labels_train.append(data_trail_labels)
                
        whole_data_trails_train = np.concatenate(whole_data_trails_train, axis=0)
        whole_data_labels_train = np.concatenate(whole_data_labels_train, axis=0)
        print("whole dataset samples lenth for train is {}".format(whole_data_trails_train.shape))
        print("whole dataset labels lenth for train is {}".format(whole_data_labels_train.shape))

        whole_data_trails_test = np.concatenate(whole_data_trails_test, axis=0)
        whole_data_labels_test = np.concatenate(whole_data_labels_test, axis=0)
        print("whole dataset samples lenth for test is {}".format(whole_data_trails_test.shape))
        print("whole dataset labels lenth for test is {}".format(whole_data_labels_test.shape))

        idx = list(range(whole_data_trails_train.shape[0]))
        np.random.shuffle(idx)
        train_X = F.normalize(torch.tensor(whole_data_trails_train[idx]), p=2.0, dim=2).numpy()
        train_y = whole_data_labels_train[idx]

        idx = list(range(whole_data_trails_test.shape[0]))
        np.random.shuffle(idx)
        test_X = F.normalize(torch.tensor(whole_data_trails_test[idx]), p=2.0, dim=2).numpy()
        test_y = whole_data_labels_test[idx]

        isExist = os.path.exists(output_dir)
        if not isExist:
            # Create a new directory because it does not exist 
            os.makedirs(output_dir)
            print("The new output dir is created!")

        unique_name = "seed_subject_{}_session_cross.mat".format(current_subject)
        output_file = os.path.join(output_dir, unique_name)
        # sio.savemat(output_file, {"train_x": train_X, "train_y": train_y, "test_x": test_X, "test_y": test_y})
        dataset = {"train_x": train_X, "train_y": train_y, "test_x": test_X, "test_y": test_y}
        # scipy.io.matlab.miobase.MatWriteError: Matrix too large to save with Matlab 5 format so use pickle
        with open(output_file, "wb") as writer:
            pickle.dump(dataset, writer, protocol = 4)
    

if __name__ == "__main__":
    # raw_signal_path = "/home/xxx/Data/bci/data/seed_v/EEG_raw/6_3_20180802.cnt"
    # raw_signal_path = "/Users/xxx/Downloads/BCI data/SEED-V-multimodal/EEG_raw/6_3_20180802.cnt"
    # subject_process_stamp(data_path=raw_signal_path, subject=6, session=3)

    # start formal scripts 
    # mix_subjects_mix_session(data_folder_path="/home/xxx/Data/bci/data/seed_v/EEG_raw/", output_dir="/home/xxx/Data/bci/EEG_uniform_darts/seed_v/mixed_subjets_mixed_session_nooverlap/")
    # mix_subjects_cross_session(data_folder_path="/home/xxx/Data/bci/data/seed_v/EEG_raw/", output_dir="/home/xxx/Data/bci/EEG_uniform_darts/seed_v/mixed_subjets_cross_session_nooverlap/")
    sep_subjects_mix_session(data_folder_path="/home/xxx/Data/bci/data/seed_v/EEG_raw/", output_dir="/home/xxx/Data/bci/EEG_uniform_darts/seed_v/sep_subjets_mixed_session_nooverlap/")
    sep_subjects_cross_session(data_folder_path="/home/xxx/Data/bci/data/seed_v/EEG_raw/", output_dir="/home/xxx/Data/bci/EEG_uniform_darts/seed_v/sep_subjets_cross_session_nooverlap/")




