import numpy as np
import time
import scipy
import scipy.signal
import scipy.io
# import self defined functions 
from torch.utils.data import Dataset
import pickle
import random
import torch
import scipy.io as sio
from scipy import interp
from torchaudio import transforms as freq_transforms 
import torch.nn.functional as F


class DatasetLoader_BCI_IV_subjects(Dataset):

    def __init__(self, setname, args, train_aug=False):

        subject_id = 1
        if args.data_folder is None:
            data_folder = '../data'
        else:
            data_folder = args.data_folder
        data = sio.loadmat(data_folder + "/cross_sub/cross_subject_data_" + str(subject_id) + ".mat")
        test_X = data["test_x"][:, :, 750:1500]  # [trials, channels, time length]
        train_X = data["train_x"][:, :, 750:1500]

        test_y = data["test_y"].ravel()
        train_y = data["train_y"].ravel()

        train_y -= 1
        test_y -= 1
        window_size = 400
        step = 50
        n_channel = 22

        def windows(data, size, step):
            start = 0
            while (start + size) < data.shape[0]:
                yield int(start), int(start + size)
                start += step

        def segment_signal_without_transition(data, window_size, step):
            segments = []
            for (start, end) in windows(data, window_size, step):
                if len(data[start:end]) == window_size:
                    segments = segments + [data[start:end]]
            return np.array(segments)

        def segment_dataset(X, window_size, step):
            win_x = []
            for i in range(X.shape[0]):
                win_x = win_x + [segment_signal_without_transition(X[i], window_size, step)]
            win_x = np.array(win_x)
            return win_x

        train_raw_x = np.transpose(train_X, [0, 2, 1])
        test_raw_x = np.transpose(test_X, [0, 2, 1])

        train_win_x = segment_dataset(train_raw_x, window_size, step)
        test_win_x = segment_dataset(test_raw_x, window_size, step)
        train_win_y = train_y
        test_win_y = test_y

        expand_factor = train_win_x.shape[1]

        train_x = np.reshape(train_win_x, (-1, train_win_x.shape[2], train_win_x.shape[3]))
        test_x = np.reshape(test_win_x, (-1, test_win_x.shape[2], test_win_x.shape[3]))
        train_y = np.repeat(train_y, expand_factor)
        test_y = np.repeat(test_y, expand_factor)

        train_x = np.reshape(train_x, [train_x.shape[0], 1, train_x.shape[1], train_x.shape[2]]).astype('float32')
        train_y = np.reshape(train_y, [train_y.shape[0]]).astype('float32')

        test_x = np.reshape(test_x, [test_x.shape[0], 1, test_x.shape[1], test_x.shape[2]]).astype('float32')
        test_y = np.reshape(test_y, [test_y.shape[0]]).astype('float32')

        test_x = test_x[2000:, :, :, :]
        test_y = test_y[2000:]

        val_x = test_x[:2000, :, :, :]
        val_y = test_y[:2000]

        train_win_x = train_win_x.astype('float32')

        val_win_x = test_win_x[:400, :, :, :].astype('float32')
        val_win_y = test_win_y[:400]

        test_win_x = test_win_x[400:, :, :, :].astype('float32')
        test_win_y = test_win_y[400:]

        self.X_val = val_win_x
        self.y_val = val_win_y

        if setname == 'train':
            self.data = train_win_x
            self.label = train_win_y
        elif setname == 'val':
            self.data = val_win_x
            self.label = val_win_y
        elif setname == 'test':
            self.data = test_win_x
            self.label = test_win_y

        self.num_class = 4

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        data, label = self.data[i], self.label[i]
        return data, label


class DatasetLoader_SEED_V_mix(Dataset):

    def __init__(self, setname, args, train_aug=False):

        subject_id = 'all'
        if args.data_folder is None:
            data_folder = '../data'
        else:
            data_folder = args.data_folder
        if args.mix_session == 'True':
            if args.seed_no_overlap =='True':
                data_path = "/home/xxx/Data/bci/EEG_uniform_darts/seed_v/mixed_subjets_mixed_session_nooverlap/mix_subject_mix_session_all_in_one.mat"
            else:
                data_path = "/home/xxx/Data/bci/EEG_uniform_darts/seed_v/mixed_subjets_mixed_session/mix_subject_mix_session_all_in_one.mat"
        else:
            if args.seed_no_overlap =='True':
                data_path = "/home/xxx/Data/bci/EEG_uniform_darts/seed_v/mixed_subjets_cross_session_nooverlap/mix_subject_cross_session_all_in_one.mat"
            else:
                data_path = "/home/xxx/Data/bci/EEG_uniform_darts/seed_v/mixed_subjets_cross_session/mix_subject_cross_session_all_in_one.mat"
        # data = sio.loadmat(data_folder + "/mix_sub/mix_subject_data_" + str(subject_id) + ".mat")
        with open(data_path, 'rb') as pickle_file:
            data = pickle.load(pickle_file)
            #try:
            #    while True:
            #        data = pickle.load(pickle_file)
            #except EOFError:
            #    print("pickle file is too large to read, even in seq")

        test_X = data["test_x"]  # [trials, channels, time length]
        train_X = data["train_x"]
        test_y = data["test_y"].ravel()
        train_y = data["train_y"].ravel()

        window_size = 400
        if args.seed_no_overlap == 'True':
            step = 400
        else:
            step = 50
        n_channel = 62

        def windows(data, size, step):
            start = 0
            while (start + size) < data.shape[0]:
                yield int(start), int(start + size)
                start += step

        def segment_signal_without_transition(data, window_size, step):
            segments = []
            for (start, end) in windows(data, window_size, step):
                if len(data[start:end]) == window_size:
                    segments = segments + [data[start:end]]
            return np.array(segments)

        def segment_dataset(X, window_size, step):
            win_x = []
            for i in range(X.shape[0]):
                win_x = win_x + [segment_signal_without_transition(X[i], window_size, step)]
                # print(segment_signal_without_transition(X[i], window_size, step).shape)
            win_x = np.array(win_x)
            # print(win_x.shape)
            return win_x

        # train_raw_x = np.transpose(train_X, [0, 2, 1])
        # test_raw_x = np.transpose(test_X, [0, 2, 1])

        train_win_x = segment_dataset(np.transpose(train_X, [0, 2, 1]), window_size, step)
        test_win_x = segment_dataset(np.transpose(test_X, [0, 2, 1]), window_size, step)
        train_win_y = train_y
        test_win_y = test_y
        print(train_win_x.shape)

        expand_factor = train_win_x.shape[1]

        train_x = np.reshape(train_win_x, (-1, train_win_x.shape[2], train_win_x.shape[3]))
        test_x = np.reshape(test_win_x, (-1, test_win_x.shape[2], test_win_x.shape[3]))
        train_y = np.repeat(train_y, expand_factor)
        test_y = np.repeat(test_y, expand_factor)

        train_x = np.reshape(train_x, [train_x.shape[0], 1, train_x.shape[1], train_x.shape[2]]).astype('float32')
        train_y = np.reshape(train_y, [train_y.shape[0]]).astype('float32')

        test_x = np.reshape(test_x, [test_x.shape[0], 1, test_x.shape[1], test_x.shape[2]]).astype('float32')
        test_y = np.reshape(test_y, [test_y.shape[0]]).astype('float32')

        # test_x = test_x[2000:, :, :, :]
        # test_y = test_y[2000:]

        # val_x = test_x[:2000, :, :, :]
        # val_y = test_y[:2000]

        train_win_x = train_win_x.astype('float32')
        
        ratio = 0.5
        idx = list(range(len(test_win_y)))
        np.random.shuffle(idx)
        test_win_x = test_win_x[idx]
        test_win_y = test_win_y[idx]

        val_win_x = test_win_x[:int(len(test_win_x)*0.5), :, :, :].astype('float32')
        val_win_y = test_win_y[:int(len(test_win_x)*0.5)]

        real_test_win_x = test_win_x[int(len(test_win_x)*0.5):, :, :, :].astype('float32')
        real_test_win_y = test_win_y[int(len(test_win_x)*0.5):]

        self.X_val = val_win_x
        self.y_val = val_win_y

        if setname == 'train':
            self.data = train_win_x
            self.label = train_win_y
        elif setname == 'val':
            self.data = val_win_x
            self.label = val_win_y
        elif setname == 'test':
            self.data = real_test_win_x
            self.label = real_test_win_y

        self.num_class = 5

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        data, label = F.normalize(torch.Tensor(self.data[i]), p=2, dim=2).numpy(), self.label[i]
        return data, label


class DatasetLoader_SEED_V_sep(Dataset):
    def __init__(self, setname, args, train_aug=False, subject_id=3):

        if args.data_folder is None:
            data_folder = '../data'
        else:
            data_folder = args.data_folder
        if args.mix_session == 'True':
            if args.seed_no_overlap =='True':
                data_path = "/home/xxx/Data/bci/EEG_uniform_darts/seed_v/sep_subjets_mixed_session_nooverlap/seed_subject_{}_session_all_in_one.mat".format(str(subject_id))
            else:
                data_path = "/home/xxx/Data/bci/EEG_uniform_darts/seed_v/sep_subjets_mixed_session/seed_subject_{}_session_all_in_one.mat".format(str(subject_id))
        else:
            if args.seed_no_overlap =='True':
                data_path = "/home/xxx/Data/bci/EEG_uniform_darts/seed_v/sep_subjets_cross_session_nooverlap/seed_subject_{}_session_cross.mat".format(str(subject_id))
            else:
                data_path = "/home/xxx/Data/bci/EEG_uniform_darts/seed_v/sep_subjets_cross_session/seed_subject_{}_session_cross.mat".format(str(subject_id))
        # data = sio.loadmat(data_folder + "/mix_sub/mix_subject_data_" + str(subject_id) + ".mat")
        with open(data_path, 'rb') as pickle_file:
            data = pickle.load(pickle_file)
            #try:
            #    while True:
            #        data = pickle.load(pickle_file)
            #except EOFError:
            #    print("pickle file is too large to read, even in seq")

        test_X = data["test_x"]  # [trials, channels, time length]
        train_X = data["train_x"]
        test_y = data["test_y"].ravel()
        train_y = data["train_y"].ravel()

        window_size = 400
        if args.seed_no_overlap == 'True':
            step = 400
        else:
            step = 50
        n_channel = 62

        def windows(data, size, step):
            start = 0
            # for no overlap
            # while (start + size) <= data.shape[0]:
            # for overlap
            while (start + size) < data.shape[0]:
                yield int(start), int(start + size)
                start += step

        def segment_signal_without_transition(data, window_size, step):
            segments = []
            for (start, end) in windows(data, window_size, step):
                if len(data[start:end]) == window_size:
                    segments = segments + [data[start:end]]
            return np.array(segments)

        def segment_dataset(X, window_size, step):
            win_x = []
            for i in range(X.shape[0]):
                win_x = win_x + [segment_signal_without_transition(X[i], window_size, step)]
            win_x = np.array(win_x)
            return win_x

        # train_raw_x = np.transpose(train_X, [0, 2, 1])
        # test_raw_x = np.transpose(test_X, [0, 2, 1])

        train_win_x = segment_dataset(np.transpose(train_X, [0, 2, 1]), window_size, step)
        test_win_x = segment_dataset(np.transpose(test_X, [0, 2, 1]), window_size, step)
        train_win_y = train_y
        test_win_y = test_y
        print(train_win_x.shape)

        expand_factor = train_win_x.shape[1]

        train_x = np.reshape(train_win_x, (-1, train_win_x.shape[2], train_win_x.shape[3]))
        test_x = np.reshape(test_win_x, (-1, test_win_x.shape[2], test_win_x.shape[3]))
        train_y = np.repeat(train_y, expand_factor)
        test_y = np.repeat(test_y, expand_factor)

        train_x = np.reshape(train_x, [train_x.shape[0], 1, train_x.shape[1], train_x.shape[2]]).astype('float32')
        train_y = np.reshape(train_y, [train_y.shape[0]]).astype('float32')

        test_x = np.reshape(test_x, [test_x.shape[0], 1, test_x.shape[1], test_x.shape[2]]).astype('float32')
        test_y = np.reshape(test_y, [test_y.shape[0]]).astype('float32')

        # test_x = test_x[2000:, :, :, :]
        # test_y = test_y[2000:]

        # val_x = test_x[:2000, :, :, :]
        # val_y = test_y[:2000]

        train_win_x = train_win_x.astype('float32')
        
        ratio = 0.5
        idx = list(range(len(test_win_y)))
        np.random.shuffle(idx)
        test_win_x = test_win_x[idx]
        test_win_y = test_win_y[idx]

        val_win_x = test_win_x[:int(len(test_win_x)*0.5), :, :, :].astype('float32')
        val_win_y = test_win_y[:int(len(test_win_x)*0.5)]

        real_test_win_x = test_win_x[int(len(test_win_x)*0.5):, :, :, :].astype('float32')
        real_test_win_y = test_win_y[int(len(test_win_x)*0.5):]

        self.X_val = val_win_x
        self.y_val = val_win_y

        if setname == 'train':
            self.data = train_win_x
            self.label = train_win_y
        elif setname == 'val':
            self.data = val_win_x
            self.label = val_win_y
        elif setname == 'test':
            self.data = real_test_win_x
            self.label = real_test_win_y

        self.num_class = 5

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        data, label = F.normalize(torch.Tensor(self.data[i]), p=2, dim=2).numpy(), self.label[i]
        return data, label