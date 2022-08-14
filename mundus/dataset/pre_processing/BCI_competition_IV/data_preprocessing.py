import os
import scipy.io as sio
import pickle
import numpy as np
import argparse


#######################
# reference: github: TNTLFreiburg: braindecode/examples/bcic_iv_2a.py
#######################


def subject_process(subject_list, data_path='./data/', num_channels=22, num_tests=6 * 48, window_length=7 * 250):
    data = []
    label = []
    for i in range(len(subject_list)):
        subject = subject_list[i]
        class_return = np.zeros(num_tests)
        data_return = np.zeros((num_tests, num_channels, window_length))

        num_valid_trial = 0

        a = sio.loadmat(data_path + 'A0' + str(subject) + 'T.mat')
        b = sio.loadmat(data_path + 'A0' + str(subject) + 'E.mat')
        a_data = a['data']
        for ii in range(0, a_data.size):
            a_data1 = a_data[0, ii]
            a_data2 = [a_data1[0, 0]]
            a_data3 = a_data2[0]
            a_X = a_data3[0]
            a_trial = a_data3[1]
            a_y = a_data3[2]
            a_fs = a_data3[3]
            a_classes = a_data3[4]
            a_artifacts = a_data3[5]
            a_gender = a_data3[6]
            a_age = a_data3[7]
            for trial in range(0, a_trial.size):
                if a_artifacts[trial] == 0:
                    data_return[num_valid_trial, :, :] = np.transpose(
                        a_X[int(a_trial[trial]):(int(a_trial[trial]) + window_length), :22])
                    class_return[num_valid_trial] = int(a_y[trial])
                    num_valid_trial += 1
        data.append(data_return[0:num_valid_trial, :, :])
        label.append(class_return[0:num_valid_trial])

        num_valid_trial = 0
        b_data = b['data']
        for ii in range(0, b_data.size):
            b_data1 = b_data[0, ii]
            b_data2 = [b_data1[0, 0]]
            b_data3 = b_data2[0]
            b_X = b_data3[0]
            b_trial = b_data3[1]
            b_y = b_data3[2]
            b_fs = b_data3[3]
            b_classes = b_data3[4]
            b_artifacts = b_data3[5]
            b_gender = b_data3[6]
            b_age = b_data3[7]
            for trial in range(0, b_trial.size):
                if b_artifacts[trial] == 0:
                    data_return[num_valid_trial, :, :] = np.transpose(
                        b_X[int(b_trial[trial]):(int(b_trial[trial]) + window_length), :22])
                    class_return[num_valid_trial] = int(b_y[trial])
                    num_valid_trial += 1
        data.append(data_return[0:num_valid_trial, :, :])
        label.append(class_return[0:num_valid_trial])
        # print(data_return.shape)
    data = tuple(data)
    label = tuple(label)
    data = np.concatenate(data, axis=0)
    label = np.concatenate(label, axis=0)

    return data, label


def bci_iv_meta_handler(data_path, num_test_subject=1):
    for j in range(1, 1 + num_test_subject):
        train_subject = [k for k in range(1, 10) if k != j]
        test_subject = [j]
        train_dataset = subject_process(train_subject, data_path=data_path)
        test_dataset = subject_process(test_subject, data_path=data_path)

        train_X = train_dataset[0]
        train_y = train_dataset[1]
        test_X = test_dataset[0]
        test_y = test_dataset[1]
        print(train_X.shape)
        print(train_y.shape)
        print(test_X.shape)
        print(test_y.shape)
        idx = list(range(len(train_y)))
        np.random.shuffle(idx)
        train_X = train_X[idx]
        train_y = train_y[idx]
        idx = list(range(len(test_y)))
        np.random.shuffle(idx)
        test_X = test_X[idx]
        test_y = test_y[idx]
        train_X = train_X[:12000, :, :]
        train_y = train_y[:12000]
        output_path = os.path.join(data_path, 'cross_sub')
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        file_path = '{}/{}{}{}'.format(output_path, 'cross_subject_data_', str(j), '.mat')
        sio.savemat(file_path, {"train_x": train_X, "train_y": train_y, "test_x": test_X, "test_y": test_y})


def bci_iv_single_handler(data_path, test_ratio = 0.1):
    raw_subject = [k for k in range(1, 10)]
    for subject_i in range(1, 10):
        train_dataset = subject_process([subject_i], data_path=data_path)

        train_X = train_dataset[0]
        train_y = train_dataset[1]
        print(train_X.shape)
        print(train_y.shape)
        idx = list(range(len(train_y)))
        np.random.shuffle(idx)
        train_X = train_X[idx]
        train_y = train_y[idx]
    
        real_train_X = train_X[:int(len(train_y)*(1-test_ratio)), :, :]
        real_train_y = train_y[:int(len(train_y)*(1-test_ratio))]
        test_X = train_X[int(len(train_y)*(1-test_ratio)):, :, :]
        test_y = train_y[int(len(train_y)*(1-test_ratio)):]
        output_path = os.path.join(data_path, 'single_sep')
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        file_path = '{}/{}{}{}'.format(output_path, 'single_subject_data_', str(subject_i), '.mat')
        sio.savemat(file_path, {"train_x": real_train_X, "train_y": real_train_y, "test_x": test_X, "test_y": test_y})


def bci_iv_normal_handler(data_path, test_ratio = 0.2):
    raw_subject = [k for k in range(1, 10)]
    train_dataset = subject_process(raw_subject, data_path=data_path)

    train_X = train_dataset[0]
    train_y = train_dataset[1]
    print(train_X.shape)
    print(train_y.shape)
    idx = list(range(len(train_y)))
    np.random.shuffle(idx)
    train_X = train_X[idx]
    train_y = train_y[idx]
    
    real_train_X = train_X[:int(len(train_y)*(1-test_ratio)), :, :]
    real_train_y = train_y[:int(len(train_y)*(1-test_ratio))]
    test_X = train_X[int(len(train_y)*(1-test_ratio)):, :, :]
    test_y = train_y[int(len(train_y)*(1-test_ratio)):]
    output_path = os.path.join(data_path, 'mix_sub')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    file_path = '{}/{}{}{}'.format(output_path, 'mix_subject_data_', str('all'), '.mat')
    sio.savemat(file_path, {"train_x": real_train_X, "train_y": real_train_y, "test_x": test_X, "test_y": test_y})



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data/bci_iv/')
    args = parser.parse_args()
    # bci_iv_meta_handler(args.data_path)
    # bci_iv_normal_handler(args.data_path)
    bci_iv_single_handler(args.data_path)