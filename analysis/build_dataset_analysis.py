import numpy as np
import pickle
import scipy.io as scio
import os
import torch
import torch.nn.functional as F

def get_labels(label_path):
    '''
        得到15个 trials 对应的标签
    :param label_path: 标签文件对应的路径
    :return: list，对应15个 trials 的标签，1 for positive, 0 for neutral, -1 for negative
    '''
    return scio.loadmat(label_path, verify_compressed_data_integrity=False)['label'][0]


def label_2_onehot(label_list):
    '''
        将原始-1， 0， 1标签转化为独热码形式
    :param label_list: 原始标签列表
    :return label_onehot: 独热码形式标签列表
    '''
    look_up_table = {-1: [1, 0, 0],
                     0: [0, 1, 0],
                     1: [0, 0, 1]}
    label_onehot = [np.asarray(look_up_table[label]) for label in label_list]
    return label_onehot


def get_frequency_band_idx(frequency_band):
    """
        获得频带对应的索引，仅对 ExtractedFeatures 目录下的数据有效
    :param frequency_band: 频带名称，'delta', 'theta', 'alpha', 'beta', 'gamma'
    :return idx: 频带对应的索引
    """
    lookup = {'delta': 0,
              'theta': 1,
              'alpha': 2,
              'beta': 3,
              'gamma': 4}
    return lookup[frequency_band]


def build_preprocessed_eeg_dataset_CNN(folder_path):
    '''
        预处理后的 EEG 数据维度为 62 * N，其中62为 channel 数量， N 为采样点个数（已下采样到200 Hz）
        此函数将预处理后的 EEG 信号转化为 CNN 网络所对应的数据格式，即 62 * 200 的二维输入（每 1s 的信号作为一个样本）,区分开不同 trial 的数据
    :param folder_path: Preprocessed_EEG 文件夹对应的路径
    :return feature_vector_dict, label_dict: 分别为样本的特征向量，样本的标签，key 为被试名字，val 为该被试对应的特征向量或标签的 list，方便 subject-independent 的测试
    '''
    feature_vector_dict = {}
    label_dict = {}
    labels = get_labels(os.path.join(folder_path, 'label.mat'))
    try:
        all_mat_file = os.walk(folder_path)
        skip_set = {'label.mat', 'readme.txt'}
        file_cnt = 0
        for path, dir_list, file_list in all_mat_file:
            for file_name in file_list:
                file_cnt += 1
                print('当前已处理到{}，总进度{}/{}'.format(file_name, file_cnt, len(file_list)))
                if file_name not in skip_set:
                    all_trials_dict = scio.loadmat(os.path.join(folder_path, file_name),
                                                   verify_compressed_data_integrity=False)
                    experiment_name = file_name.split('.')[0]
                    feature_vector_trial_dict = {}
                    label_trial_dict = {}
                    for key in all_trials_dict.keys():
                        if 'eeg' not in key:
                            continue
                        feature_vector_list = []
                        label_list = []
                        cur_trial = all_trials_dict[key]  # 维度为 62 * N，每200个采样点截取一个样本，不足200时舍弃
                        length = len(cur_trial[0])
                        pos = 0
                        while pos + 200 <= length:
                            feature_vector_list.append(np.asarray(cur_trial[:, pos:pos + 200]))
                            raw_label = labels[int(key.split('_')[-1][3:]) - 1]  # 截取片段对应的 label，-1, 0, 1
                            label_list.append(raw_label)
                            pos += 200
                        trial = key.split('_')[1][3:]
                        feature_vector_trial_dict[trial] = np.asarray(feature_vector_list)
                        label_trial_dict[trial] = np.asarray(label_2_onehot(label_list))

                    feature_vector_dict[experiment_name] = feature_vector_trial_dict
                    label_dict[experiment_name] = label_trial_dict
                else:
                    continue

    except FileNotFoundError as e:
        print('加载数据时出错: {}'.format(e))

    return feature_vector_dict, label_dict