""" Generate commands for pre-train phase. """
import os


def run_exp(weight_lr=0.1, alpha_lr=0.05, gamma=0.5, step_size=20, pre_batch_size=8):
    max_epoch = 240
    shot = 20
    query = 10
    way = 4
    gpu = 0
    weight_lr = weight_lr
    alpha_lr = alpha_lr
    input_channel = 62
    
    the_command = 'python3 lauch.py' \
        + ' --dataset=' + str('seed_v') \
        + ' --input_channels=' + str(input_channel) \
        + ' --pre_max_epoch=' + str(max_epoch) \
        + ' --shot=' + str(shot) \
        + ' --train_query=' + str(query) \
        + ' --way=' + str(way) \
        + ' --pre_step_size=' + str(step_size) \
        + ' --pre_gamma=' + str(gamma) \
        + ' --gpu=' + str(gpu) \
        + ' --w_lr=' + str(weight_lr) \
        + ' --alpha_lr=' + str(alpha_lr) \
        + ' --pre_batch_size=' + str(pre_batch_size) \
        + ' --phase=dependent' \
        + ' --Search_nodes=1' \
        + ' --model_type=Search_seed' \
        + ' --exp_spc=overlap_mix_session_mix_subjects_channel_final' \
        + ' --mix_session=True' \
        + ' --seed_no_overlap=False'

    os.system(the_command)


run_exp(weight_lr=0.01, alpha_lr=0.005, gamma=0.5, step_size=20, pre_batch_size=18)


def run_exp(weight_lr=0.1, alpha_lr=0.05, gamma=0.5, step_size=20, pre_batch_size=8):
    max_epoch = 240
    shot = 20
    query = 10
    way = 4
    gpu = 0
    weight_lr = weight_lr
    alpha_lr = alpha_lr
    input_channel = 62
    
    the_command = 'python3 lauch.py' \
        + ' --dataset=' + str('seed_v') \
        + ' --input_channels=' + str(input_channel) \
        + ' --pre_max_epoch=' + str(max_epoch) \
        + ' --shot=' + str(shot) \
        + ' --train_query=' + str(query) \
        + ' --way=' + str(way) \
        + ' --pre_step_size=' + str(step_size) \
        + ' --pre_gamma=' + str(gamma) \
        + ' --gpu=' + str(gpu) \
        + ' --w_lr=' + str(weight_lr) \
        + ' --alpha_lr=' + str(alpha_lr) \
        + ' --pre_batch_size=' + str(pre_batch_size) \
        + ' --phase=dependent' \
        + ' --Search_nodes=1' \
        + ' --model_type=Search_seed' \
        + ' --exp_spc=overlap_cross_session_mix_subjects_channel_final' \
        + ' --mix_session=False' \
        + ' --seed_no_overlap=False'

    os.system(the_command)


run_exp(weight_lr=0.01, alpha_lr=0.005, gamma=0.5, step_size=20, pre_batch_size=18)