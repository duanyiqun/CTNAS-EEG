""" Generate commands for pre-train phase. """
import os


def run_exp(lr=0.05, gamma=0.5, step_size=20, pre_batch_size=8):
    max_epoch = 240
    shot = 20
    query = 10
    way = 4
    gpu = 3
    base_lr = 0.01
    weight_lr=0.02
    alpha_lr=0.01
    input_channel=62

    searched_structure_path = '/home/xxx/Data/bci/EEG_MI_DARTS/Mudus_BCI/logs_seed/normal_search/seed_v_Search_seed_batchsize24_w_lr0.01_alpha_lr0.005_gamma0.5_step20_maxepoch240_all_mixsub_mixsession/max_acc.pth'

    the_command = 'python3 lauch.py' \
        + ' --pre_max_epoch=' + str(max_epoch) \
        + ' --dataset=' + str('seed_v') \
        + ' --input_channels=' + str(input_channel) \
        + ' --shot=' + str(shot) \
        + ' --train_query=' + str(query) \
        + ' --way=' + str(way) \
        + ' --pre_step_size=' + str(step_size) \
        + ' --pre_gamma=' + str(gamma) \
        + ' --gpu=' + str(gpu) \
        + ' --base_lr=' + str(base_lr) \
        + ' --pre_lr=' + str(lr) \
        + ' --pre_batch_size=' + str(pre_batch_size) \
        + ' --searched_weights=' + str(searched_structure_path) \
        + ' --phase=dependent' \
        + ' --model_type=Search_retrain_seed' \
        + ' --exp_spc=all_mixsub_mix_session_retrain' \
        + ' --w_lr=' + str(weight_lr) \
        + ' --alpha_lr=' + str(alpha_lr) \
        + ' --mix_session=True'
    os.system(the_command)


run_exp(lr=0.01, gamma=0.5, step_size=20, pre_batch_size=128)


