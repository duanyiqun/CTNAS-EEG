""" Generate commands for pre-train phase. """
import os


def run_exp(lr=0.05, gamma=0.5, step_size=20, pre_batch_size=8):
    max_epoch = 240
    shot = 20
    query = 10
    way = 4
    gpu = 2
    base_lr = 0.01
    weight_lr=0.01
    alpha_lr=0.01
    searched_structure_path = '/data00/home/xxx/BCI/Mudus/Mudus_BCI/logs/normal_search/BCI_IV_Search_batchsize32_w_lr0.01_alpha_lr0.005_gamma0.5_step20_maxepoch240_Single_Search_Formal_7_val_node_2_layer4_space_subject1/max_acc.pth'
    
    the_command = 'python3 lauch.py' \
        + ' --pre_max_epoch=' + str(max_epoch) \
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
        + ' --phase=dep_single' \
        + ' --model_type=single_retrain' \
        + ' --exp_spc=specific_retrain_argmax_drop_prob_0_subject_1' \
        + ' --w_lr=' + str(weight_lr) \
        + ' --alpha_lr=' + str(alpha_lr) 
    os.system(the_command)


run_exp(lr=0.1, gamma=0.5, step_size=20, pre_batch_size=256)


