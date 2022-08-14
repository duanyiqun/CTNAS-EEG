""" Generate commands for pre-train phase. """
import os


def run_exp(lr=0.05, gamma=0.5, step_size=20, pre_batch_size=20):
    max_epoch = 30
    shot = 20
    query = 10
    way = 4
    gpu = 0
    base_lr = 0.005
    
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
        + ' --phase=pre_train' 

    os.system(the_command)


run_exp(lr=0.05, gamma=0.5, step_size=20, pre_batch_size=256) 


