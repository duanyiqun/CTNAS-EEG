""" Generate commands for pre-train phase. """
import os


def run_exp(weight_lr=0.1, alpha_lr=0.05, gamma=0.5, step_size=20, pre_batch_size=8):
    max_epoch = 240
    shot = 20
    query = 10
    way = 4
    gpu = 3
    weight_lr = weight_lr
    alpha_lr = alpha_lr
    
    the_command = 'python3 lauch.py' \
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
        + ' --phase=dep_single' \
        + ' --Search_nodes=2' \
        + ' --model_type=Search' \
        + ' --single_id=3' \
        + ' --exp_spc=Migrate_Seed_subject3'

    os.system(the_command)


run_exp(weight_lr=0.01, alpha_lr=0.005, gamma=0.5, step_size=20, pre_batch_size=32)


def run_exp(weight_lr=0.1, alpha_lr=0.05, gamma=0.5, step_size=20, pre_batch_size=8):
    max_epoch = 240
    shot = 20
    query = 10
    way = 4
    gpu = 2
    weight_lr = weight_lr
    alpha_lr = alpha_lr
    
    the_command = 'python3 lauch.py' \
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
        + ' --phase=dep_single' \
        + ' --Search_nodes=2' \
        + ' --model_type=Search' \
        + ' --single_id=4' \
        + ' --exp_spc=Migrate_Seed__subject4'

    os.system(the_command)


run_exp(weight_lr=0.01, alpha_lr=0.005, gamma=0.5, step_size=20, pre_batch_size=32)



def run_exp(weight_lr=0.1, alpha_lr=0.05, gamma=0.5, step_size=20, pre_batch_size=8):
    max_epoch = 240
    shot = 20
    query = 10
    way = 4
    gpu = 2
    weight_lr = weight_lr
    alpha_lr = alpha_lr
    
    the_command = 'python3 lauch.py' \
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
        + ' --phase=dep_single' \
        + ' --Search_nodes=2' \
        + ' --model_type=Search' \
        + ' --single_id=5' \
        + ' --exp_spc=Migrate_Seed__subject5'

    os.system(the_command)


run_exp(weight_lr=0.01, alpha_lr=0.005, gamma=0.5, step_size=20, pre_batch_size=32)


def run_exp(weight_lr=0.1, alpha_lr=0.05, gamma=0.5, step_size=20, pre_batch_size=8):
    max_epoch = 240
    shot = 20
    query = 10
    way = 4
    gpu = 2
    weight_lr = weight_lr
    alpha_lr = alpha_lr
    
    the_command = 'python3 lauch.py' \
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
        + ' --phase=dep_single' \
        + ' --Search_nodes=2' \
        + ' --model_type=Search' \
        + ' --single_id=6' \
        + ' --exp_spc=Migrate_Seed__subject6'

    os.system(the_command)


run_exp(weight_lr=0.01, alpha_lr=0.005, gamma=0.5, step_size=20, pre_batch_size=32)


def run_exp(weight_lr=0.1, alpha_lr=0.05, gamma=0.5, step_size=20, pre_batch_size=8):
    max_epoch = 240
    shot = 20
    query = 10
    way = 4
    gpu = 2
    weight_lr = weight_lr
    alpha_lr = alpha_lr
    
    the_command = 'python3 lauch.py' \
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
        + ' --phase=dep_single' \
        + ' --Search_nodes=2' \
        + ' --model_type=Search' \
        + ' --single_id=7' \
        + ' --exp_spc=Migrate_Seed__subject7'

    os.system(the_command)


run_exp(weight_lr=0.01, alpha_lr=0.005, gamma=0.5, step_size=20, pre_batch_size=32)



def run_exp(weight_lr=0.1, alpha_lr=0.05, gamma=0.5, step_size=20, pre_batch_size=8):
    max_epoch = 240
    shot = 20
    query = 10
    way = 4
    gpu = 2
    weight_lr = weight_lr
    alpha_lr = alpha_lr
    
    the_command = 'python3 lauch.py' \
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
        + ' --phase=dep_single' \
        + ' --Search_nodes=2' \
        + ' --model_type=Search' \
        + ' --single_id=8' \
        + ' --exp_spc=Migrate_Seed_subject8'

    os.system(the_command)


run_exp(weight_lr=0.01, alpha_lr=0.005, gamma=0.5, step_size=20, pre_batch_size=32)


def run_exp(weight_lr=0.1, alpha_lr=0.05, gamma=0.5, step_size=20, pre_batch_size=8):
    max_epoch = 240
    shot = 20
    query = 10
    way = 4
    gpu = 2
    weight_lr = weight_lr
    alpha_lr = alpha_lr
    
    the_command = 'python3 lauch.py' \
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
        + ' --phase=dep_single' \
        + ' --Search_nodes=2' \
        + ' --model_type=Search' \
        + ' --single_id=9' \
        + ' --exp_spc=SMigrate_Seed_subject9'

    os.system(the_command)


run_exp(weight_lr=0.01, alpha_lr=0.005, gamma=0.5, step_size=20, pre_batch_size=32)


def run_exp(weight_lr=0.1, alpha_lr=0.05, gamma=0.5, step_size=20, pre_batch_size=8):
    max_epoch = 240
    shot = 20
    query = 10
    way = 4
    gpu = 2
    weight_lr = weight_lr
    alpha_lr = alpha_lr

    the_command = 'python3 lauch.py' \
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
        + ' --phase=dep_single' \
        + ' --Search_nodes=2' \
        + ' --model_type=Search' \
        + ' --single_id=1' \
        + ' --exp_spc=Migrate_Seed_subject1'

    os.system(the_command)


run_exp(weight_lr=0.01, alpha_lr=0.005, gamma=0.5, step_size=20, pre_batch_size=32)


def run_exp(weight_lr=0.1, alpha_lr=0.05, gamma=0.5, step_size=20, pre_batch_size=8):
    max_epoch = 240
    shot = 20
    query = 10
    way = 4
    gpu = 2
    weight_lr = weight_lr
    alpha_lr = alpha_lr

    the_command = 'python3 lauch.py' \
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
        + ' --phase=dep_single' \
        + ' --Search_nodes=2' \
        + ' --model_type=Search' \
        + ' --single_id=2' \
        + ' --exp_spc=Migrate_Seed_subject2'

    os.system(the_command)


run_exp(weight_lr=0.01, alpha_lr=0.005, gamma=0.5, step_size=20, pre_batch_size=32)
