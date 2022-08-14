""" Generate commands for pre-train phase. """
import os

#######mix session########

def run_exp(single_id = 1, weight_lr=0.1, alpha_lr=0.05, gamma=0.5, step_size=20, pre_batch_size=8):
    max_epoch = 240
    shot = 20
    query = 10
    way = 4
    gpu = 0
    weight_lr = weight_lr
    alpha_lr = alpha_lr
    input_channel=62
    # single_id = 1
    
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
        + ' --w_lr=' + str(weight_lr) \
        + ' --alpha_lr=' + str(alpha_lr) \
        + ' --pre_batch_size=' + str(pre_batch_size) \
        + ' --phase=dep_single' \
        + ' --Search_nodes=1' \
        + ' --model_type=Search_seed' \
        + ' --single_id=' + str(single_id) \
        + ' --exp_spc=overlap_single_highlr_channel_remain_mixsession_' +str("s{}".format(single_id)) \
        + ' --mix_session=True' \
        + ' --seed_no_overlap=False'
    os.system(the_command)

for i in range(15):
    if i < 2:
        continue
    else:
        run_exp(single_id=i, weight_lr=0.01, alpha_lr=0.005, gamma=0.5, step_size=20, pre_batch_size=12)



#######cross session########

def run_exp(single_id = 1, weight_lr=0.1, alpha_lr=0.05, gamma=0.5, step_size=20, pre_batch_size=24):
    max_epoch = 240
    shot = 20
    query = 10
    way = 4
    gpu = 3
    weight_lr = weight_lr
    alpha_lr = alpha_lr
    input_channel=62
    # single_id = 1
    
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
        + ' --w_lr=' + str(weight_lr) \
        + ' --alpha_lr=' + str(alpha_lr) \
        + ' --pre_batch_size=' + str(pre_batch_size) \
        + ' --phase=dep_single' \
        + ' --Search_nodes=1' \
        + ' --model_type=Search_seed' \
        + ' --single_id=' + str(single_id) \
        + ' --exp_spc=overlap_single_highlr_channel_remain_crosssession_' +str("s{}".format(single_id)) \
        + ' --mix_session=False' \
        + ' --seed_no_overlap=False'
    os.system(the_command)
    
for i in range(15):
    if i < 2:
        continue
    else:
        run_exp(single_id=i, weight_lr=0.01, alpha_lr=0.005, gamma=0.5, step_size=20, pre_batch_size=12)



