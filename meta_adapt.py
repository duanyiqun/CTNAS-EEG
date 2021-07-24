""" Generate commands for meta-adaptation phase. """
import os


def run_exp(num_batch=12, shot=20, query=15, lr1=0.0001, lr2=0.0005, base_lr=0.005, update_step=20, gamma=0.8):
    max_epoch = 20
    way = 4
    step_size = 3
    gpu = 0

    the_command = 'python3 lauch.py' \
                  + ' --max_epoch=' + str(max_epoch) \
                  + ' --num_batch=' + str(num_batch) \
                  + ' --shot=' + str(shot) \
                  + ' --train_query=' + str(query) \
                  + ' --way=' + str(way) \
                  + ' --meta_lr1=' + str(lr1) \
                  + ' --meta_lr2=' + str(lr2) \
                  + ' --step_size=' + str(step_size) \
                  + ' --gamma=' + str(gamma) \
                  + ' --gpu=' + str(gpu) \
                  + ' --base_lr=' + str(base_lr) \
                  + ' --update_step=' + str(update_step)

    os.system(the_command + ' --phase=meta_train')
    os.system(the_command + ' --phase=meta_eval')


print("Eval model with hyper setting {}".format(
    'num_batch=12, shot=1, query=15, lr1=0.0001, lr2=0.005, base_lr=0.005, update_step=20, gamma=0.8'))
run_exp(num_batch=12, shot=1, query=15, lr1=0.0001, lr2=0.005, base_lr=0.005, update_step=20, gamma=0.8)
print("Eval model with hyper setting {}".format(
    'num_batch=12, shot=5, query=15, lr1=0.0001, lr2=0.005, base_lr=0.005, update_step=20, gamma=0.8'))
run_exp(num_batch=12, shot=5, query=15, lr1=0.0001, lr2=0.005, base_lr=0.005, update_step=20, gamma=0.8)
print("Eval model with hyper setting {}".format(
    'num_batch=12, shot=10, query=15, lr1=0.0001, lr2=0.005, base_lr=0.005, update_step=20, gamma=0.8'))
run_exp(num_batch=12, shot=10, query=15, lr1=0.0001, lr2=0.005, base_lr=0.005, update_step=20, gamma=0.8)
print("Eval model with hyper setting {}".format(
    'num_batch=12, shot=20, query=15, lr1=0.0001, lr2=0.005, base_lr=0.005, update_step=20, gamma=0.8'))
run_exp(num_batch=12, shot=20, query=15, lr1=0.0001, lr2=0.005, base_lr=0.005, update_step=20, gamma=0.8)



