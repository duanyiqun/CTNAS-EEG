
### Cross Task Neural Architecture Search for EEG Signal Classifications


<img src=./images/metanet.png width = "780" height = "220"  align=center />

A convenient code base for fast model customization of EEG models. By introducing NAS into EEG signals, hopefully we could increase the automatic  level  of designing models for BCI signals. 

Also, an implementation of paper [Cross Task Neural Architecture Search for EEG Signal Recognition](https://arxiv.org/pdf/2210.06298.pdf).

### Citation
<div style='display: none'>

If you find this research is useful for you research

```angular2html
@article{duan2022cross,
  title={Cross Task Neural Architecture Search for EEG Signal Classifications},
  author={Duan, Yiqun and Wang, Zhen and Li, Yi and Tang, Jianhang and Wang, Yu-Kai and Lin, Chin-Teng},
  journal={arXiv preprint arXiv:2210.06298},
  year={2022}
}
```
</div>

###  Installation

The code is basically based on Pytorch 1.12.0, tensorboardX, and mne
Run is follow to prepare the environment of mundus.
```ssh
git clone this repo
cd repodir
pip install -r requirements.txt
```
###  Data preparation

Considering privacy issue, we do not provide any instant brain dynamics data through this repo.

Download BCI-IV dataset through link [https://www.bbci.de/competition/iv/](https://www.bbci.de/competition/iv/)
Then run data preparation scripts in mundus. 

###  Training

The ```./lauch.py``` file contains entry for all the model structures in CTNAS. 

An example for mixed subject training, this process comeout with a searched structure.
Here, Search_nodes denotes the node number in the searching space. 
```python
max_epoch = 240
shot = 20
query = 10
way = 4
gpu = 0
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
    + ' --phase=dependent' \
    + ' --Search_nodes=2' \
    + ' --model_type=Search' \
    + ' --exp_spc=allsubject_alpha_exp1_reim'
os.system(the_command)
```

An example for fix structure and retrain for high accuracy. 

```python
max_epoch = 240
shot = 20
query = 10
way = 4
gpu = 3
base_lr = 0.01
weight_lr=0.02
alpha_lr=0.01
searched_structure_path = '/data00/home/xx/BCI/Mudus/Mudus_BCI/logs/normal_search/BCI_IV_Search_batchsize32_w_lr0.01_alpha_lr0.005_gamma0.5_step20_maxepoch240_Mix_Search_Formal_4_val_node_2_layer4_new_search_space_with_skip_Elu_flattennoadapp/max_acc.pth'

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
    + ' --phase=dependent' \
    + ' --model_type=Search_retrain' \
    + ' --exp_spc=current_best_retrain_argmax_drop_prob_0_subject_all_exp2' \
    + ' --w_lr=' + str(weight_lr) \
    + ' --alpha_lr=' + str(alpha_lr) 
os.system(the_command)
```

We can customize the neural structure for each subject as well.
- Please refer to [single_train_search_sequence.py](single_train_search_sequence.py) for customize structures for each subject.
- Then refer to [single_train_search_retrain_squence.py](single_train_search_retrain_squence.py) for retrain the searched structures on each subject. 


###  Notes

1. Cross task neural architecture searching for EEG signals, refer to ```./mundus/models/backbones/DARTS/```, currently the constraint code is removed for training stability. 
2. Visualization utilities of the searched results, refer to  ```./mundus/visualization/search_visual```
3. Simple data preparation for common Motor Imaginary and Emotion datasets. Strongly recommend first run on BCI-Comp-IV for verification as Emotion signals are much noisy. 
4. Launch training through ```./mundus/runners/```
5. Training curve visualization through tensorboard.

####  Training example visualization

The training example on BCI-IV Competition IV 2a datasets:

<img src=./images/vis_example.png width = "450" height = "260" alt="图片名称" align=center />


