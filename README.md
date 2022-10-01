
### Cross Task Neural Architecture Search for EEG Signal Classifications
<img src=./images/metanet.png width = "700" height = "200"  align=center />

A convenient code base for fast model customization of EEG models. By introducing NAS into EEG signals, hopefully we could increase the automatic  level  of designing models for BCI signals. 

Also, an implementation of paper ``Cross Task Neural Architecture Search for EEG Signal Recognition''.
<div style='display: none'>


If you find this code base is useful for you research, please kindly cite our paper. 
```angular2html

```
</div>



#### 🌍 Notes

1. Cross task neural architecture searching for EEG signals, refer to ```./mundus/models/backbones/DARTS/```, currently the constraint code is removed for training stability. 
2. Visualization utilities of the searched results, refer to  ```./mundus/visualization/search_visual```
3. Simple data preparation for common Motor Imaginary and Emotion datasets. Strongly recommend first run on BCI-Comp-IV for verification as Emotion signals are much noisy. 
4. Launch training through ```./mundus/runners/```
5. Training curve visualization through tensorboard.

#### 💻 Installation

The code is basically based on Pytorch 1.12.0, tensorboardX, and mne
Run is follow to prepare the environment of mundus.
```ssh
git clone this repo
cd repodir
pip install -r requirements.txt
```
#### 📖 Data preparation

Considering privacy issue, we do not provide any instant brain dynamics data through this repo.
Download BCI-IV dataset through link [https://www.bbci.de/competition/iv/](https://www.bbci.de/competition/iv/)
Then run data preparation scripts in mundus. 

#### ⌚️ Training

The ```./lauch.py``` file contains entry for all the model structures in CTNAS. 

We give example search scripts in `./mix_train_search.py` for mixed-subject searching and `./mix_train_search_retrain.py` for retrain. 


#### 🚗 Training example visualization

The training example on BCI-IV Competition IV 2a datasets:

<img src=./images/vis_example.png width = "450" height = "260" alt="图片名称" align=center />


