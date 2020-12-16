﻿# cs-433-project-2-banachtarski
## Authors: Alessandro Lombardi, Niccolò Polvani, Filippo Zacchei

This repository contains notebooks to train the models and load the pre-trained models used by our team
"BanachTarski" in the Kaggle competition "Mechanisms of Action (MoA) Prediction". 
## Final position: 72nd out of 4,373 teams (top 2%)
### Final public score (25% of test dataset): 0.01826
### Final private score (hidden 75% of test dataset): 0.01610,  (1st team has 0.01599)
Like most teams in the competition, our solution is highly influenced by public notebooks, we reproduced the strongest models (mostly neural networks) changing sometimes preprocessing, architecture, hyperparameters and training (view notebooks nn.ipynb, tabnet.ipynb, multi_input_resnet.ipynb). Finally we performed a weighted average for the predictions of our models (view notebook blend.ipynb).

# How to run the notebooks
The following commands were executed on Windows10 using powershell, but similar commands will work on unix-based systems.<br>
**Prerequisites**:<br>
* git
* python 3
* scientific python packages, that come with an anaconda installation (numpy, matplotlib, pandas..)

**Download the repository**<br>
Assuming you want to install the repository on a new folder on Desktop, open Powershell and execute:

```
cd Desktop
mkdir projectML
cd projectML
git clone 
```



In order to be able to run all the notebooks you need to download the competition datasets and our pre-trained models from this link https://drive.google.com/drive/folders/1MVSqgMLR-OkzOL6bVrwHX3tVu0SC7Qpp?usp=sharing <br>
**Folder input**:  contains datasets for competition, pre-trained models<br>
**Folder output**: contains empty folders in which the notebooks will write the trained models and submissions<br>

Since this folders are pretty heavy to download (2.76 GB approximately), you can alternatively download the competition datasets here: <br>
https://www.kaggle.com/c/lish-moa/data <br>
But you won't be able to run the notebook blend.ipynb
