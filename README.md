# cs-433-project-2-banachtarski
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
git clone https://github.com/CS-433/cs-433-project-2-banachtarski.git
cd cs-433-project-2-banachtarski
```


**Download datasets and models**<br>
In order to be able to run all the notebooks you need to download the competition datasets and our pre-trained models from this link https://drive.google.com/drive/folders/1MVSqgMLR-OkzOL6bVrwHX3tVu0SC7Qpp?usp=sharing <br>
**Folder input**:  contains datasets for competition, pre-trained models<br>
**Folder output**: contains empty folders in which the notebooks will write the trained models and submissions<br>
Unzip the downloaded .zip files and copy the folders **input** and **output** inside the directory of cs-433-project-2-banachtarski, so that if you execute ls you will obtain:

```
ls
Directory: your_path_to_home\Desktop\projectML\cs-433-project-2-banachtarski

Mode                 LastWriteTime         Length Name
----                 -------------         ------ ----
d-----        13/12/2020     18:29                code
d-----        10/12/2020     09:30                input
d-----        14/12/2020     15:59                output
-a----        13/12/2020     15:47             94 .gitignore
-a----        13/12/2020     15:47             68 README.md
```

**Alternatively**<br>
Since this folders are pretty heavy to download (2.76 GB approximately), you can download **only** the output folder from the Google Drive link above, and the competition datasets here: <br>
https://www.kaggle.com/c/lish-moa/data <br>
Unzip the file lish-moa.zip, create a folder called **input** inside cs-433-project-2-banachtarski and copy the folder lish-moa inside the folder input.<br>
Commands assuming your current directory is "your_path_to_home/Downloads" and you unzipped lish-moa:

```
mkdir input
mv .\lish-moa\ .\input\ 
mv .\input\ ..\Desktop\projectML\cs-433-project-2-banachtarski\
cd ..\Desktop\projectML\cs-433-project-2-banachtarski\
ls
```

**Expected output**

```
Directory: your_path_to_home\Desktop\projectML\cs-433-project-2-banachtarski

Mode                 LastWriteTime         Length Name
----                 -------------         ------ ----
d-----        16/12/2020     11:02                code
d-----        16/12/2020     11:27                input
-a----        16/12/2020     11:02             94 .gitignore
-a----        16/12/2020     11:02           2062 README.md
```
Following the last procedure you won't be able to run the notebook blend.ipynb, since you don't have the pre-trained models.

**Installing necessary python packages**<br>
Run the following commands in order to have the necessary packages to be able to run the notebooks:
```
pip install torch

pip install pytorch-tabnet

pip install iterative-stratification

pip install tensorflow
```

## Last but not least
Before running the notebooks, ensure that your current working directory is: <br>
```
your_path_to_home\Desktop\projectML\cs-433-project-2-banachtarski\code 
```
This is **very important**, otherwise you won't be able to read the input files from the notebooks
