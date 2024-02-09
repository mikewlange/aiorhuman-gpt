## &#8592;[Back](../README.md)
# Set up Environment
## Install ClearML

First, [sign up for free](https://app.clear.ml)

Install the `clearml` python package:

```bash
pip install clearml
```

## Connect ClearML SDK to the Server
1.  Execute the following command to run the ClearML setup wizard:

```sh
clearml-init
```

## Install Docker
https://www.docker.com/get-started/

## Install ngrok 
ngrok.com

## Setup Conda Env - Optional

```sh
conda create -n "py39_demo" python=3.9 ipython
```
```sh
conda activate py39_demo
```
```sh
pip3 install Markdown
pip3 install benepar
pip3 install clearml
pip3 install clearml_serving 
pip3 install empath
pip3 install interpret
pip3 install ipython 
pip3 install ipywidgets 
pip3 install joblib
pip3 install matplotlib 
pip3 install nltk
pip3 install numpy 
pip3 install optuna
pip3 install pandas 
pip3 install requests
pip3 install scikit_learn 
pip3 install seaborn 
pip3 install sentence_transformers
pip3 install spacy 
pip3 install tensorboard
pip3 install tensorflow
pip3 install textstat
pip3 install torch
pip3 install torchvision
pip3 install tqdm
pip3 install transformers 
pip3 install bs4
```

```sh
python3 -c 'import benepar; benepar.download("benepar_en3")'
python3 -m spacy download en_core_web_lg
```