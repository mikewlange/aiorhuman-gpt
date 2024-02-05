# Setup

-   clearml and clearml-serving set up and install.
-   Install and setup
    -   Setup your [**ClearML Server**](https://github.com/allegroai/clearml-server) or use the [Free tier Hosting](https://app.clear.ml)
    -   Setup local access (if you haven't already), see instructions [here](https://clear.ml/docs/latest/docs/getting_started/ds/ds_first_steps#install-clearml)
    -   install docker
    -   setup a CONDA environemnt for the project, if you want.

## Install ClearML

First, [sign up for free](https://app.clear.ml)

Install the `clearml` python package:

```bash
pip install clearml
```

## Connect ClearML SDK to the Server
1.  Execute the following command to run the ClearML setup wizard:

```

clearml-init
```

## Install Docker
https://www.docker.com/get-started/

## Install ngrok 
ngrok.com

## Setup Conda Env - Optional

```sh
conda create -n "py39_demo" python=3.9 ipython
conda activate py39_demo
pip install textstat 
pip install Markdown
pip install benepar
pip install clearml1
pip install clearml_serving 
pip install empath
pip install interpret
pip install ipython 
pip install ipywidgets 
pip install joblib
pip install matplotlib 
pip install nltk
pip install numpy 
pip install optuna
pip install pandas 
pip install requests
pip install scikit_learn 
pip install seaborn 
pip install sentence_transformers
pip install spacy 
pip install tensorboard
pip install tensorflow
pip install textstat
pip install torch
pip install torchvision
pip install tqdm
pip install transformers 
python -c 'import benepar; benepar.download(\"benepar_en3\")'
python -m spacy download en_core_web_lg
```