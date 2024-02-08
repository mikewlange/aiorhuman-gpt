# Config and Deploy Infrastructure

### COMPLETED: 
- Setup your [**ClearML Server**](https://github.com/allegroai/clearml-server) or use the [Free tier Hosting](https://app.clear.ml)
- Setup local access (if you haven't already), see instructions [here](https://clear.ml/docs/latest/docs/getting_started/ds/ds_first_steps#install-clearml)
ut not much. 

## Install clearml-serving CLI:

```shell
pip install clearml-serving #or into your virtualenv or conda env. conda active env
```

## Create the Serving Service Controller. 
```shell 
clearml-serving create --name "aiorhuman inference service demo"`
```
-   The new serving service UID should be printed `New Serving Service created: id=ooooahah12345` Lets look at this in ClearML. 

## Write down the Serving Service ID
## Clone demo repo repository. 

```shell
git clone https://github.com/mikewlange/aiorhuman-gpt 
```

## Edit the docker-compose-triton.yml file.
- clearml-serving/docker/docker-compose-triton.yml
- find the enviroment: ``CLEARML_EXTRA_PYTHON_PACKAGES`` and add the packages you need for your model. we'll add ours here. 
```yaml
CLEARML_EXTRA_PYTHON_PACKAGES: ${CLEARML_EXTRA_PYTHON_PACKAGES:-textstat empath torch transformers nltk openai datasets diffusers benepar spacy sentence_transformers optuna interpret markdown bs4}
```

## Edit the environment variables file 
- (`docker/example.env`) with your clearml-server credentials and Serving Service UID. For example, you should have something like

```python
  CLEARML_WEB_HOST="https://app.clear.ml"
  CLEARML_API_HOST="https://api.clear.ml"
  CLEARML_FILES_HOST="https://files.clear.ml"
  CLEARML_API_ACCESS_KEY="<access_key_here>"
  CLEARML_API_SECRET_KEY="<secret_key_here>"
  CLEARML_SERVING_TASK_ID="<serving_service_id_here>"
```

## Spin the clearml-serving containers 
- using docker-compose (or if running on Kubernetes use the helm chart) 
- We are deploying a Pytorch model. So we want to use NVIDIA Triton Inference https://developer.nvidia.com/triton-inference-server, made for gpu, but it will work on cpu dev machine (my laptop in this case). In production using k8 and help charts is the ay to go. https://github.com/allegroai/clearml-helm-charts 

```shell
cd docker && docker-compose --env-file example.env -f docker-compose-triton.yml up 
```

> **Notice**: Any model that registers with "Triton" engine, will run the pre/post processing code on the Inference service container, and the model inference itself will be executed on the Triton Engine container.

Let's review what we did. 

10. Explore ClearML
11. Exlore Docker. 