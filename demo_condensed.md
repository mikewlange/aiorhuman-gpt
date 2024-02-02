## Introduction / Overview
- The Goal here is to build a classification model, and a API for use with a custom GPT while augmenting the response. 
- We will be using clearml for our ai ops and clearml-serving for our inference platform and API. 
- Design an API for use externally. specifically a custom GPT to analyse student essays. 
- About me: 

## 🧬 Human vs 🅰👁️ Essay Detection 
<img src="https://mikewlange.github.io/ai-or-human/images/ai_or_human_overview.png" alt="Alt Text"/>

## Prereqs 
### Setup ClearML https://clear.ml/  
#### ClearML  
---- 
1.  Setup your [**ClearML Server**](https://github.com/allegroai/clearml-server) or use the [Free tier Hosting](https://app.clear.ml)
2.  Setup local access (if you haven't already), see instructions [here](https://clear.ml/docs/latest/docs/getting_started/ds/ds_first_steps#install-clearml) 
----

#### Other
3. Install Docker https://docs.docker.com/engine/install/
4. Create an ngrok account ngrok.com 
5. Download training data
    - LLM generated - https://www.kaggle.com/datasets/geraltrivia/llm-detect-gpt354-generated-and-rewritten-essays
    - other Human and LLM - https://www.kaggle.com/datasets/thedrcat/daigt-v2-train-dataset 


## Product Demo. 
Launch the GPT and test. 

## Proof of Concept Review - Notebook
My objective objective is to develop an effective classifier that can identify unique linguistic and structural patterns in human and llm-generated text and determin the source of the essay -Human or LLM. 

We aim to identify these distinct characteristics in AI-generated essays, serving as a practical introduction to AI production models. This approach involves creating an ensemble model that combines transformer models with an interpretable model, the Explainable Boosting Machine (EBM), striking a balance between effectiveness and interpretability.

## ClearML Overview 
- Quick reviw on clearml and clearml serving. 

## Prepare and Train Model. 
1. clone the repo
```bash
git clone https://github.com/mikewlange/human-or-llm-gpt.git
```

2. Install requrements. 
```bash
pip install -r ai-or-human-gpt/aiorhuman_model/requirements.txt 
```

3. Train your model. We are going to use the default params. When the training is complete, it is will be uploaded to clearml to use for inference anywhere.

```bash
python ai-or-human-gpt/aiorhuman_model/bert_bilstm_model.py
```

#### Converting your Model. 
Here is a one shot prompt to use gpt 4.5 to convert to pytorch model code to a format that makes it better for use in clearml. It's good to do this from the get go, but just in case you're comverting other model code and what not. 

```markdown 
- Review the original PyTorch, TensorFlow, or Keras training script to identify all hardcoded hyperparameters, configurations, and key components like model definition, data loading, training loop, and evaluation function.

- Replace all hardcoded hyperparameters and configurations with argparse arguments. For each parameter:
    - Add an argparse argument in the script's main function or a dedicated argument parsing function.
    - Ensure each argument has a descriptive name, default value, and help description.
   - Replace the usage of hardcoded values in the script with these argparse arguments.

- Integrate ClearML by:
    - Initializing a ClearML Task at the beginning of your script. Use `Task.init(project_name='Your Project Name', task_name='Your Task Name')`.
    - Connect the argparse arguments to ClearML with `task.connect(vars(args))` after parsing the arguments.
    - Throughout the script, use ClearML's logging functions to log metrics, models, and artifacts as needed, especially in training and evaluation sections.
    - Leave NO todo's, placeholders or missing pieces

- Adapt the script for deployment by:
    - Ensuring the model is in evaluation mode and using an example input batch to trace/script the model for a production environment.
    - Saving the deployment-ready model using `torch.jit.trace` or `torch.jit.script` and `save` method.
    - Using ClearML's `OutputModel` to log the model in ClearML for deployment.

- Test the modified script to ensure it runs successfully from end to end, argparse correctly parses and applies all hyperparameters, ClearML logs all necessary information, and the model is prepared and uploaded for deployment.

- Validate the script by running it with different sets of arguments to ensure flexibility and robustness of the parameterization.

- Remember to handle exceptions and edge cases, especially in data loading and model training sections, to ensure the script is robust and error-tolerant.
```

### Deploy Clearml-Serving Infrastructure. 

#### ClearML  
---- 
1.  Setup your [**ClearML Server**](https://github.com/allegroai/clearml-server) or use the [Free tier Hosting](https://app.clear.ml)
2.  Setup local access (if you haven't already), see instructions [here](https://clear.ml/docs/latest/docs/getting_started/ds/ds_first_steps#install-clearml) 
----

#### Create and Inference Servive Controller
> check out [this tutorial](https://clear.ml/docs/latest/docs/clearml_serving/clearml_serving_tutorial) for more general setup. this is specific to this project. but not much. 

3.  Install clearml-serving CLI:

```shell
pip install clearml-serving #or into your virtualenv or conda env. conda active env
```

4.  Create the Serving Service Controller. 
```shell 
clearml-serving create --name "aiorhuman inference service"`
```
-   The new serving service UID should be printed `New Serving Service created: id=ooooahah12345` Lets look at this in ClearML. 

5.  Write down the Serving Service UID

#### Deploy Dockers To Internets
6.  Clone clearml-serving repository. 

```shell
git clone https://github.com/allegroai/clearml-serving.git 
```

7. Now we are going to edit the clearml-serving/docker/docker-compose-triton.yml file.
- find the enviroment: ``CLEARML_EXTRA_PYTHON_PACKAGES`` and add the packages you need for your model. we'll add ours here. 
```yaml
CLEARML_EXTRA_PYTHON_PACKAGES: ${CLEARML_EXTRA_PYTHON_PACKAGES:-textstat empath torch transformers nltk openai datasets diffusers benepar spacy sentence_transformers optuna interpret markdown bs4}
```

8.  Edit the environment variables file (`docker/example.env`) with your clearml-server credentials and Serving Service UID. For example, you should have something like

```python
  CLEARML_WEB_HOST="https://app.clear.ml"
  CLEARML_API_HOST="https://api.clear.ml"
  CLEARML_FILES_HOST="https://files.clear.ml"
  CLEARML_API_ACCESS_KEY="<access_key_here>"
  CLEARML_API_SECRET_KEY="<secret_key_here>"
  CLEARML_SERVING_TASK_ID="<serving_service_id_here>"
```

9. Spin the clearml-serving containers with docker-compose (or if running on Kubernetes use the helm chart) 
-- We are deploying a Pytorch model. So we will want NVIDIA Triton Inference https://developer.nvidia.com/triton-inference-server or Triton with gpu, but for our porposes cpu Triton is fine so we can test of my laptop. In production I use clearml-agents to train the models on colab gpu or my linux machine. 

```shell
cd docker && docker-compose --env-file example.env -f docker-compose-triton.yml up 
```

> **Notice**: Any model that registers with "Triton" engine, will run the pre/post processing code on the Inference service container, and the model inference itself will be executed on the Triton Engine container.

Let's review what we did. 

10. Explore ClearML
11. Exlore Docker. 

## Setup API and Inference

```python
class Preprocess(object):
    def __init__(self):
        self.model = BERTBiLSTMClassifier(
            bert_model_name='bert-base-uncased',
            num_classes=2,
            dropout_rate=0.1,
            lstm_hidden_size=128,
            lstm_layers=2
        )
        self.model.load_state_dict(torch.load('bert_bilstm_model.pth'))  # Load your trained model weights - this is pulled from clearml artifact storage (anywhere, aws, drive)
        self.model.eval()  # Set the model to evaluation mode
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_length = 128

    def preprocess(self, body: Union[bytes, dict], state: dict, collect_custom_statistics_fn: Optional[Callable[[dict], None]]) -> Any:
        cleaned_text = self.clean_text(body['text'])
        inputs = self.tokenizer(cleaned_text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt")
        return inputs

    def process(self, data: Any, state: dict, collect_custom_statistics_fn: Optional[Callable[[dict], None]]) -> Any:
        input_ids = data['input_ids'].to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        attention_mask = data['attention_mask'].to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
        return outputs

    def postprocess(self, data: Any, state: dict, collect_custom_statistics_fn: Optional[Callable[[dict], None]]) -> dict:
        predictions = torch.argmax(data, dim=1).tolist()
        return {'predictions': predictions}

    # .. all our other features

```
## Test the feature generation. 

Lets make a test class for the features. I'm just running the features code. 

```python
import unittest
import sys
import logging
import pandas as pd
sys.path.append('examples/model')

from preprocess import Preprocess

# Set up logging
logging.basicConfig(level=logging.INFO)

class PreprocessTest(unittest.TestCase):

    def setUp(self):
        # Initialize the Preprocess object before each test
        self.preprocess = Preprocess()

    def test_generate_features(self):
        # Test the generate_features_from_mygpt function to ensure it returns a DataFrame
        input_text = "This is a test essay about writing code that writes essays that are about writing essays"
        output_df = self.preprocess.generate_features(input_text,is_inference=True)
        pd.set_option('display.float_format', '{:.6f}'.format)
        print(output_df)
        logging.info(f'Output DataFrame: \n{output_df.to_string()}')

if __name__ == '__main__':
    unittest.main()
```

Run 
```bash
python test_preprocess.py
```

## Orchistrate Model

## Design and Deploy GPT