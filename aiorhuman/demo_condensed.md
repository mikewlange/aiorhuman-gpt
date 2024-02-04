## Introduction / Overview
- The Goal here is to build a classification model, and a API for use with a custom GPT while augmenting the response. 
- We will be using clearml for our ai ops and clearml-serving for our inference platform and API. 
- Design an API for use externally. specifically a custom GPT to analyse student essays. 
- About me: 

## 🧬 Human vs 🅰👁️ Essay Detection 
<img src="https://mikewlange.github.io/ai-or-human/images/ai_or_human_overview.png" alt="Alt Text"/>


## Product Demo. 
Launch the GPT and test. 

## Proof of Concept Review - Notebook
[Notebook](https://mikewlange.github.io/ai-or-human/ai-or-human-notebook.html)

My objective objective is to develop an effective classifier that can identify unique linguistic and structural patterns in human and llm-generated text and determin the source of the essay -Human or LLM. 

We aim to identify these distinct characteristics in AI-generated essays, serving as a practical introduction to AI production models. This approach involves creating an ensemble model that combines transformer models with an interpretable model, the Explainable Boosting Machine (EBM), striking a balance between effectiveness and interpretability.

## ClearML Overview 
- Quick reviw on clearml and clearml serving. 
- Check these out. Victor is awesome! He explains every nook and cranny of the software. https://www.youtube.com/@ClearML 

## Setup Prerequisites 

**ClearML**
- Follow the https://clear.ml/docs/latest/docs/getting_started/ds/ds_first_steps 
This includs sign up (it's free)
  - https://app.clear.ml/ sign up
  - Install ClearML [click here for instrucions](https://clear.ml/docs/latest/docs/getting_started/ds/ds_first_steps#install-clearml)
  - Connect ClearML SDK to the Server [instructions](https://clear.ml/docs/latest/docs/getting_started/ds/ds_first_steps#connect-clearml-sdk-to-the-server)

**Docker**
- Install Docker https://docs.docker.com/engine/install/

**ngrok**
- Create an ngrok account ngrok.com 

**Anaconda**
- Can't hurt. 

## Prepare and Train Model. 
1. clone the repo
```bash
git clone https://github.com/mikewlange/aiorhuman-gpt.git
```

2. Install requrements. 
```bash
pip install -r aiorhuman-gpt/aiorhuman_model/requirements.txt 
```

3. Setup Conda Env and Install (you can wing it..might be more fun to run into issues.
```bash
conda create -n "py39" python=3.9 ipython
conda activate py39
pip install textstat 
pip install empath 
pip install torch 
pip install transformers 
pip install nltk 
pip install openai 
pip install datasets 
pip install diffusers 
pip install benepar 
pip install spacy 
pip install sentence_transformers 
pip install optuna 
pip install interpret 
pip install markdown 
pip install bs4
pip install clearml
pip install clearml-serving
python3 -c 'import benepar; benepar.download(\"benepar_en3\")'
python3 -m spacy download en_core_web_lg
```

4. Open the Folder in VSCode or whaever you use for python and edit the file ../aiorhuman/bert_bilstm_demo.py 
5. Update the config values ``CLEARML_PROJECT_NAME`` and ``CLEARML_TASk_NAME`` with values of your choice. 
> You can use Talk and Project names to get ClearML Taslk data and whatnot, but I like using IDs if you're haphazzard with your naming conventions 
6. Train your model. We are going to use the default params. When the training is complete, it is will be uploaded to clearml to use for inference anywhere.

```bash
python ai-or-human-gpt/aiorhuman_model/bert_bilstm_model.py
```

**OR (more on this later)**

4. Upload existing model. but you'd have to do this after you create your service below - but yea. 

```shell
clearml-serving --id 57187db30bfa46f5876ea198f3e46ecb model upload 
--name "bert bilstm model" 
--project "serving examples" 
--framework "pytorch" 
--path bert_bilstm_model.pth
```
 
**Configuring your model code to utilise ClearML.** 
Here is a one shot prompt to use gpt 4.5 to convert your pytorch model code to a format that makes it better for use in clearml. Needless to say, it may not work in the first shot depending on the complexity of the model. And mine is specific, but ut worked for both the Bert and EBM models. 

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

**Create and Inference Servive Controller**
> check out [this tutorial](https://clear.ml/docs/latest/docs/clearml_serving/clearml_serving_tutorial) for more general setup. this is specific to this project. but not much. 

1.  Install clearml-serving CLI:

```shell
pip install clearml-serving #or into your virtualenv or conda env. conda active env
```

2.  Create the Serving Service Controller. 
```shell 
clearml-serving create --name "aiorhuman inference service"
```
-   The new serving service UID should be printed `New Serving Service created: id=ohohohahahah123456u` Lets look at this in ClearML. 

3.  Write down the Serving Service UID

**Deploy Dockers To Internets**

4. Add Extra Packages To install

- Go back to your project. Now we are going to edit the clearml-serving/docker/docker-compose-triton.yml file.
- find the enviroment: ``CLEARML_EXTRA_PYTHON_PACKAGES`` and add the packages you need for your model. we'll add ours here. 
```yaml
CLEARML_EXTRA_PYTHON_PACKAGES: ${CLEARML_EXTRA_PYTHON_PACKAGES:-textstat empath torch transformers nltk openai datasets diffusers benepar spacy sentence_transformers optuna interpret markdown bs4}
```

5.  Edit the Environment Variables file
- (`docker/example.env`) with your clearml-server credentials and Serving Service UID. For example, you should have something like

```python
  CLEARML_WEB_HOST="https://app.clear.ml"
  CLEARML_API_HOST="https://api.clear.ml"
  CLEARML_FILES_HOST="https://files.clear.ml"
  CLEARML_API_ACCESS_KEY="<access_key_here>"
  CLEARML_API_SECRET_KEY="<secret_key_here>"
  CLEARML_SERVING_TASK_ID="<serving_service_id_here>"
```

6. Spin the clearml-serving containers with docker-compose (or if running on Kubernetes use the helm chart) 
-- We are deploying a Pytorch model. So we will want NVIDIA Triton Inference https://developer.nvidia.com/triton-inference-server or Triton with gpu, but for our porposes cpu Triton is fine so we can test of my laptop. In production I use clearml-agents to train the models on colab gpu or my linux machine. 

```shell
cd docker && docker-compose --env-file example.env -f docker-compose-triton.yml up 
```

> YO! If you're not on a GPU, this will still work. However, you might see odd errors that make you nervous in the log. Go with deployment on an NVIDIA gpu

Let's review what we did. 

10. Explore ClearML
11. Exlore Docker.

> **Notice**: Any model that registers with "Triton" engine, will run the pre/post processing code on the Inference service container, and the model inference itself will be executed on the Triton Engine container.

### Setup API and Inference
1. Head back to your project and look at 'aiorhuman_model/preprocess.py' and talk about the setup. 

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
**Test the feature generation.** 

Lets make a test class for the features. I'm just running the features code. 

```python
import unittest
import sys
import logging
import pandas as pd
sys.path.append('.')

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
python aiorhuman_model/tests/features.py
```

## Deploy Inference and Orchistrate Model. 
Make sure you put in your proper -id and model-id from your clearml-server
```sh
clearml-serving --id 57187db30bfa46f5876ea198f3e46ecb model add \
--engine triton --endpoint "bert_infer" \
--preprocess "clearml-serving-human-or-llm-gpt/aiorhuman_model/preprocess.py" \
--model-id 6cca54290c5d426dbcc088201274656e \
--input-size 1 128 \
--input-name "input_ids" \
--input-type float32 \
--output-size -1 2 \
--output-name "output" \
--output-type float32 \
--tags "bert-infer-add"
```

**OR**

```sh
clearml-serving --id 57187db30bfa46f5876ea198f3e46ecb model auto-update 
--engine triton --endpoint "bert_infer" \
--preprocess "clearml-serving-human-or-llm-gpt/aiorhuman_model/preprocess.py" \ 
--model-id 6cca54290c5d426dbcc088201274656e \
--input-size 1 128 \
--input-name "input_ids" \
--input-type float32 \
--output-size -1 2 \
--output-name "output" \
--output-type float32 \
--tags "bert_infer-add" \
--max-versions 2 \
--tags "bert-infer-autoupdate"
```

**Have on Deck**
```sh
clearml-serving model remove -h --endpoint 'bert_infer'
```

## Wait 5 Min
really. don't get impatient when you do the above. it takes a hot minute to refresh. 

---
## Install ngrok if you have not! GPTs will not run localhost. 
- https://ngrok.com/ and create a tunnel to your local machine inference server. you'll need this to work with custom GPT actions 
```shell
ngrok config add-authtoken <TOKEN>
ngrok http http://localhost:8080
```
---

## Test your api. 

1. Update your test files with service ids 
```sh
python aiorhuman_model/tests/api.py
``` 
1. In code
```python
import requests
import json

def test_model(text, endpoint_url):
    # Prepare the request payload
    payload = json.dumps({
        "text": text
    })
    # Send a POST request to the model endpoint
    response = requests.post(endpoint_url, data=payload, headers={'Content-Type': 'application/json'})

    # Parse the response
    if response.status_code == 200:
        print("Response from model:", response.json())
    else:
        print("Failed to get response, status code:", response.status_code)

# Example usage
text_sample = "As the education landscape continues to evolve, the debate over the benefits of students attending school from home has become increasingly relevant."
model_endpoint_url = "https://4435-173-31-239-51.ngrok-free.app/serve/bert_infer"

test_model(text_sample, model_endpoint_url)
```

2. Curl 
```curl
curl -X POST "https://4435-173-31-239-51.ngrok-free.app/serve/bert_infer" \
-H "Content-Type: application/json" \
-d "{\"text\":\"This is a test essay. As the education landscape continues to evolve, the debate over the benefits of students attending school from home has become increasingly relevant.\"}"
```

3. Postman.
You can grab the collection from [here](clearml-serving-human-or-llm-gpt/aiorhuman_model/tests)

Coo. Moving on. 

## Design and Deploy GPT\

## GPTs. 

What a way to interact with an api. it could be the perfect test harness becuse it can generate pretty much any kind of test data in real time. 

In a nutshell, we're going to. You must complete all steps to avoid unneeded frunstration. 

0. API - done. 
1. Create a privacy policy. Gotta have it. 
2. Create our swagger 
3. Create our knowledge File. This is RAG. Retrieval-augmented generation (RAG) is a technique for enhancing the accuracy and reliability of generative AI models with facts fetched from external sources.
4. Our interaction with GPT to put it all together. This is bad ass. 
5. When it works, a little tear will roll down your face. trust me on this. I've watched some great tutorials, there are only a couple good ones, on you tube about this process, and we are all aware this is a bleding edge as it comes today. in this little world. 

and GO

## PP
put this on the interent somewhere. Or create another ngrok tunnel. Or use [pinggy](https://pinggy.io/) - easy and free. Or use the demo one here: https://mikewlange.github.io/human-or-llm-gpt/GPT/privacy_policy.html 
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Privacy Policy</title>
</head>
<body>
    <header>
        <h1>Privacy Policy</h1>
    </header>
    <section>
        <h2>1. Introduction</h2>
        <p>This is the privacy policy for our website. It explains how we collect, use, and protect your personal information when you use our services.</p>
    </section>
    <section>
        <h2>2. Information We Collect</h2>
        <p>We may collect the following types of information:</p>
        <ul>
            <li>Your name and contact information</li>
            <li>Information about your usage of our services</li>
            <li>Information about your favorite shampoos</li>
        </ul>
    </section>
    <section>
        <h2>3. How We Use Your Information</h2>
        <p>We use your information for the following purposes:</p>
        <ul>
            <li>To provide and improve our services</li>
            <li>To communicate with your friends about what you type in here</li>
        </ul>
    </section>
    <section>
        <h2>4. How We Protect Your Information</h2>
        <p>We take the security of your information as serious as companies like Equifax do.</p>
    </section>
    <section>
        <h2>5. Contact Us</h2>
        <p>If you have any questions or concerns about our privacy policy, please contact us at <a href="mailto:contact@example.com">contact@example.com</a>.</p>
    </section>
</body>
</html>
```

## Some Swagger. 

Don't overthink this part and write up a complex swagger. it'll be tougher to get going. Your knoledge file and how you converse with the assistant will guide how the return values are processed . raggidy ann. 

Don't try and make this more complex than it is. it's self explanitory and use the bare minimum for now. 

```yaml
openapi: 3.0.0
info:
  title: AI or LLM API
  version: 1.0.0
  description: Called a model to predict if the text was written by an LLM or a Human. Return analyses as well. 
    extract features.
servers:
  - url: https://4435-173-31-239-51.ngrok-free.app/serve
    description: Local development server
paths:
  /bert_infer:
    post:
      summary: Analyze text and generate predictions and features
      operationId: bert_infer
      requestBody:
        description: Text to be analyzed
        required: true
        content:
          application/json:
            schema:
              type: object
              properties:
                text:
                  type: string
                  description: Text content to analyze
              required:
                - text
            example:
              text: Sample text to analyze.
      responses:
        "200":
          description: Successful response with analysis results
          content:
            application/json:
              schema:
                type: object
                properties:
                  bert_predictions:
                    type: array
                    items:
                      type: integer
                    description: Predictions from the BERT-based model
                  features:
                    type: object
                    additionalProperties:
                      type: number
                    description: Extracted features from the text
              example:
                bert_predictions:
                  - 1
                features:
                  flesch_kincaid_grade: 8.2
                  semantic_density: 0.5
                  ...: null
        "400":
          description: Bad request when the input text is not provided
        "500":
          description: Internal server error for any unhandled exceptions
```

## Knoledge File. 
[here is is](https://mikewlange.github.io/human-or-llm-gpt/GPT/knoledge.md)

## Lets put it all together. 