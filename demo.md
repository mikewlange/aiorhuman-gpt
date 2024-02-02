## 🧬 Human vs 🅰👁️ Essay Detection 
<img src="https://mikewlange.github.io/ai-or-human/images/ai_or_human_overview.png" alt="Alt Text"/>

## First 
- clearml and clearml-serving set up and install. 
- Install and setup
    - Setup your [**ClearML Server**](https://github.com/allegroai/clearml-server) or use the [Free tier Hosting](https://app.clear.ml)
    - Setup local access (if you haven't already), see instructions [here](https://clear.ml/docs/latest/docs/getting_started/ds/ds_first_steps#install-clearml)
    - install docker
    - setup a CONDA environemnt for the project, if you want. 

### [Video Starts]
## Introduction / Overview
- The Goal here is to build a classification model, and a API for use with a custom GPT while augmenting the response. 
- We will be using clearml for our ai ops and clearml-serving for our inference platform and API. 
- Design an API for use externally. specifically a custom GPT to analyse student essays. 

## About me
- 

## Product Demo. 
Launch the GPT and test. 

## Proof of Concept Notebook
**Hypothesis**: *Certain linguistic and structural patterns unique to AI-generated text can be identified and used for classification. We anticipate that our analysis will reveal distinct characteristics in AI-generated essays, enabling us to develop an effective classifier for this purpose.*

## Motivations
- **Learning and Challenge**: Enhancing knowledge in Natural Language Processing (NLP) and staying intellectually active between jobs.
- **Tool Development**: Potential creation of a tool to differentiate between human and AI-generated content, useful across various fields.
- **Educational Value**: Serves as a practical introduction to production models in AI with integration into 3rd party application. for this we're doing a GPT.

## Objective
- **Model Development**: Building an ensemble model that combines two transformer models with an interpretable model like the Explainable Boosting Machine 
- **Challenges**: Balancing effectiveness with interpretability.
- **Approach**:
  - Developing an Explainable Boosting Machine (EBM) that uses custom features to compliment BERT's performance with better interpretability.

## Clear ML
talk about clear ml a bit
Key concept for this demo.

Here is the architecture of ClearML-Server 
<img src="https://mikewlange.github.io/ai-or-human/images/clearml_ops_image.webp" width="800px" alt="Alt Text" >


and ClearML-Serving. 
<img src="https://github.com/allegroai/clearml-serving/raw/main/docs/design_diagram.png?raw=true" width="800px" alt="Alt Text" >
ClearML - our AIML Ops tool. 

ClearML-Sering

- CLI - Secure configuration interface for on-line model upgrade/deployment on running Serving Services

- Serving Service Task - Control plane object storing configuration on all the endpoints. Support multiple separated instance, deployed on multiple clusters.

- Inference Services - Inference containers, performing model serving pre/post processing. Also support CPU model inferencing.

- Serving Engine Services - Inference engine containers (e.g. Nvidia Triton, TorchServe etc.) used by the Inference Services for heavier model inference. I use nvidia triton. we convert the EBM model to ONYX and run through TIS. 

- Statistics Service - Single instance per Serving Service collecting and broadcasting model serving & performance statistics 

- Time-series DB - Statistics collection service used by the Statistics Service, e.g. Prometheus

- Dashboards - Customizable dashboard-ing solution on top of the collected statistics, e.g. Grafana

## Training Data
An in depth review is out of scope here. However, check it out. The data generation is for the POC only. 

What I did here was take standardised essay prompts and genererate essays using LLMS. 

> **LAST 10%**: For the prod model I use https://github.com/microsoft/promptbench to generate primpts that generate essays that fool the existing model GAN. It sounds complex, but you're basically creating a mad-lib out of your prompt and let the engines fill in the blanks. This way we can generate data from most all public LLMs. So I would design a prompt that can fool my BiLSTM-BERT Transformer classificaiton model. You cant just say fool it. Other than the mad-lib part of it there are things to consider like clarity and inherent properties of an essay. And since one of our models trains on custom features we create, those are also model weights that cen be used. 


# 🧮 **Model Development**
<img src="https://mikewlange.github.io/ai-or-human/images/aiorhuman_develop_model.drawio.png" alt="Alt Text" />

1. **BertForSequenceClassification**:
   - **Architecture**: BERT (Bidirectional Encoder Representations from Transformers) for sequence classification. [4]

2. **BertModel + BiLSTM**:
   - **Architecture**: The model is composed of the BertModel layer followed by BiLSTM layers. This is further connected to a dropout layer for regularization, a fully connected linear layer with ReLU activation, and a final linear layer for classification.

3. **Explainable Boosting Machine (EBM) for Feature Classification**:

   - **Type**: Glass-box model, notable for interpretability and effectiveness.
   - **Function**: Classifies based on extracted features from the essays.
   - **Configuration**: Includes settings for interaction depth, learning rate, and validation size.
   - **Insights**: Provides understanding of feature importance and model behavior.
   - **Causality** This is a 'casual' model and the EBM is for helping dertermine causality aloong with our feature stats

## Features

<img src="https://mikewlange.github.io/ai-or-human/images/aiorhuman_engineer_features.drawio.png" alt="Alt Text"/>

<div style="display: inline-block;padding: 10px;background-color: #f4f3ee;border: 1px solid #FF1493;border-radius: 4px;margin-bottom: 10px;line-height: 1.5;color: #333;" class="alert"> For <b>Feature Engineering</b>, we focus on extracting attributes from the essay text data. In these features, we target capturing nuanced differences in textual characteristics, such as readability, semantic density, and syntactic patterns, distinguishing between AI-generated and human-written texts.</div>

## Key Analytical Areas
1. **Readability Scores**:
   - Identifying unique patterns in AI vs. human-written essays.
   - Analysis using scores like `Flesch-Kincaid Grade Level`, `Gunning Fog Index`, etc.

2. **Semantic Density**:
   - Understanding the concentration of meaning-bearing words in AI-generated vs. human text.

3. **Semantic Flow Variability**:
   - Examining idea transitions between sentences in human and AI-generated texts.

4. **Psycholinguistic Features**:
   - Using the LIWC tool for psychological and emotional content evaluation.

5. **Textual Entropy**:
   - Measuring unpredictability or randomness, focusing on differences between AI and human content.

6. **Syntactic Tree Patterns**:
   - Parsing essays to analyze syntactic tree patterns, especially structural tendencies in language models.

 **📊 Feature Distribution Statistics**

## Understanding Key Statistical Concepts

<div style="display: inline-block;padding: 10px;background-color: #f4f3ee;border: 1px solid #FF1493;border-radius: 4px;margin-bottom: 10px;line-height: 1.5;color: #333;" class="alert"><b>Is there a statistacally signifigant difference in feature x's distribution betwen LLM and Human?</b> Each of these measures provides a different perspective on the data, with the <b>p-values</b> offering insights into statistical significance and the effect size measures (Cohen's d and Glass's delta) providing information about the magnitude of the differences observed.</div>


1. **T-Test p-value**:
    - **Purpose**: Determines if differences between groups are statistically significant.
    - **Interpretation**: A low p-value (< 0.05) suggests significant differences, challenging the null hypothesis.

2. **Mann-Whitney U p-value**:
    - **Usage**: Ideal for non-normally distributed data, comparing two independent samples.
    - **Significance**: Similar to the T-test, a lower p-value indicates notable differences between the groups.

3. **Kruskal-Wallis p-value**:
    - **Application**: Used for comparing more than two independent samples.
    - **Meaning**: A low p-value implies significant variance in at least one of the samples from the others.

4. **Cohen's d**:
    - **Function**: Measures the standardized difference between two means.
    - **Values**: Interpreted as small (0.2), medium (0.5), or large (0.8) effects.

5. **Glass's delta**:
    - **Comparison with Cohen's d**: Similar in purpose but uses only the standard deviation of one group for normalization.
    - **Utility**: Effective when the groups' standard deviations differ significantly.

## Note on Sample Size and Statistical Tests
- **Small Samples (Under 5000 Records)**: T-Test, Mann-Whitney U, and Kruskal-Wallis tests are effective.
- **Large Samples (Over 5000 Records)**: Focus on **effect sizes** (Cohen's d and Glass's delta), as p-values will generally approach 0.

## Moving On
**Now were going to get into the code. Starting with**
- model training and orchistration. 
- build our inference API 
- Test and integrate with a custom GPT. This part is a doosey. Along with others on youtube I found this not to be a trivial task. 
   - yaml for swagger def
   - knoledge file for RAG
   - Privacy Policy 
   - ngrok so you can test locally. 

## Lets Get to Code. 
1.  Setup your [**ClearML Server**](https://github.com/allegroai/clearml-server) or use the [Free tier Hosting](https://app.clear.ml)
2.  Setup local access (if you haven't already), see instructions [here](https://clear.ml/docs/latest/docs/getting_started/ds/ds_first_steps#install-clearml)

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


## Converting your Model. 
Our infrastructure is now watiting for code :)

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

[paste your model along with it]

## Model Code. 
We're going to trust it, since i've run it before, and let it deploy my model to my clearml-server ready for inference and orchistration. 

1. open up the model code and take a lok at this
If we look at this code here
```python
# Script and save the model using the example inputs
model_path = f'{CFG.SCRATCH_PATH}/clearml-serving/clearml_serving/bert_bilstm_model.pth'
traced_model = torch.jit.trace(model, example_inputs)
traced_model.save(model_path)

# Log the scripted model in ClearML (if needed)
output_model = OutputModel(task=task)
output_model.update_weights(model_path)
```
2. We can see that we are saving the model and then logging it to ClearML. This runs and you have a model. Move on. It's a good idea to create tests. Manual adjustments may need to be made. Update the configs to point to your training files. 

```bash
pip install -r ai-or-human-gpt/aiorhuman_model/requirements.txt 
```

3. Now you can train your model. We are going to use the default params. 
```bash
python ai-or-human-gpt/aiorhuman_model/bert_bilstm_model.py
```

> LAST 10%: this is where you would setup and use a clearml-agent to run these remotely. 
> LAST 10%: train all three models so we can build multi-model inference. 

4. Check out the model in clearml

### Our API

1.  lets looks at preprocess.py. this is what clearml-serving engine is looking for when it deployes our model. the api is going to get data, we just have to tell our code how to perform inference. 

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
        self.model.load_state_dict(torch.load('bert_bilstm_model.pth'))  # Load your trained model weights -  this is not for inference. it's to have our BERTBiLSTMClassifier obj in here. When we deploy, this is a self contained resource, the rest is abstracted away so we don't want to import file objects. I think? I'm implimenting this in clearml for the first time too.  
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

    # ... other preprocessing methods ...

```

API is programmed to return the Softmax probability of it's forcast. we are not going to run inference here on the EBM, for simplicitys sake. It inproves berts score by about 5 percenteged. Not that it's more accurate, but differently avvurate avereaging a better score. And since the scores are so close, I can use the features that are generated to forcast the EBM and return those as the analysis part of the forcast. margin of error of 5% is acceptable givin the complexeties of scoring two models to get almost the same forcast. You see where I'm going. Trade offs to reduce complexety must always be looked at. 

Lets review the full preprocessing code. 

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

## Deploy 
if everything above is setup and tested. this deployes it and prepare the model for inference. 
```shell
clearml-serving --id 57187db30bfa46f5876ea198f3e46ecb model add \
--engine triton --endpoint "bert_infer" \
--preprocess "clearml-serving-human-or-llm-gpt/aiorhuman_model/preprocess.py" \
--model-id 7f64ef1ff35544a7858bbb6dcd04c6ee \
--input-size 1 128 \
--input-name "input_ids" \
--input-type float32 \
--output-size -1 2 \
--output-name "output" \
--output-type float32 
```

Review above. There is an auto update script. But running the abive deploys any changes to your preprocess code.

> **last 10%**: to push this into the clouds I use Kubernets and helm charts. The charts I uses are all [here](https://github.com/allegroai/clearml-helm-charts/tree/main/charts/clearml-serving). 
 

## Lets test it locally. 

- i like to do curl and postman tests. for complex apis I like to create in postman. I built a Flask harness for this, but it's an unnessecary step. 
```curl
curl -X POST "http://127.0.0.1:8080/serve/bert_infer" \
-H "Content-Type: application/json" \
-d "{\"text\":\"As the education landscape continues to evolve, the debate over the benefits of students attending school from home has become increasingly relevant. This essay will delve into the multifaceted advantages of remote learning, particularly for students with anxiety or depression, the impact of drama and rumors on student performance in a traditional school setting, and the potential advantages of distance learning on the student body as a whole. By examining these aspects, we aim to develop a strong argument supporting the idea of students attending school from home.\n\nFor students grappling with anxiety or depression, the traditional school environment can be overwhelming and exacerbate their mental health challenges. In a study published in the Journal of Medical Internet Research, it was found that remote learning provided a less stressful and more flexible environment for students dealing with mental health issues. Attending school from home allows these students to create a personalized, comfortable space where they can focus on their studies without the added pressure of social interactions or the fear of judgment from their peers. By minimizing the triggers that often come with a traditional school setting, remote learning offers a valuable opportunity for these students to manage their mental health and concentrate on their academic pursuits.\n\nMoreover, the impact of drama and rumors on student performance in a traditional school setting cannot be overlooked. The social dynamics and peer interactions in a physical school can sometimes lead to the proliferation of rumors and drama, which can significantly impact a student's emotional well-being and academic focus. In contrast, attending school from home provides a shield from these negative influences, allowing students to remain focused on their studies without being distracted or distressed by external factors. This isolation from detrimental social dynamics can lead to a more positive and productive learning environment for students, fostering their academic growth and emotional stability.\n\nFurthermore, the potential advantages of distance learning on the student body as a whole extend beyond individual well-being. Remote learning promotes inclusivity by accommodating students with diverse needs and circumstances, such as those with physical disabilities, chronic illnesses, or those balancing familial responsibilities. A study by the National Education Association highlighted that distance learning facilitates greater equity and access to education for students who may face barriers in a traditional school setting. By embracing remote learning, educational institutions can create an environment that caters to the varied needs of their student body, fostering an inclusive and supportive learning community.\n\nConsidering these aspects, it is evident that students attending school from home can reap a multitude of benefits. This alternative mode of learning offers a supportive and conducive environment for students with anxiety or depression, shields them from the detrimental impact of drama and rumors, and promotes inclusivity and equity within the student body. As such, it is crucial for educational institutions to recognize the potential advantages of remote learning and consider its implementation as a means of enhancing the overall well-being and academic success of their students.\n\nIn conclusion, the benefits of students attending school from home are substantial and wide-ranging. From supporting students with anxiety or depression to mitigating the impact of social dynamics on academic performance and fostering inclusivity, remote learning offers a promising avenue for educational advancement. By prioritizing the well-being and educational needs of students, embracing remote learning can pave the way for a more holistic and supportive approach to education. Therefore, it is imperative for educators and policymakers to consider the potential advantages of remote learning and seriously contemplate its integration into the educational framework.\"}"
```

## Install ngrok 
- https://ngrok.com/ and create a tunnel to your local machine inference server. you'll need this to work with custom GPT actions 
```shell
ngrok config add-authtoken <TOKEN>
ngrok http http://localhost:8080
```

## Our Inference URL
https://4435-173-31-239-51.ngrok-free.app/serve/bert_infer

Test your ngrok url and we're ready to start the reason we are all here, the GPT. it makes it all worth while. 

## GPTs. 
I've never coded like this. If you've never really seen how what we used to call chatbots, were really meant to do. It's a whole new way to input instructions to a machine. it's qrite facinating. 

In a nutshell, we're going to. You must complete all steps to avoid unneeded frunstration. 

1. Create a privacy policy. Gotta have it. 
2. Create our swagger 
3. Create our Knoledge File. This is RAG. Retrieval-augmented generation (RAG) is a technique for enhancing the accuracy and reliability of generative AI models with facts fetched from external sources.
4. Our interaction with GPT to put it all together. This is bad ass. 
5. When it works, a little tear will roll down your face. trust me on this. I've watched some great tutorials, there are only a couple good ones, on you tube about this process, and we are all aware this is a bleding edge as it comes today. in this little world. 

and GO

## PP
put this on the interent somewhere. Or create another ngrok tunnel. Or use [pinggy](https://pinggy.io/) - easy and free. 
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

Don't overthink this part and write up a complex swagger. it'll be tougher to get going. Your knolege file and how you converse with the assistant will dictate your return post processing. raddidy ann. 

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
[here is is]('clearml-serving-human-or-llm-gpt/GPT/knoledge.md')

## Lets put it all together. 

Lauch chat gpt and create a new GPT. 