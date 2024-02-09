
<div align="center">

### This is a fork of [ClearML Serving](https://github.com/allegroai/clearml-serving)

<a href="https://app.clear.ml"><img src="https://github.com/allegroai/clearml/blob/master/docs/clearml-logo.svg?raw=true" width="250px"></a>

**ClearML Serving - Model deployment made easy**
</div>

> &#10039; This project does not touch the code in the clearml_serving directory. However, due to its awesomeness and tight integration with this demo, it was pertinant to bring along for the ride. Enjoy!


<div align="center">

## Inspired by:
<img src="https://www.kaggle.com/static/images/site-logo.svg" width='200px' alt="Alt Text"/>

#### LLM - Detect AI Generated Text
*Identify which essay was written by a large language model*
</div>

<br>

<div align="center">

# 🅰👁️ | 🧬  

<img src="presentation/images/ai_human_logo.png" width='150px' alt="Meow"/>
</div>


## Architecture
<img src="presentation/images/ai_or_human_overview.drawio.png" alt="Alt Text"/>

## Original Concept
[Check it out here](https://mikewlange.github.io/aiorhuman-gpt/aiorhuman/ai-or-human-notebook.html)

## Watch and Code Along.
[![Watch the video](presentation/images/cover_image.png)](https://youtu.be/fv-MYQ5fVNc)
### [Mike Lange Description]
<img align="left" width="30" src="presentation/images/me.png" /> 
Um. Ok. Well. This is a step by step guide to building and deploying your own Custom GPT and API with End to End ClearML integration. Fun stuff.

### [GPT 4 Description]
<img align="left" width="30" src="presentation/images/gpt.png" /> 

Embark on a comprehensive journey to create, deploy, and leverage your very own Custom GPT and API, fully integrated with ClearML for a seamless experience. 

This step-by-step tutorial takes you through the entire process, starting with the construction of a robust PyTorch model that combines the prowess of a ``BERT/BiLSTM`` with the interpretability of an ``Explainable Boosting Machine`` (EBM) from InterpretML.

Discover how to bring your API to life using the power of ``HuggingFace``, ``PyTorch``, ``ClearML``, ``ClearML-Serving``, ``Docker``, ``NVIDIA Triton``, and **MORE** ensuring your model is not just a marvel of machine learning but also a fully operational service. Dive into the world of custom GPT actions, designing a system that communicates effectively with your API, and explore the innovative concept of ``Retrieval Augmented Generation`` (RAG) to enhance your GPT's responses with real-time, llm based enhanced explainability.

Whether you're a seasoned data scientist or an enthusiastic beginner &#9996;, this guide promises to equip you with the knowledge and tools needed to bring your AI visions to reality.

Wow, right? I hope it's that exciting. I hope I do all that. 

&#9996; That last part may be off. This is not easy stuff. You will work at getting it all going if you wish to recreate it. But just dive into the areas that are new to you, and kick its ass.  


### &#9851; &#8594; [Setup Your Environment](presentation/1.Setup.md) 

### 🏗 &#8594; [Deploy Infrastructure](presentation/2.Deploy_Infrastructure.md) 

### 🚄 &#8594; [Train and Publish Model ](presentation/3.Train_Publish_Model.md) 

### 😻 &#8594; [Build API](presentation/4.Build_Deploy_API.md) 

### 🪠 &#8594; [Tie it all together with a Custom GPT](presentation/5.Build_GPT.md)

### 😂 &#8594; [Have a Laugh](https://www.lifehack.org/articles/lifestyle/30-ways-add-fun-your-daily-routine.html)

#### 🤟 If your organization is looking for someone with my skillset, I am available as of 2-12-2024. Send me a message or check out my resume at [http://www.mikelange.com](mikelange.com) I look forward to chatting!
 
## Technology Thank You! 
- &#8594; Hugging Face: https://huggingface.co/  
- &#8594; Interpret ML: https://interpret.ml/ 
- &#8594; Apache https://www.apache.org/ 
- &#8594; ClearML: https://clear.ml/ 
- &#8594; Thanks ``Victor`` For the awesome Vids!  
- &#8594; Kaggle: https://Kaggle.com
- &#8594; PyTorch: https://PyTorch.org  
- &#8594; Tensorflow: https://www.tensorflow.org/
- &#8594; OpenAI: https://openai.com   
- &#8594; Docker: https://docker.com   
- &#8594; NVIDIA: https://nvidia.com  
- &#8594; Optuna: https://optuna.org/
- &#8594; Anaconda: https://anaconda.org/  
- &#8594; Spacy: https://spacy.io/
- &#8594; Berkley Neural Parser: https://spacy.io/universe/project/self-attentive-parser
- &#8594; scikit-learn https://scikit-learn.org
- &#8594; BERT: https://huggingface.co/docs/transformers/en/model_doc/bert
- &#8594; Empath: https://github.com/Ejhfast/empath-client
- &#8594; textstat: https://pypi.org/project/textstat/
- and all the others I didn't mention.

TODOS: 
- [ ] Improve the models. Help needed. 
- [ ] Clean up code. 
- [ ] ClearML Dashboards
- [ ] ClearML - Improve Integration
- [ ] Automate Training
- [ ] Improved metrics with training and inference 
- [ ] Grafana dashboards 
- [ ] Configure helm charts for k8 deployment 
- [ ] Speed up feature gen pipeline 
- [ ] Build EBM to ONYX pipeline to serve full ensemble on Triton (https://github.com/interpretml/ebm2onnx)
- [ ] ``Mad-Lib`` the training data gen prompts to do GAN optimizations. (explain this process better)
- [ ] Greatly reduce knoledge file
- [ ] impliment https://github.com/interpretml/TalkToEBM and 
- [ ] https://github.com/interpretml/interpret-text to expand explability. 

References: 

