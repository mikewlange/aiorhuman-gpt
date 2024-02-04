import torch
import torch.nn as nn
from transformers import (BertModel, BertTokenizer,)
from typing import Any, Optional, Callable, Union
import logging
import markdown
from bs4 import BeautifulSoup
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import joblib
from clearml import StorageManager
import pickle
import os
import traceback

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CFG:
    # Device configuration
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Training configuration
    DATA_ETL_STRATEGY = 1
    TRAINING_DATA_COUNT = 1000
    CLEARML_OFFLINE_MODE = False
    CLEARML_ON = False
    SCRATCH_PATH = 'scratch'
    ARTIFACTS_PATH = 'artifacts'
    ENSAMBLE_STRATEGY = 1
    KAGGLE_RUN = False
    SUBMISSION_RUN = True
    EXPLAIN_CODE = False
    
# clearml-serving --id a0f6246f53174a8c886ee61a9905127e model add --engine triton --endpoint "bert_infer" --preprocess "examples/model/preprocess.py" --model-id aaa4d84a9d53466287a17dff91da94b8 --input-size 1 128 --input-name "input_ids" --input-type float32 --output-size -1 2 --output-name "output" --output-type float32
# cd docker && docker-compose --env-file example.env -f docker-compose-triton.yml up
# curl -X POST "http://127.0.0.1:8080/serve/bert_infer" -H "Content-Type: application/json" -d '{"text":"Cars. Cars have been around since they became famous in the 1900s, when Henry Ford created and built the first ModelT. Cars have played a major role in our every day lives since then. But now, people are starting to question if limiting car usage would be a good thing. To me, limiting the use of cars might be a good thing to do.\n\nIn like matter of this, article, "In German Suburb, Life Goes On Without Cars," by Elizabeth Rosenthal states, how automobiles are the linchpin of suburbs, where middle class families from either Shanghai or Chicago tend to make their homes. Experts say how this is a huge impediment to current efforts to reduce greenhouse gas emissions from tailpipe. Passenger cars are responsible for 12 percent of greenhouse gas emissions in Europe...and up to 50 percent in some carintensive areas in the United States. Cars are the main reason for the greenhouse gas emissions because of a lot of people driving them around all the time getting where they need to go. Article, "Paris bans driving due to smog," by Robert Duffer says, how Paris, after days of nearrecord pollution, enforced a partial driving ban to clear the air of the global city. It also says, how on Monday, motorist with evennumbered license plates were ordered to leave their cars at home or be fined a 22euro fine 31. The same order would be applied to oddnumbered plates the following day. Cars are the reason for polluting entire cities like Paris. This shows how bad cars can be because, of all the pollution that they can cause to an entire city.\n\nLikewise, in the article, "Carfree day is spinning into a big hit in Bogota," by Andrew Selsky says, how programs that's set to spread to other countries, millions of Columbians hiked, biked, skated, or took the bus to work during a carfree day, leaving streets of this capital city eerily devoid of traffic jams. It was the third straight year cars have been banned with only buses and taxis permitted for the Day Without Cars in the capital city of 7 million. People like the idea of having carfree days because, it allows them to lesson the pollution that cars put out of their exhaust from people driving all the time. The article also tells how parks and sports centers have bustled throughout the city uneven, pitted sidewalks have been replaced by broad, smooth sidewalks rushhour restrictions have dramatically cut traffic and new restaurants and upscale shopping districts have cropped up. Having no cars has been good for the country of Columbia because, it has aloud them to repair things that have needed repairs for a long time, traffic jams have gone down, and restaurants and shopping districts have popped up, all due to the fact of having less cars around.\n\nIn conclusion, the use of less cars and having carfree days, have had a big impact on the environment of cities because, it is cutting down the air pollution that the cars have majorly polluted, it has aloud countries like Columbia to repair sidewalks, and cut down traffic jams. Limiting the use of cars would be a good thing for America. So we should limit the use of cars by maybe riding a bike, or maybe walking somewhere that isn't that far from you and doesn't need the use of a car to get you there. To me, limiting the use of cars might be a good thing to do. "}'
class BERTBiLSTMClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes, dropout_rate=0.1, lstm_hidden_size=128, lstm_layers=2):
        super(BERTBiLSTMClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.lstm = nn.LSTM(self.bert.config.hidden_size, lstm_hidden_size, lstm_layers, 
                            batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(lstm_hidden_size * 2, num_classes)  # *2 for bidirectional
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        lstm_output, (h_n, c_n) = self.lstm(sequence_output)
        pooled_output = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
        x = self.dropout(pooled_output)
        x = self.relu(x)
        x = self.fc(x)
        return x

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Preprocess(object):
    def __init__(self):
        self.bert_model = BERTBiLSTMClassifier('bert-base-uncased', 2)
        # Assuming the BERT model weights are already loaded
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_length = 128
        self.features = pd.DataFrame()
        self.ebm_predictions = None
        # Load the EBM model
        #self.ebm_model = self.load_ebm_model('68210179fd1143d09d5b5a2cb3ec9c4a')

    def load_ebm_model(self, model_id):

        model_path = StorageManager.get_local_copy(remote_url='https://files.clear.ml/Models%20-%20Text%20Classification/train%20ebm%20model.d5ab4a7111fe409ba5bc371ccd2bf051/models/ebm.pkl')
        # Load the model from the downloaded file
        with open(model_path, 'rb') as f:
            ebm_model = pickle.load(f)
        
        return ebm_model

    def preprocess(self, body: Union[bytes, dict], state: dict, collect_custom_statistics_fn: Optional[Callable[[dict], None]]) -> Any:
        try:
            logging.info(body)
            logging.info(f"Received body for preprocessing: {body}")

            self.features = self.generate_features(body['text'], is_inference=True)
            if self.features is None:
                logging.error("self.features is None after generate_features call")
            else:
                logging.info(f"Features generated successfully with shape: {self.features.shape}")

            cleaned_text = self.preprocess_text(body['text'])
            inputs = self.tokenizer(cleaned_text, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt")
            return inputs
        except Exception as e:
            logging.error(f"An error occurred: {traceback.print_exc()}")
            traceback.print_exc()

    def process(self, data: Any, state: dict, collect_custom_statistics_fn: Optional[Callable[[dict], None]]) -> Any:
        try:
            input_ids = data['input_ids'].to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            attention_mask = data['attention_mask'].to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            
            with torch.no_grad():
                bert_outputs = self.bert_model(input_ids, attention_mask)
            
        except Exception as e:
            logging.error(f"An error occurred: {traceback.print_exc()}")
            traceback.print_exc()
        
        # Use the EBM model to get predictions for the generated features - skip for the demo
        #self.ebm_predictions = self.ebm_model.predict_proba(self.features)
        
        return bert_outputs

    def postprocess(self, data: Any, state: dict, collect_custom_statistics_fn: Optional[Callable[[dict], None]]) -> dict:
        bert_predictions = torch.argmax(data, dim=1).tolist()

        #softmax_probabilities = torch.softmax(data, axis=1)

        return {'bert_predictions': bert_predictions, "features": self.features.to_dict()}
    

    def preprocess_text(self, text):
        try:
            # Remove markdown formatting
            html = markdown.markdown(text)
            text = BeautifulSoup(html, features="html.parser").get_text()
            # Replace newlines and remove extra whitespaces
            text = re.sub(r'[\n\r]+', ' ', text)
            text = ' '.join(text.split())
            # Remove 'Task' prefix from the prompt
            text = re.sub(r'^(?:Task(?:\s*\d+)?\.?\s*)?', '', text)
            text = re.sub('\n+', '', text)
            text = re.sub(r'[A-Z]+_[A-Z]+', '', text)
            # Remove punctuation except specified characters
            punctuation_to_remove = r'[^\w\s' + re.escape('.?!,') + ']'
            text = re.sub(punctuation_to_remove, '', text)
            # Tokenize and lemmatize text
            tokens = word_tokenize(text)
            lemmatizer = WordNetLemmatizer()
            lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]

            # Join tokens back to string
            return ' '.join(lemmatized_tokens)
        except Exception as e:
            logging.error(f"Error in preprocess_pipeline: {e}")
            return text

    '''Clean Data'''
    # Function to preprocess text
    def pipeline_etl_clean_data(self,df):
        import logging
        import markdown
        from bs4 import BeautifulSoup
        import re
        from nltk.tokenize import word_tokenize
        from nltk.stem import WordNetLemmatizer
        import nltk

        # Download necessary NLTK packages
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)
        # Ensure the necessary NLTK packages are downloaded
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('wordnet', quiet=True)
        except Exception as e:
            logging.error(f"An error occurred while downloading NLTK packages: {e}")

        # Function to remove markdown formatting
        def remove_markdown(text):
            try:
                html = markdown.markdown(text)
                soup = BeautifulSoup(html, features="html.parser")
                return soup.get_text()
            except Exception as e:
                logging.error(f"Error in remove_markdown: {e}")
                return text

        # Function to remove 'Task' prefix from the prompt
        def remove_task_on_prompt(text):
            try:
                pattern = r'^(?:Task(?:\s*\d+)?\.?\s*)?'
                return re.sub(pattern, '', text)
            except Exception as e:
                logging.error(f"Error in remove_task_on_prompt: {e}")
                return text

        # Function to replace newline and carriage return characters
        def replace_newlines(text):
            try:
                return re.sub(r'[\n\r]+', ' ', text)
            except Exception as e:
                logging.error(f"Error in replace_newlines: {e}")
                return text

        # Function to remove extra whitespaces
        def remove_extra_whitespace(text):
            try:
                return ' '.join(text.split())
            except Exception as e:
                logging.error(f"Error in remove_extra_whitespace: {e}")
                return text

        # Function to remove punctuation except for specified characters
        def remove_punctuation_except(text, punctuation_to_retain):
            try:
                punctuation_to_remove = r'[^\w\s' + re.escape(punctuation_to_retain) + ']'
                return re.sub(punctuation_to_remove, '', text)
            except Exception as e:
                logging.error(f"Error in remove_punctuation_except: {e}")
                return text

        def remove_emojis_and_newlines(text):
            # Regex pattern for matching emojis
            emoji_pattern = re.compile("["
                                u"\U0001F600-\U0001F64F"  # emoticons
                                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                u"\U00002500-\U00002BEF"  # chinese characters
                                u"\U00002702-\U000027B0"
                                u"\U00002702-\U000027B0"
                                u"\U000024C2-\U0001F251"
                                u"\U0001f926-\U0001f937"
                                u"\U00010000-\U0010FFFF"
                                u"\u2640-\u2642"
                                u"\u2600-\u2B55"
                                u"\u200d"
                                u"\u23cf"
                                u"\u23e9"
                                u"\u231a"
                                u"\ufe0f"  # dingbats
                                u"\u3030"
                                "]+", flags=re.UNICODE)

            # Remove newline characters
            text = re.sub('\n+', ' ', text)
            # Remove emojis
            text = emoji_pattern.sub(r'', text)
            return text
        
        def replace_newlines(text):
            return re.sub(r'[\r\n]+', ' ', text)
        # Function to tokenize and lemmatize text
        def process_text(text):
            try:
                tokens = word_tokenize(text)
                lemmatizer = WordNetLemmatizer()
                lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
                return lemmatized_tokens
            except Exception as e:
                logging.error(f"Error in process_text: {e}")
                return []
        # Main preprocessing logic
        try:
            PUNCTUATION_TO_RETAIN = '.?!,'  # Punctuation characters to retain    
            for index, row in df.iterrows():
                text = row['text']
                text = remove_markdown(text)
                text = replace_newlines(text)
                text = remove_extra_whitespace(text)
                text = remove_task_on_prompt(text)
                text = remove_punctuation_except(text, PUNCTUATION_TO_RETAIN)
                #text = remove_emojis_and_newlines(text)
                text = re.sub('\n+', '', text)
                text = re.sub(r'[A-Z]+_[A-Z]+', '', text)
                text = replace_newlines(text)
                # Remove occurrences of \n\n from the text
                # text = text.replace('\n\n', '')
                tokens = process_text(text)
                preprocessed_text = ' '.join(tokens)
                
                # Update the 'preprocessed_text' column with the processed text
                df.at[index, 'text'] = preprocessed_text
            df_essays = pd.DataFrame(df)
            return df_essays
        except Exception as e:
            logging.error(f"Error in preprocess_text: {e}")

    '''Readability Scores'''
    # =============================================================================
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    def apply_textstat_function(self,df, column_name, function_to_apply):
        import logging
        import textstat
        try:
            df.loc[:, column_name] = df.loc[:, 'text'].apply(function_to_apply)
            logging.info(f"Function {function_to_apply.__name__} applied to column {column_name}")
            return df
        except Exception as e:
            logging.error(e)

    # @PipelineDecorator.component(return_values=["df_readability_essays"], name='Readability Scores - Features Pipeline', 
    #                             cache=True, task_type=TaskTypes.data_processing)       
    def process_readability_scores(self,df_essays):
        import logging
        import textstat

        try:
            # Calculate readability scores
            print(df_essays.info())
            print(df_essays.shape)
            df_essays['flesch_kincaid_grade'] = df_essays['text'].apply(textstat.flesch_kincaid_grade)
            df_essays['gunning_fog'] = df_essays['text'].apply(textstat.gunning_fog)
            df_essays['coleman_liau_index'] = df_essays['text'].apply(textstat.coleman_liau_index)
            df_essays['smog_index'] = df_essays['text'].apply(textstat.smog_index)
            df_essays['ari'] = df_essays['text'].apply(textstat.automated_readability_index)
            df_essays['dale_chall'] = df_essays['text'].apply(textstat.dale_chall_readability_score)
            df_readability_essays = df_essays
            return df_readability_essays

        except Exception as e:
            logging.error(f"Error in process_readability_scores: {e}")
            raise


    '''Semantic Density'''
    # =============================================================================
    #@PipelineDecorator.component(return_values=["df_semantic_essays"], name='Semantic Density - Features Pipeline', 
    #                             cache=True, task_type=TaskTypes.data_processing)  
    def process_semantic_density(self,df_essays):
        import logging
        import numpy as np
        import pandas as pd
        import nltk
        from nltk.tokenize import word_tokenize
        import string
        from sentence_transformers import SentenceTransformer

        # Configure logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

        # Ensure that the necessary NLTK models are downloaded
        nltk.download('punkt', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        nltk.download('stopwords', quiet=True)

        def get_meaning_bearing_tags():
            return {'NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS'}

        def tokenize_text(text):
            try:
                return word_tokenize(text.lower())
            except TypeError as e:
                logging.error(f"Error tokenizing text: {e}")
                return []

        def tag_words(words):
            try:
                return nltk.pos_tag(words)
            except Exception as e:
                logging.error(f"Error tagging words: {e}")
                return []

        def filter_words(tokens):
            return [token for token in tokens if token.isalpha() or token in string.punctuation]

        mb_tags = get_meaning_bearing_tags()

        df_essays['semantic_density'] = 0
        df_essays['text_tagged_nltk'] = ""

        def process_row(row):
            index, data = row
            text = data['text']
            tokens = tokenize_text(text)
            words = filter_words(tokens)
            tagged = tag_words(words)
            mb_words = [word for word, tag in tagged if tag in mb_tags]
            full_sentence = " ".join(word + "/" + tag for word, tag in tagged)
            density = len(mb_words) / len(words) if words else 0
            data['semantic_density'] = density
            data['text_tagged_nltk'] = full_sentence
            return index, data

        processed_rows = map(process_row, df_essays.iterrows())

        df_semantic_essays = pd.DataFrame.from_dict(dict(processed_rows), orient='index')
        return df_semantic_essays

    '''Semantic Flow Variability'''
    # =============================================================================
    #@PipelineDecorator.component(return_values=["df_semantic_essays"], name='Semantic Flow Variability - Features Pipeline', 
    #                             cache=True, task_type=TaskTypes.data_processing)  
    def process_semantic_flow_variability(self,df):
        import logging
        import numpy as np
        import pandas as pd
        import nltk
        from sentence_transformers import SentenceTransformer
        import concurrent.futures
        # Configure logging
        """
        Process a DataFrame to calculate Semantic Flow Variability for each text entry.

        Semantic Flow Variability is calculated by measuring the cosine similarity between
        sentence embeddings of consecutive sentences in a text. It's a measure of how varied
        the semantic content is across the text.

        Args:
            df (pandas.DataFrame): DataFrame containing a 'text' column.

        Returns:
            pandas.DataFrame: The input DataFrame with an additional column 'semantic_flow_variability'.
        """

        # logging.basicConfig(level=logging.INFO,
        #                     format='%(asctime)s - %(levelname)s - %(message)s')
        # logger = logging.getLogger(__name__)

        # Load a pre-trained sentence transformer model
        model_MiniLM = 'sentence-transformers/all-MiniLM-L6-v2'

        try:
            model = SentenceTransformer(model_MiniLM)
        except Exception as e:
            logging.error(f"Error loading the sentence transformer model: {e}")
            model = None

        def cosine_similarity(v1, v2):
            return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

        def semantic_flow_variability(text):
            if not model:
                logging.error(
                    "Model not loaded. Cannot compute semantic flow variability.")
                return np.nan

            try:
                # Split the text into sentences
                sentences = nltk.sent_tokenize(text)
                if len(sentences) < 2:
                    logging.info(
                        "Not enough sentences for variability calculation.")
                    return 0

                # Generate embeddings for each sentence
                sentence_embeddings = model.encode(
                    sentences, convert_to_tensor=True, show_progress_bar=False)

                # Move embeddings to CPU and convert to numpy - this is necessary for the next step
                sentence_embeddings = sentence_embeddings.cpu().numpy()

                # Calculate cosine similarity between consecutive sentences
                similarities = [cosine_similarity(sentence_embeddings[i], sentence_embeddings[i+1])
                                for i in range(len(sentence_embeddings)-1)]

                # Return the standard deviation of the similarities as a measure of variability
                return np.std(similarities)
            except Exception as e:
                logging.error(f"Error calculating semantic flow variability: {e}")
                return np.nan

        if df is not None and 'text' in df:
            # Use concurrent processing for parallel execution
            with concurrent.futures.ThreadPoolExecutor() as executor:
                df['semantic_flow_variability'] = list(
                    executor.map(semantic_flow_variability, df['text']))
        else:
            logging.error("Invalid DataFrame or missing 'text' column.")

        df_semantic_essays = df
        return df_semantic_essays

 
    '''Psycholinguistic Features'''
    # =============================================================================
    #@PipelineDecorator.component(return_values=["df_psyco_essays"], name='Psycholinguistic Features - Features Pipeline',
    #                             cache=True, task_type=TaskTypes.data_processing)
    def apply_empath_analysis(self,df, text_column='text'):
        import pandas as pd
        import logging
        from empath import Empath

        # Initialize logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        """
        Apply Empath analysis to a DataFrame column, expanding results into separate columns.

        Empath analysis interprets the text for various emotional and thematic elements,
        returning a dictionary of categories and their respective scores. This function
        applies the analysis to a specified column of a DataFrame and expands the results
        into separate columns.

        Args:
            df (pandas.DataFrame): The DataFrame to analyze.
            text_column (str): The name of the column containing text to analyze.

        Returns:
            pandas.DataFrame: The original DataFrame with added columns for Empath analysis results.
        """
        lexicon = Empath()

        def empath_analysis(text):
            try:
                return lexicon.analyze(text, normalize=True)
            except Exception as e:
                logger.error(f"Error during Empath analysis: {e}")
                return {}

        try:
            df['empath_analysis'] = df['text'].apply(empath_analysis)
            empath_columns = df['empath_analysis'].apply(pd.Series)
            df = pd.concat([df, empath_columns], axis=1)
            df.drop(columns=['empath_analysis'], inplace=True)
            
            return df
        except Exception as e:
            # Log an error message if an exception occurs
            logger.error(f"Error applying Empath analysis to DataFrame: {e}")
            # Return the original DataFrame to avoid data loss
            return df
        

    '''Textrual Entropy'''
    # =============================================================================
    #@PipelineDecorator.component(return_values=["df_essays"], name='Textual Entropy - Features Pipeline',
    #                             cache=True, task_type=TaskTypes.data_processing)
    def process_textual_entropy(self,df):
        import numpy as np
        from collections import Counter
        import logging
        import pandas as pd
        
        # Configure logging
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s - %(levelname)s - %(message)s")
        logger = logging.getLogger(__name__)
        """
        Calculate the Shannon entropy of a text string.

        Entropy is calculated by first determining the frequency distribution
        of the characters in the text, and then using these frequencies to 
        calculate the probabilities of each character. The Shannon entropy 
        is the negative sum of the product of probabilities and their log2 values.

        Args:
            text (str): The text string to calculate entropy for.

        Returns:
            float: The calculated entropy of the text, or 0 if text is empty/non-string.
            None: In case of an exception during calculation.
        """
        
        def calc_entropy(text):
            freq_dist = Counter(text)
            probs = [freq / len(text) for freq in freq_dist.values()]
            # Calculate entropy, avoiding log2(0)
            entropy = -sum(p * np.log2(p) for p in probs if p > 0)
            return entropy
        try:
            if not isinstance(df, pd.DataFrame):
                logger.warning("Input is not a DataFrame.")
                return None

            # Loop through each row and apply the function to 'text' column
            
            df["textual_entropy"] = df["text"].apply(calc_entropy)
            df_entropy = df
            return df_entropy
        except Exception as e:
            logger.error(f"Error calculating entropy: {e}")
            return None
    '''Syntactic Tree Patterns'''
    # =============================================================================
    # Configure logging

    #@PipelineDecorator.component(return_values=["df_essays"], name='Syntactic Tree Patterns - Features Pipeline',
    #                             cache=True, task_type=TaskTypes.data_processing)
    def process_syntactic_tree_patterns(self,df_essays):
        """
        Process a DataFrame containing essays to extract various syntactic tree pattern features.

        The function uses spaCy, benepar, and NLTK to analyze syntactic structures of text,
        calculating various metrics such as tree depth, branching factors, nodes, leaves,
        and production rules. It also includes text analysis features like token length,
        sentence length, and entity analysis.

        Args:
            df_essays (pandas.DataFrame): DataFrame containing a 'text' column with essays.

        Returns:
            pandas.DataFrame: DataFrame with additional columns for each extracted syntactic and textual feature.
        """
        import spacy
        import benepar
        import numpy as np
        import pandas as pd
        import logging
        from collections import Counter
        from nltk import Tree
        from transformers import T5TokenizerFast
        from tqdm import tqdm
        tqdm.pandas()
        import time
        # Start time
        #start_time = time.time()
        # Configure logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logger = logging.getLogger(__name__)
        import traceback

        start_time = time.time()
        """
        Process a DataFrame containing essays to extract various syntactic tree pattern features.

        The function uses spaCy, benepar, and NLTK to analyze syntactic structures of text,
        calculating various metrics such as tree depth, branching factors, nodes, leaves,
        and production rules. It also includes text analysis features like token length,
        sentence length, and entity analysis.

        Args:
            df_essays (pandas.DataFrame): DataFrame containing a 'text' column with essays.

        Returns:
            pandas.DataFrame: DataFrame with additional columns for each extracted syntactic and textual feature.
        """
        tokenizer = T5TokenizerFast.from_pretrained('t5-base', model_max_length=512, validate_args=False)
        try:
            nlp = spacy.load('en_core_web_lg')
            if spacy.__version__.startswith('2'):
                benepar.download('benepar_en3')
                nlp.add_pipe(benepar.BeneparComponent("benepar_en3"))
            else:
                nlp.add_pipe("benepar", config={"model": "benepar_en3"})
        except Exception as e:
            logger.error(f"Failed to load spaCy model: {e}")
            return df_essays

        # Define helper functions for tree analysis...
        # (include spacy_to_nltk_tree, tree_depth, tree_branching_factor, count_nodes, count_leaves, etc.)
        def spacy_to_nltk_tree(node):
            if node.n_lefts + node.n_rights > 0:
                return Tree(node.orth_, [spacy_to_nltk_tree(child) for child in node.children])
            else:
                return node.orth_

        def tree_depth(node):
            if not isinstance(node, Tree):
                return 0
            else:
                return 1 + max(tree_depth(child) for child in node)

        def tree_branching_factor(node):
            if not isinstance(node, Tree):
                return 0
            else:
                return len(node)

        def count_nodes(node):
            if not isinstance(node, Tree):
                return 1
            else:
                return 1 + sum(count_nodes(child) for child in node)

        def count_leaves(node):
            if not isinstance(node, Tree):
                return 1
            else:
                return sum(count_leaves(child) for child in node)

        def production_rules(node):
            rules = []
            if isinstance(node, Tree):
                rules.append(node.label())
                for child in node:
                    rules.extend(production_rules(child))
            return rules

        def count_labels_in_tree(tree, label):
            if not isinstance(tree, Tree):
                return 0
            count = 1 if tree.label() == label else 0
            for subtree in tree:
                count += count_labels_in_tree(subtree, label)
            return count

        def count_phrases_by_label(trees, label, doc):
            if label == 'NP':
                noun_phrases = [chunk.text for chunk in doc.noun_chunks]
                return noun_phrases
            else:
                return sum(count_labels_in_tree(tree, label) for tree in trees if isinstance(tree, Tree))

        def count_subtrees_by_label(trees, label):
            return sum(count_labels_in_tree(tree, label) for tree in trees if isinstance(tree, Tree))

        def average_phrase_length(trees):
            lengths = [len(tree.leaves()) for tree in trees if isinstance(tree, Tree)]
            return np.mean(lengths) if lengths else 0

        def subtree_height(tree, side):
            if not isinstance(tree, Tree) or not tree:
                return 0
            if side == 'left':
                return 1 + subtree_height(tree[0], side)
            elif side == 'right':
                return 1 + subtree_height(tree[-1], side)

        def average_subtree_height(trees):
            heights = [tree_depth(tree) for tree in trees if isinstance(tree, Tree)]
            return np.mean(heights) if heights else 0

        def pos_tag_distribution(trees):
            pos_tags = [tag for tree in trees for word, tag in tree.pos()]
            return Counter(pos_tags)

        def process_tree_or_string(obj):
            if isinstance(obj, Tree):
                return obj.height()
            else:
                return None

        def syntactic_ngrams(tree):
            ngrams = []
            if isinstance(tree, Tree):
                ngrams.extend(list(nltk.ngrams(tree.pos(), 2)))
            return ngrams
        
        # Process each essay and extract features
        for index, row in df_essays.iterrows():
            text = row['text']
            try:
                doc = nlp(text)
                trees = [spacy_to_nltk_tree(sent.root) for sent in doc.sents if len(tokenizer.tokenize(sent.text)) < 512]
                trees = [tree for tree in trees if isinstance(tree, Tree)]

                # Extract features
                depths = [tree_depth(tree) for tree in trees if isinstance(tree, Tree)]
                branching_factors = [tree_branching_factor(tree) for tree in trees if isinstance(tree, Tree)]
                nodes = [count_nodes(tree) for tree in trees if isinstance(tree, Tree)]
                leaves = [count_leaves(tree) for tree in trees if isinstance(tree, Tree)]
                rules = [production_rules(tree) for tree in trees if isinstance(tree, Tree)]
                rule_counts = Counter([rule for sublist in rules for rule in sublist])

                # Text analysis features
                num_sentences = len(list(doc.sents))
                num_tokens = len(doc)
                unique_lemmas = set([token.lemma_ for token in doc])
                total_token_length = sum(len(token.text) for token in doc)
                average_token_length = total_token_length / num_tokens if num_tokens > 0 else 0
                average_sentence_length = num_tokens / num_sentences if num_sentences > 0 else 0
                num_entities = len(doc.ents)
                num_noun_chunks = len(list(doc.noun_chunks))
                pos_tags = [token.pos_ for token in doc]
                num_pos_tags = len(set(pos_tags))
                distinct_entities = set([ent.text for ent in doc.ents])
                total_entity_length = sum(len(ent.text) for ent in doc.ents)
                average_entity_length = total_entity_length / num_entities if num_entities > 0 else 0
                total_noun_chunk_length = sum(len(chunk.text) for chunk in doc.noun_chunks)
                average_noun_chunk_length = total_noun_chunk_length / num_noun_chunks if num_noun_chunks > 0 else 0
                ngrams = []
                for tree in trees:
                    ngrams.extend(syntactic_ngrams(tree))

                # Assign calculated feature values to the DataFrame
                # Assign calculated feature values to the DataFrame
                df_essays.at[index, 'num_sentences'] = num_sentences
                df_essays.at[index, 'num_tokens'] = num_tokens
                df_essays.at[index, 'num_unique_lemmas'] = len(unique_lemmas)
                df_essays.at[index, 'average_token_length'] = average_token_length
                df_essays.at[index, 'average_sentence_length'] = average_sentence_length
                df_essays.at[index, 'num_entities'] = num_entities
                df_essays.at[index, 'num_noun_chunks'] = num_noun_chunks
                df_essays.at[index, 'num_pos_tags'] = num_pos_tags
                df_essays.at[index, 'num_distinct_entities'] = len(distinct_entities)
                df_essays.at[index, 'average_entity_length'] = average_entity_length
                df_essays.at[index, 'average_noun_chunk_length'] = average_noun_chunk_length
                df_essays.at[index, 'max_depth'] = max(depths) if depths else 0
                df_essays.at[index, 'avg_branching_factor'] = np.mean(branching_factors) if branching_factors else 0
                df_essays.at[index, 'total_nodes'] = sum(nodes)
                df_essays.at[index, 'total_leaves'] = sum(leaves)
                df_essays.at[index, 'unique_rules'] = len(rule_counts)
                df_essays.at[index, 'most_common_rule'] = rule_counts.most_common(1)[0][0] if rule_counts else None
                df_essays.at[index, 'tree_complexity'] = sum(nodes) / sum(leaves) if leaves else 0
                df_essays.at[index, 'depth_variability'] = np.std(depths)
                #df_essays.at[index, 'subtree_freq_dist'] = Counter([' '.join(node.leaves()) for tree in trees for node in tree.subtrees() if isinstance(node, Tree)])
                df_essays.at[index, 'tree_height_variability'] = np.std([subtree_height(tree, 'left') for tree in trees if isinstance(tree, Tree)])
                
                #df_essays.at[index, 'pos_tag_dist'] = pos_tag_distribution(trees)
                #df_essays.at[index, 'syntactic_ngrams'] = ngrams

            except Exception as e:
                logger.error(f"Error processing text: {e}")
                traceback.print_exc()
                # Assign NaNs in case of error
                #df_essays.at[index, 'num_sentences'] = 0
                # ... Assign NaNs for other features ...

        return df_essays

    def scale_columns(self,df, columns_to_scale, scaler=None, scale_type='MinMaxScaler',is_inference=False):
        """
        Scale the specified columns in a DataFrame and add a suffix to the column names.

        Args:
            df (pandas.DataFrame): The DataFrame to scale.
            columns_to_scale (list): List of column names to scale.
            scaler (object, optional): Scaler object to use for scaling. If None, a new scaler object will be created.
            scale_type (str, optional): The type of scaler to use. Default is 'MinMaxScaler'. Options: 'MinMaxScaler', 'StandardScaler'.

        Returns:
            pandas.DataFrame: The full DataFrame with scaled columns added.
            pandas.DataFrame: A separate DataFrame with only the specified columns scaled.
            object: The scaler object used for scaling.
        """
        
        print("is_inference : ",is_inference)
        if scale_type == 'MinMaxScaler':
            scaler = MinMaxScaler() if scaler is None else scaler
        elif scale_type == 'StandardScaler':
            scaler = StandardScaler() if scaler is None else scaler
        else:
            raise ValueError("Invalid scale_type. Options: 'MinMaxScaler', 'StandardScaler'")

        if(is_inference == True): # train
            #raise ValueError("Scaler object is None.")
            scaled_columns = scaler.transform(df[columns_to_scale])
        else:
            scaled_columns = scaler.fit_transform(df[columns_to_scale])
            
        scaled_df = pd.DataFrame(scaled_columns, columns=[col + '_scaled' for col in columns_to_scale])

        full_df = pd.concat([df.drop(columns=columns_to_scale), scaled_df], axis=1)

        return full_df, scaled_df, scaler
       
    def generate_features_for_inference(self,df_essays,is_inference):
        try:
            #df_essays_copy = pd.read_pickle("scratch/df_essays_copy.pkl")

            ## Run them through the pipeline to get the features
            df_essays = self.pipeline_etl_clean_data(df_essays)

            '''Readability Scores'''
            # =============================================================================
            df_readability_essays = self.process_readability_scores(df_essays)

            '''Semantic Density'''
            # =============================================================================
            df_semantic_essays = self.process_semantic_density(df_readability_essays)

            '''Semantic Flow Variability'''
            # =============================================================================
            df_variability_essays = self.process_semantic_flow_variability(df_semantic_essays)

            '''Psycholuigustic Features'''
            # =============================================================================
            df_psyco_essays = self.apply_empath_analysis(df_variability_essays)

            '''Textrual Entropy'''
        # =============================================================================
            df_entropy = self.process_textual_entropy(df_psyco_essays)

            '''Syntactic Tree Patterns'''
            # =============================================================================
            df_essays = self.process_syntactic_tree_patterns(df_entropy)
            
            return df_essays[['flesch_kincaid_grade', 'gunning_fog', 'coleman_liau_index', 'smog_index', 'ari', 'dale_chall', 'textual_entropy', 'semantic_density', 'semantic_flow_variability','num_sentences', 'num_tokens', 'num_unique_lemmas', 'average_token_length', 'average_sentence_length', 'num_entities', 'num_noun_chunks', 'num_pos_tags', 'num_distinct_entities', 'average_entity_length', 'average_noun_chunk_length', 'max_depth', 'avg_branching_factor', 'total_nodes', 'total_leaves', 'unique_rules', 'tree_complexity', 'depth_variability','help','office','dance','money','wedding','domestic_work','sleep','medical_emergency','cold','hate','cheerfulness','aggression','occupation','envy','anticipation','family','vacation','crime','attractive','masculine','prison','health','pride','dispute','nervousness','government','weakness','horror','swearing_terms','leisure','suffering','royalty','wealthy','tourism','furniture','school','magic','beach','journalism','morning','banking','social_media','exercise','night','kill','blue_collar_job','art','ridicule','play','computer','college','optimism','stealing','real_estate','home','divine','sexual','fear','irritability','superhero','business','driving','pet','childish','cooking','exasperation','religion','hipster','internet','surprise','reading','worship','leader','independence','movement','body','noise','eating','medieval','zest','confusion','water','sports','death','healing','legend','heroic','celebration','restaurant','violence','programming','dominant_heirarchical','military','neglect','swimming','exotic','love','hiking','communication','hearing','order','sympathy','hygiene','weather','anonymity','trust','ancient','deception','fabric','air_travel','fight','dominant_personality','music','vehicle','politeness','toy','farming','meeting','war','speaking','listen','urban','shopping','disgust','fire','tool','phone','gain','sound','injury','sailing','rage','science','work','appearance','valuable','warmth','youth','sadness','fun','emotional','joy','affection','traveling','fashion','ugliness','lust','shame','torment','economics','anger','politics','ship','clothing','car','strength','technology','breaking','shape_and_size','power','white_collar_job','animal','party','terrorism','smell','disappointment','poor','plant','pain','beauty','timidity','philosophy','negotiate','negative_emotion','cleaning','messaging','competing','law','friends','payment','achievement','alcohol','liquid','feminine','weapon','children','monster','ocean','giving','contentment','writing','rural','positive_emotion','musical','num_sentences', 'num_tokens', 'num_unique_lemmas', 'average_token_length', 'average_sentence_length', 'num_entities', 'num_noun_chunks', 'num_pos_tags', 'num_distinct_entities', 'average_entity_length', 'average_noun_chunk_length', 'max_depth', 'avg_branching_factor', 'total_nodes', 'total_leaves', 'unique_rules', 'tree_complexity', 'depth_variability']]
            
            readability_columns = ['flesch_kincaid_grade', 'gunning_fog', 'coleman_liau_index', 'smog_index', 'ari', 'dale_chall', 'textual_entropy', 'semantic_density', 'semantic_flow_variability']
            #readability_columns = ['num_sentences', 'num_tokens', 'num_unique_lemmas', 'average_token_length', 'average_sentence_length', 'num_entities', 'num_noun_chunks', 'num_pos_tags', 'num_distinct_entities', 'average_entity_length', 'average_noun_chunk_length', 'max_depth', 'avg_branching_factor', 'total_nodes', 'total_leaves', 'unique_rules', 'tree_complexity', 'depth_variability','help','office','dance','money','wedding','domestic_work','sleep','medical_emergency','cold','hate','cheerfulness','aggression','occupation','envy','anticipation','family','vacation','crime','attractive','masculine','prison','health','pride','dispute','nervousness','government','weakness','horror','swearing_terms','leisure','suffering','royalty','wealthy','tourism','furniture','school','magic','beach','journalism','morning','banking','social_media','exercise','night','kill','blue_collar_job','art','ridicule','play','computer','college','optimism','stealing','real_estate','home','divine','sexual','fear','irritability','superhero','business','driving','pet','childish','cooking','exasperation','religion','hipster','internet','surprise','reading','worship','leader','independence','movement','body','noise','eating','medieval','zest','confusion','water','sports','death','healing','legend','heroic','celebration','restaurant','violence','programming','dominant_heirarchical','military','neglect','swimming','exotic','love','hiking','communication','hearing','order','sympathy','hygiene','weather','anonymity','trust','ancient','deception','fabric','air_travel','fight','dominant_personality','music','vehicle','politeness','toy','farming','meeting','war','speaking','listen','urban','shopping','disgust','fire','tool','phone','gain','sound','injury','sailing','rage','science','work','appearance','valuable','warmth','youth','sadness','fun','emotional','joy','affection','traveling','fashion','ugliness','lust','shame','torment','economics','anger','politics','ship','clothing','car','strength','technology','breaking','shape_and_size','power','white_collar_job','animal','party','terrorism','smell','disappointment','poor','plant','pain','beauty','timidity','philosophy','negotiate','negative_emotion','cleaning','messaging','competing','law','friends','payment','achievement','alcohol','liquid','feminine','weapon','children','monster','ocean','giving','contentment','writing','rural','positive_emotion','musical','num_sentences', 'num_tokens', 'num_unique_lemmas', 'average_token_length', 'average_sentence_length', 'num_entities', 'num_noun_chunks', 'num_pos_tags', 'num_distinct_entities', 'average_entity_length', 'average_noun_chunk_length', 'max_depth', 'avg_branching_factor', 'total_nodes', 'total_leaves', 'unique_rules', 'tree_complexity', 'depth_variability']
            if os.path.exists("scaler_semantic_features.pkl"):
                scaler = joblib.load("scaler_semantic_features.pkl")
            else:
                # Create a new StandardScaler object
                scaler = MinMaxScaler()
                # Save the scaler object for future use
                
                            
            # Scale the columns using MinMaxScaler
            readability_scaled_backin_df, readability_scaled_df, readability_scaler = self.scale_columns(df_essays, readability_columns,scaler, 
                                                                                                         scale_type='MinMaxScaler',is_inference=is_inference)
            joblib.dump(readability_scaler, "scaler_semantic_features.pkl")

            psycho_columns = ['help','office','dance','money','wedding','domestic_work','sleep','medical_emergency','cold','hate','cheerfulness','aggression','occupation','envy','anticipation','family','vacation','crime','attractive','masculine','prison','health','pride','dispute','nervousness','government','weakness','horror','swearing_terms','leisure','suffering','royalty','wealthy','tourism','furniture','school','magic','beach','journalism','morning','banking','social_media','exercise','night','kill','blue_collar_job','art','ridicule','play','computer','college','optimism','stealing','real_estate','home','divine','sexual','fear','irritability','superhero','business','driving','pet','childish','cooking','exasperation','religion','hipster','internet','surprise','reading','worship','leader','independence','movement','body','noise','eating','medieval','zest','confusion','water','sports','death','healing','legend','heroic','celebration','restaurant','violence','programming','dominant_heirarchical','military','neglect','swimming','exotic','love','hiking','communication','hearing','order','sympathy','hygiene','weather','anonymity','trust','ancient','deception','fabric','air_travel','fight','dominant_personality','music','vehicle','politeness','toy','farming','meeting','war','speaking','listen','urban','shopping','disgust','fire','tool','phone','gain','sound','injury','sailing','rage','science','work','appearance','valuable','warmth','youth','sadness','fun','emotional','joy','affection','traveling','fashion','ugliness','lust','shame','torment','economics','anger','politics','ship','clothing','car','strength','technology','breaking','shape_and_size','power','white_collar_job','animal','party','terrorism','smell','disappointment','poor','plant','pain','beauty','timidity','philosophy','negotiate','negative_emotion','cleaning','messaging','competing','law','friends','payment','achievement','alcohol','liquid','feminine','weapon','children','monster','ocean','giving','contentment','writing','rural','positive_emotion','musical']
            if os.path.exists("scaler_psycho_features.pkl"):
                scaler_psyco = joblib.load("scaler_psycho_features.pkl")
            else:
                # Create a new StandardScaler object
                scaler_psyco = MinMaxScaler()
                # Save the scaler object for future use

            psycho_scaled_df_backin_df, psycho_scaled_df, psycho_scaler = self.scale_columns(df_essays, psycho_columns,scaler_psyco, scale_type='MinMaxScaler',is_inference=is_inference)
            joblib.dump(psycho_scaler, "scaler_psycho_features.pkl")

            # Define the columns to scale
            text_features = ['num_sentences', 'num_tokens', 'num_unique_lemmas', 'average_token_length', 'average_sentence_length', 'num_entities', 'num_noun_chunks', 'num_pos_tags', 'num_distinct_entities', 'average_entity_length', 'average_noun_chunk_length', 'max_depth', 'avg_branching_factor', 'total_nodes', 'total_leaves', 'unique_rules', 'tree_complexity', 'depth_variability']
            
            if os.path.exists("scaler_tree_features.pkl"):
                scaler_text = joblib.load("scaler_tree_features.pkl")
            else:
                # Create a new StandardScaler object
                scaler_text = MinMaxScaler()
                # Save the scaler object for future use
            # Scale the columns using MinMaxScaler
            tree_feature_scaler_backin_df, tree_features_scaled_df, tree_feature_scaler = self.scale_columns(df_essays, 
                                                                                                        text_features,scaler_text, scale_type='MinMaxScaler',is_inference=is_inference)
            joblib.dump(tree_feature_scaler, "scaler_tree_features.pkl")
            final_features_df = pd.concat([readability_scaled_df,tree_features_scaled_df,psycho_scaled_df], axis=1)

            return df_essays[['num_sentences', 'num_tokens', 'num_unique_lemmas', 'average_token_length', 'average_sentence_length', 'num_entities', 'num_noun_chunks', 'num_pos_tags', 'num_distinct_entities', 'average_entity_length', 'average_noun_chunk_length', 'max_depth', 'avg_branching_factor', 'total_nodes', 'total_leaves', 'unique_rules', 'tree_complexity', 'depth_variability','help','office','dance','money','wedding','domestic_work','sleep','medical_emergency','cold','hate','cheerfulness','aggression','occupation','envy','anticipation','family','vacation','crime','attractive','masculine','prison','health','pride','dispute','nervousness','government','weakness','horror','swearing_terms','leisure','suffering','royalty','wealthy','tourism','furniture','school','magic','beach','journalism','morning','banking','social_media','exercise','night','kill','blue_collar_job','art','ridicule','play','computer','college','optimism','stealing','real_estate','home','divine','sexual','fear','irritability','superhero','business','driving','pet','childish','cooking','exasperation','religion','hipster','internet','surprise','reading','worship','leader','independence','movement','body','noise','eating','medieval','zest','confusion','water','sports','death','healing','legend','heroic','celebration','restaurant','violence','programming','dominant_heirarchical','military','neglect','swimming','exotic','love','hiking','communication','hearing','order','sympathy','hygiene','weather','anonymity','trust','ancient','deception','fabric','air_travel','fight','dominant_personality','music','vehicle','politeness','toy','farming','meeting','war','speaking','listen','urban','shopping','disgust','fire','tool','phone','gain','sound','injury','sailing','rage','science','work','appearance','valuable','warmth','youth','sadness','fun','emotional','joy','affection','traveling','fashion','ugliness','lust','shame','torment','economics','anger','politics','ship','clothing','car','strength','technology','breaking','shape_and_size','power','white_collar_job','animal','party','terrorism','smell','disappointment','poor','plant','pain','beauty','timidity','philosophy','negotiate','negative_emotion','cleaning','messaging','competing','law','friends','payment','achievement','alcohol','liquid','feminine','weapon','children','monster','ocean','giving','contentment','writing','rural','positive_emotion','musical','num_sentences', 'num_tokens', 'num_unique_lemmas', 'average_token_length', 'average_sentence_length', 'num_entities', 'num_noun_chunks', 'num_pos_tags', 'num_distinct_entities', 'average_entity_length', 'average_noun_chunk_length', 'max_depth', 'avg_branching_factor', 'total_nodes', 'total_leaves', 'unique_rules', 'tree_complexity', 'depth_variability']]
            

        except Exception as e:
            # if any fail, revert to the bert inference
            print(f"Error in feature extraction: {e}")
            
    def generate_features(self, text, is_inference=False):
        #print(text)
        # Create a DataFrame from the string
        if(isinstance(text,str)):
            df = pd.DataFrame({'text': [text]})
            print(df)
        else:
            df = text

        df = self.generate_features_for_inference(df,is_inference)
        print("process completed")
        return df

