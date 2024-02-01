import pandas as pd
import pickle
from interpret.glassbox import ExplainableBoostingClassifier
from interpret import show
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from interpret.perf import ROC, PR
from clearml import Task, OutputModel
from preprocess import Preprocess
import logging
import markdown
from bs4 import BeautifulSoup
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd
import logging
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import joblib


# Initialize ClearML Task
#task = Task.init(project_name='Your Project Name', task_name='Your EBM Task Name', task_type=Task.TaskTypes.training)
task = Task.init(project_name='Models - Text Classification', task_name='train ebm model', output_uri=True)
# Configuration (consider moving sensitive info to external config file or environment variables)
CFG = {
    'SCRATCH_PATH' : 'scratch',
    'CLEARML_ON': True,
}

# Connect your model_config to ClearML

# PREPROCESSING FUNCTIONS -- START
'''Clean Data'''
    # Function to preprocess text
def preprocess_text(text):
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
'''GENERATE FEATURES'''
# =============================================================================
# =============================================================================

'''Readability Scores'''
# =============================================================================
# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def apply_textstat_function(df, column_name, function_to_apply):
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
def process_readability_scores(df_essays):
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
def process_semantic_density(df_essays):
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
def process_semantic_flow_variability(df):
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
def apply_empath_analysis(df, text_column='text'):
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
def process_textual_entropy(df):
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
def process_syntactic_tree_patterns(df_essays):
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
                # df_essays.at[index, 'num_sentences'] = np.nan
                # ... Assign NaNs for other features ...

        return df_essays


def scale_columns(df, columns_to_scale, scaler=None, scale_type='MinMaxScaler'):
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
    if scale_type == 'MinMaxScaler':
        scaler = MinMaxScaler() if scaler is None else scaler
    elif scale_type == 'StandardScaler':
        scaler = StandardScaler() if scaler is None else scaler
    else:
        raise ValueError("Invalid scale_type. Options: 'MinMaxScaler', 'StandardScaler'")

    scaled_columns = scaler.fit_transform(df[columns_to_scale])
    scaled_df = pd.DataFrame(scaled_columns, columns=[col + '_scaled' for col in columns_to_scale])

    full_df = pd.concat([df.drop(columns=columns_to_scale), scaled_df], axis=1)

    return full_df, scaled_df, scaler

def scale_readability_scored(df_essays):
    import logging
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler, StandardScaler

    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # Define columns to scale
    columns_to_scale = ['flesch_kincaid_grade', 'gunning_fog', 'coleman_liau_index', 'smog_index', 'ari', 'dale_chall', 'semantic_density', 'semantic_flow_variability', 'textual_entropy', 'num_sentences', 'num_tokens', 'num_unique_lemmas', 'average_token_length', 'average_sentence_length', 'num_entities', 'num_noun_chunks', 'num_pos_tags', 'num_distinct_entities', 'average_entity_length', 'average_noun_chunk_length', 'max_depth', 'avg_branching_factor', 'total_nodes', 'total_leaves', 'unique_rules', 'tree_complexity', 'depth_variability', 'tree_height_variability']

    # Scale the columns
    try:
        df_essays, scaled_df, readability_scaler = scale_columns(df_essays, columns_to_scale)
        joblib.dump(readability_scaler, 'scaler_semantic_features.pkl', compress=True)
        return df_essays
    except Exception as e:
        logger.error(f"Error scaling columns: {e}")
        return df_essays
    
def scale_psycho_features(df_essays):
    import logging
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler, StandardScaler

    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # Define the columns to scale
    columns_to_scale = ['help','office','dance','money','wedding','domestic_work','sleep','medical_emergency','cold','hate','cheerfulness','aggression','occupation','envy','anticipation','family','vacation','crime','attractive','masculine','prison','health','pride','dispute','nervousness','government','weakness','horror','swearing_terms','leisure','suffering','royalty','wealthy','tourism','furniture','school','magic','beach','journalism','morning','banking','social_media','exercise','night','kill','blue_collar_job','art','ridicule','play','computer','college','optimism','stealing','real_estate','home','divine','sexual','fear','irritability','superhero','business','driving','pet','childish','cooking','exasperation','religion','hipster','internet','surprise','reading','worship','leader','independence','movement','body','noise','eating','medieval','zest','confusion','water','sports','death','healing','legend','heroic','celebration','restaurant','violence','programming','dominant_heirarchical','military','neglect','swimming','exotic','love','hiking','communication','hearing','order','sympathy','hygiene','weather','anonymity','trust','ancient','deception','fabric','air_travel','fight','dominant_personality','music','vehicle','politeness','toy','farming','meeting','war','speaking','listen','urban','shopping','disgust','fire','tool','phone','gain','sound','injury','sailing','rage','science','work','appearance','valuable','warmth','youth','sadness','fun','emotional','joy','affection','traveling','fashion','ugliness','lust','shame','torment','economics','anger','politics','ship','clothing','car','strength','technology','breaking','shape_and_size','power','white_collar_job','animal','party','terrorism','smell','disappointment','poor','plant','pain','beauty','timidity','philosophy','negotiate','negative_emotion','cleaning','messaging','competing','law','friends','payment','achievement','alcohol','liquid','feminine','weapon','children','monster','ocean','giving','contentment','writing','rural','positive_emotion','musical']

    # Scale the columns
    try:
        df_essays, scaled_df, psycho_scaler = scale_columns(df_essays, columns_to_scale)
        joblib.dump(psycho_scaler, 'scaler_psycho_features.pkl', compress=True)
        return df_essays
    except Exception as e:
        logger.error(f"Error scaling columns: {e}")
        return df_essays
    

def scale_tree_features(df_essays):
    import logging
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler, StandardScaler

    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # Define the columns to scale
    columns_to_scale = ['num_sentences', 'num_tokens', 'num_unique_lemmas', 'average_token_length', 'average_sentence_length', 'num_entities', 'num_noun_chunks', 'num_pos_tags', 'num_distinct_entities', 'average_entity_length', 'average_noun_chunk_length', 'max_depth', 'avg_branching_factor', 'total_nodes', 'total_leaves', 'unique_rules', 'tree_complexity', 'depth_variability']

    # Scale the columns
    try:
        df_essays, scaled_df, tree_scaler = scale_columns(df_essays, columns_to_scale)
        joblib.dump(tree_scaler, 'scaler_tree_features.pkl', compress=True)
        return df_essays
    except Exception as e:
        logger.error(f"Error scaling columns: {e}")
        return df_essays
        
'''Create Pipeline'''
#@PipelineDecorator.pipeline(name="Preprocessing Pipeline - Feature Generation", project="LLM-detect-ai-gen-text-LIVE/dev/features_pipeline")
def execute_features_pipeline(df_essays):
    import logging
    import numpy as np
    import pandas as pd
    
    print("df_essays:\n" + df_essays.head().to_string())
    print(df_essays.info())

    df_essays['text'].apply(preprocess_text)

    '''Readability Scores'''
    # =============================================================================
    df_readability_essays = process_readability_scores(df_essays)

    '''Semantic Density'''
    # =============================================================================
    df_semantic_essays = process_semantic_density(df_readability_essays)

    '''Semantic Flow Variability'''
    # =============================================================================
    df_variability_essays = process_semantic_flow_variability(df_semantic_essays)

    '''Psycholuigustic Features'''
    # =============================================================================
    df_psyco_essays = apply_empath_analysis(df_variability_essays)

    '''Textrual Entropy'''
# =============================================================================
    df_entropy = process_textual_entropy(df_psyco_essays)

    '''Syntactic Tree Patterns'''
    # =============================================================================
    df_essays = process_syntactic_tree_patterns(df_entropy)
    
    '''Scale Features'''
    # =============================================================================
    
    
    df_essays = scale_readability_scored(df_essays)
    df_essays = scale_psycho_features(df_essays)
    df_essays = scale_tree_features(df_essays)
    
    
    return df_essays
    
    

def generate_features(df):

    # Create a DataFrame from the string
    # df = pd.DataFrame({'text': [text]})
    # print(text.head(1))

    df_essays = execute_features_pipeline(df)
    print("process completed")
    return df_essays

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='EBM Text Classification Configuration')
    parser.add_argument('--exclude', nargs='+', help='List of features to exclude', default=[])
    parser.add_argument('--max-bins', type=int, default=255, help='Maximum number of bins for numeric features')
    parser.add_argument('--validation-size', type=float, default=0.20, help='Size of validation set')
    parser.add_argument('--outer-bags', type=int, default=25, help='Number of outer bags for boosting')
    parser.add_argument('--inner-bags', type=int, default=25, help='Number of inner bags for boosting')
    parser.add_argument('--learning-rate', type=float, default=0.01, help='Learning rate for boosting')
    parser.add_argument('--greediness', type=float, default=0.0, help='Greediness for boosting')
    parser.add_argument('--smoothing-rounds', type=int, default=0, help='Number of smoothing rounds')
    parser.add_argument('--early-stopping-rounds', type=int, default=50, help='Number of rounds for early stopping')
    parser.add_argument('--early-stopping-tolerance', type=float, default=0.0001, help='Tolerance for early stopping')
    parser.add_argument('--objective', type=str, default='roc_auc', help='Objective for training')
    parser.add_argument('--n-jobs', type=int, default=-2, help='Number of parallel jobs to run')
    parser.add_argument('--random-state', type=int, default=42, help='Random state for reproducibility')

    return parser.parse_args()

args = parse_args()
task.connect(vars(args))

# MODEL TRAIN 
kaggle_training_data = pd.read_csv('/Users/lange/dev/clearml-serving/examples/kaggle_train_data.csv/train_v2_drcat_02.csv')
random_kaggle_training_data_0 = kaggle_training_data[kaggle_training_data['label'] == 0].sample(n=100) 
random_kaggle_training_data_1 = kaggle_training_data[kaggle_training_data['label'] == 1].sample(n=100) 
combined_data = pd.concat([random_kaggle_training_data_0, random_kaggle_training_data_1])
df_combined = combined_data.reset_index(drop=True)
df_combined.drop_duplicates(inplace=True)
texts = df_combined[['text','label']]#.str.lower().tolist()  # Lowercase for uncased BERT
labels = df_combined['label'].tolist()

preprocess = Preprocess()

features = preprocess.generate_features(texts)

model_config = {
    'feature_names': features.columns.tolist(),
    'feature_types': None,
    'exclude': args.exclude,
    'max_bins': args.max_bins,
    'validation_size': args.validation_size,
    'outer_bags': args.outer_bags, # recommended for best accuracy
    'inner_bags': args.inner_bags, # recommended for best accuracy
    'learning_rate': args.learning_rate,
    'greediness': args.greediness,
    'smoothing_rounds': args.smoothing_rounds,
    'early_stopping_rounds': args.early_stopping_rounds,
    'early_stopping_tolerance': args.early_stopping_tolerance,
    'objective': args.objective,
    'n_jobs':args.n_jobs,
    'random_state': args.random_state
}
#config_dict = task.connect(model_config, name="model_config")


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Hyperparameter tuning setup
param_test = {
    'learning_rate': [0.001, 0.005, 0.01, 0.03],
    'max_rounds': [5000, 10000, 15000, 20000],
    'min_samples_leaf': [2, 3, 5],
    'max_leaves': [3, 5, 10]
}
n_HP_points_to_test = 10

# EBM Model Initialization
ebm_clf = ExplainableBoostingClassifier(feature_names=features.columns.tolist(), feature_types=None, n_jobs=-2, random_state=42)

# Randomized Search for Hyperparameter Tuning
ebm_gs = RandomizedSearchCV(
    estimator=ebm_clf,
    param_distributions=param_test,
    n_iter=n_HP_points_to_test,
    scoring="roc_auc",
    cv=3,
    refit=True,
    random_state=314,
    verbose=False
)
model = ebm_gs.fit(X_train, y_train)

# Retraining with best parameters
best_params = ebm_gs.best_params_
ebm = ExplainableBoostingClassifier(**best_params)
ebm.fit(X_train, y_train)

# Evaluation
roc_auc = ROC(ebm.predict_proba).explain_perf(X_test, y_test, name='EBM ROC AUC')
pr_curve = PR(ebm.predict_proba).explain_perf(X_test, y_test, name='EBM Precision-Recall')



# Save the trained model
model_path = "examples/model/ebm.pkl"
with open(model_path, 'wb') as f:
    pickle.dump(ebm, f)

# Log the model to ClearML

output_model = OutputModel(task=task)
output_model.update_weights(model_path)

#Finalize the ClearML task
task.close()



