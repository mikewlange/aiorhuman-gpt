import pandas as pd
import pickle
from interpret.glassbox import ExplainableBoostingClassifier
from interpret import show
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from interpret.perf import ROC, PR
from clearml import Task, OutputModel
from preprocess import Preprocess
import sys

sys.path.append('examples/model')
import sys
import logging
from preprocess import Preprocess

# Set up logging
logging.basicConfig(level=logging.INFO)
task = Task.init(project_name='Models - Text Classification', task_name='train ebm model', output_uri=True)
class EBMModel():

    def setUp(self):
        # Initialize the Preprocess object before each test
        self.preprocess = Preprocess()

    def test_preprocess_text(self):
        # Test the preprocess_text function for expected behavior
        input_text = "HThe social dynamics and peer interactions in a physical school can sometimes lead to the proliferation of rumors and drama, which can significantly impact a student's emotional well-being and academic focus. In contrast, attending school from home provides a shield from these negative influences, allowing students to remain focused on their studies without being distracted or distressed by external factors. This isolation from detrimental social dynamics can lead to a more positive and productive learning environment for students, fostering their academic growth and emotional stability.\n\nFurthermore, the potential advantages of distance learning on the student body as a whole extend beyond individual well-being. Remote learning promotes inclusivity by accommodating students with diverse needs and circumstances, such as those with physical disabilities, chronic illnesses, or those balancing familial responsibilities. A study by the National Education Association highlighted that distance learning facilitates greater equity and access to education for students who may face barriers in a traditional school setting. By embracing remote learning, educational institutions can create an environment that caters to the varied needs of their student body, fostering an inclusive and supportive learning community.\n\nConsidering these aspects, it is evident that students attending school from home can reap a multitude of benefits. This alternative mode of learning offers a supportive and conducive environment for students with anxiety or depression, shields them from the detrimental impact of drama and rumors, and promotes inclusivity and equity within the student body. As such, it is crucial for educational institutions to recognize the potential advantages of remote learning and consider its implementation as a means of enhancing the overall well-being and academic success of their students.\n\nIn conclusion, the benefits of students attending school from home are substantial and wide-ranging. From supporting students with anxiety or depression to mitigating the impact of social dynamics on academic performance and fostering inclusivity, remote learning offers a promising avenue for educational advancement. By prioritizing the well-being and educational needs of students, embracing remote learning can pave the way for a more holistic and supportive approach to education. Therefore, it is imperative for educators and policymakers to consider the potential advantages of remote learning and seriously contemplate its integration into the educational framework."
        #expected_output = "hello world this is a test"
        output = self.preprocess.preprocess_text(input_text)
        
        logging.info(f'Output: {output}')

    def test_generate_features(self):
        # Test the generate_features_from_mygpt function to ensure it returns a DataFrame
        input_text = "HThe social dynamics and peer interactions in a physical school can sometimes lead to the proliferation of rumors and drama, which can significantly impact a student's emotional well-being and academic focus. In contrast, attending school from home provides a shield from these negative influences, allowing students to remain focused on their studies without being distracted or distressed by external factors. This isolation from detrimental social dynamics can lead to a more positive and productive learning environment for students, fostering their academic growth and emotional stability.\n\nFurthermore, the potential advantages of distance learning on the student body as a whole extend beyond individual well-being. Remote learning promotes inclusivity by accommodating students with diverse needs and circumstances, such as those with physical disabilities, chronic illnesses, or those balancing familial responsibilities. A study by the National Education Association highlighted that distance learning facilitates greater equity and access to education for students who may face barriers in a traditional school setting. By embracing remote learning, educational institutions can create an environment that caters to the varied needs of their student body, fostering an inclusive and supportive learning community.\n\nConsidering these aspects, it is evident that students attending school from home can reap a multitude of benefits. This alternative mode of learning offers a supportive and conducive environment for students with anxiety or depression, shields them from the detrimental impact of drama and rumors, and promotes inclusivity and equity within the student body. As such, it is crucial for educational institutions to recognize the potential advantages of remote learning and consider its implementation as a means of enhancing the overall well-being and academic success of their students.\n\nIn conclusion, the benefits of students attending school from home are substantial and wide-ranging. From supporting students with anxiety or depression to mitigating the impact of social dynamics on academic performance and fostering inclusivity, remote learning offers a promising avenue for educational advancement. By prioritizing the well-being and educational needs of students, embracing remote learning can pave the way for a more holistic and supportive approach to education. Therefore, it is imperative for educators and policymakers to consider the potential advantages of remote learning and seriously contemplate its integration into the educational framework."
        output_df = self.preprocess.generate_features(input_text)
        logging.info(f'Output DataFrame: \n{output_df}')

    # Add more tests as needed for other methods in your Preprocess class

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

features = preprocess.generate_features(texts,is_inference=False)

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

# Visualization (optional, depending on your environment)
# show(roc_auc)
# show(pr_curve)

# Save the trained model
model_path = "examples/model/ebm.pkl"
with open(model_path, 'wb') as f:
    pickle.dump(ebm, f)

# Log the model to ClearML

output_model = OutputModel(task=task)
output_model.update_weights(model_path)

#Finalize the ClearML task
task.close()
