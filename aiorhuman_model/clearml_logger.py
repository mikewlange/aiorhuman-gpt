from clearml.automation.controller import PipelineDecorator
from clearml import TaskTypes, PipelineController, StorageManager, Dataset, Task
from clearml import InputModel, OutputModel
import logging
from IPython.display import display
import ipywidgets as widgets
import argparse
class ClearMLTaskHandler:
    def __init__(self, project_name, task_name, config=None):
        self.task = self.get_or_create_task(project_name, task_name)
        self.logger = None  # Initialize logger attribute
        self.setup_widget_logger()

        if config:
            self.set_config(config)

    def get_or_create_task(self, project_name, task_name):
        try:
            tasks = []
            if(CFG.CLEARML_OFFLINE_MODE):
                Task.set_offline(offline_mode=True)
            else:
                tasks = Task.get_tasks(project_name=project_name, task_name=task_name)
            
            if tasks:
                if(tasks[0].get_status() == "created" and task[0].task_name == task_name):
                    task = tasks[0]
                    return task
                else:
                    if(CFG.CLEARML_OFFLINE_MODE):
                        Task.set_offline(offline_mode=True)
                        
                    task = Task.init(project_name=project_name, task_name=task_name)
                    return task
            else:
                if(CFG.CLEARML_OFFLINE_MODE):
                    Task.set_offline(offline_mode=True)
                    task = Task.init(project_name=project_name, task_name=task_name)
                else:
                    task = Task.init(project_name=project_name, task_name=task_name)
                return task
        except Exception as e:
            print(f"Error occurred while searching for existing task: {e}")
            return None

    def set_parameters(self, parameters):
        """
        Set hyperparameters for the task.
        :param parameters: Dictionary of parameters to set.
        """
        self.task.set_parameters(parameters)

    def set_config(self, config):
        if isinstance(config, dict):
            self.task.connect(config)
        elif isinstance(config, argparse.Namespace):
            self.task.connect(config.__dict__)
        elif isinstance(config, (InputModel, OutputModel, type, object)):
            self.task.connect_configuration(config)
        else:
            logging.warning("Unsupported configuration type")

    def log_data(self, data, title):
        self.task.get_logger()
        if isinstance(data, np.ndarray):
            self.task.get_logger().report_image(title, 'array', iteration=0, image=data)
        elif isinstance(data, pd.DataFrame):
            self.task.get_logger().report_table(title, 'dataframe', iteration=0, table_plot=data)
        elif isinstance(data, str) and os.path.exists(data):
            self.task.get_logger().report_artifact(title, artifact_object=data)
        else:
            self.task.get_logger().report_text(f"{title}: {data}")
    
    def upload_artifact(self, name, artifact):
        """
        Upload an artifact to the ClearML server.
        :param name: Name of the artifact.
        :param artifact: Artifact object or file path.
        """
        self.task.upload_artifact(name, artifact_object=artifact)

    def get_artifact(self, name):
        """
        Retrieve an artifact from the ClearML server.
        :param name: Name of the artifact to retrieve.
        :return: Artifact object.
        """
        return self.task.artifacts[name].get()
    
    def setup_widget_logger(self):
            handler = OutputWidgetHandler()
            handler.setFormatter(logging.Formatter('%(asctime)s  - [%(levelname)s] %(message)s'))
            self.logger = logging.getLogger()  # Create a new logger instance
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)


# Just in case we can't use clearml in kaggle
class OutputWidgetHandler(logging.Handler):
    def __init__(self, *args, **kwargs):
        super(OutputWidgetHandler, self).__init__(*args, **kwargs)
        layout = {'width': '100%', 'border': '1px solid black'}
        self.out = widgets.Output(layout=layout)

    def emit(self, record):
        formatted_record = self.format(record)
        new_output = {'name': 'stdout', 'output_type': 'stream', 'text': formatted_record+'\n'}
        self.out.outputs = (new_output, ) + self.out.outputs

    def show_logs(self):
        display(self.out)

    def clear_logs(self):
        self.out.clear_output()

# Keeping this out for simpicity 
def upload_dataset_from_dataframe(dataframe, new_dataset_name, dataset_project, description="", tags=[], file_name="dataset.pkl"):
    from pathlib import Path
    from clearml import Dataset
    import pandas as pd
    import logging
    try:
        print(dataframe.head())
        file_path = Path(file_name)
        pd.to_pickle(dataframe, file_path)
        new_dataset = Dataset.create(new_dataset_name,dataset_project, description=description)
        new_dataset.add_files(str(file_path))
        if description:
            new_dataset.set_description(description)
        if tags:
            new_dataset.add_tags(tags)
        new_dataset.upload()
        new_dataset.finalize()
        return new_dataset
    except Exception as e:
        return logging.error(f"Error occurred while uploading dataset: {e}")
    
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
