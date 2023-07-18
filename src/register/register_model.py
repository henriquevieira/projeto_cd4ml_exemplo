import os

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.store.artifact.runs_artifact_repo import RunsArtifactRepository
from mlflow.models.signature import infer_signature
from urllib.parse import urlparse

from src.models.train_model import TrainModel

class RegisterModel:

    def __init__(self, experiment_name, data, algorithm_name, params_filepath):
        self.experiment_name = experiment_name
        self.data = data
        self.algorithm_name = algorithm_name
        self.params_filepath = params_filepath

        self.remote_server_uri = os.environ['MLFLOW_TRACKING_URL']

    def log_params(self, params):

        for param_name in params.keys():
            param_content = params[param_name]
            mlflow.log_param(param_name, param_content)

    def log_metrics(self, metrics):

        for metric_name in metrics.keys():
            metric_content = metrics[metric_name]
            mlflow.log_metric(metric_name, metric_content)

    def set_complement_info(self, run):
        client = MlflowClient(tracking_uri=self.remote_server_uri)
        # TODO REMOVE HARDCODE
        name_model = "RandomForestClassifierBreastCancerModel"
        desc = "A new version of the model"
        
        new_run_id = run.info.run_id
        version = client.search_model_versions("run_id='{}'".format(new_run_id))[0].version
        
        # client.set_experiment_tag(experiment_id, "teste", "0")
        client.transition_model_version_stage(name_model, version, "Production", archive_existing_versions = True)
        # mv = client.get_model_version(name=name_model, version=version)
        mv = client.update_model_version(name_model, version, desc)
        print("Name: {}".format(mv.name))
        print("Version: {}".format(mv.version))
        print("Description: {}".format(mv.description))
        print("Status: {}".format(mv.status))
        print("Stage: {}".format(mv.current_stage))



    def do_register(self):

        mlflow.set_tracking_uri(uri=self.remote_server_uri)
        # TODO REMOVE HARDCODE
        mlflow.set_experiment(experiment_name=self.experiment_name)

        X, y = self.data.seperate_x_and_y()
        train_model = TrainModel(algorithm_name = self.algorithm_name, 
                                 params_filepath = self.params_filepath, 
                                 X = X, y = y)

        with mlflow.start_run() as run:

            trained_model = train_model.get_trained_model()

            y_pred = train_model.predict()
            X_train = train_model.get_X_train()
            y_test = train_model.get_y_test()

            params  = train_model.get_params()
            metrics = train_model.eval_metrics(y_test, y_pred)

            self.log_params(params)
            self.log_metrics(metrics)

            signature = infer_signature(X_train, y_test)
            
            # TODO REMOVE HARDCODE
            mlflow.sklearn.log_model(trained_model, 
                                    "model",
                                    registered_model_name="RandomForestClassifierBreastCancerModel",
                                    signature=signature)

            mlflow.log_artifacts("data", artifact_path="data")
            self.set_complement_info(run=run)