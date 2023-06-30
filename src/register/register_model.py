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

        # self.remote_server_uri = "http://192.168.68.53:12000/" # this value has been replaced
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
        
        client.set_experiment_tag(experiment_id, "teste", "0")
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

            # predicted_qualities = lr.predict(test_x)
            y_pred = train_model.predict()
            X_train = train_model.get_X_train()
            y_test = train_model.get_y_test()

            params  = train_model.get_params()
            metrics = train_model.eval_metrics(y_test, y_pred)

            # print("Elasticnet model (alpha={:f}, l1_ratio={:f}):".format(alpha, l1_ratio))
            # print("  RMSE: %s" % rmse)
            # print("  MAE: %s" % mae)
            # print("  R2: %s" % r2)

            # mlflow.log_param("n_estimators", n_estimators)
            # mlflow.log_param("criterion", criterion)
            # mlflow.log_param("random_state", random_state)
            # mlflow.log_metric("accuracy", accuracy)
            # mlflow.log_metric("f1", f1)
            # mlflow.log_metric("roc_auc", roc_auc)

            self.log_params(params)
            self.log_metrics(metrics)

            signature = infer_signature(X_train, y_test)

            # tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            # Model registry does not work with file store
            # if tracking_url_type_store != "file":
            #     # Register the model
            #     # There are other ways to use the Model Registry, which depends on the use case,
            #     # please refer to the doc for more information:
            #     # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            
            # TODO REMOVE HARDCODE
            mlflow.sklearn.log_model(trained_model, 
                                    "model",
                                    registered_model_name="RandomForestClassifierBreastCancerModel",
                                    signature=signature)
            # else:
            # mlflow.sklearn.log_model(classifier, "model") #, signature=signature)

            mlflow.log_artifacts("data", artifact_path="data")
            self.set_complement_info(run=run)