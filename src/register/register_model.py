
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from urllib.parse import urlparse

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src.models.train_model import TrainModel

class RegisterModel:

    def __init__(self, data):

        self.data = data

        self.remote_server_uri = "http://192.168.68.53:12000/" # this value has been replaced
        self.tags = {
                "Projeto": "Tutorial CD4ML",
                "team": "Ciencia de dados",
                "dataset": "Breast Cancer"
            }

    def eval_metrics(self, actual, pred):

        accuracy = accuracy_score(actual, pred)
        f1 = f1_score(actual, pred)
        roc_auc = roc_auc_score(actual, pred)
        
        return accuracy, f1, roc_auc

    def log_params(self):

        for param_name in self.params.keys():
            param_content = self.params[param_name]
            mlflow.log_param(param_name, param_content)

    def log_metrics(self):

        for metric_name in self.metrics.keys():
            metric_content = self.metrics[metric_name]
            mlflow.log_metric(metric_name, metric_content)

    def do_register(self):

        mlflow.set_tracking_uri(uri=self.remote_server_uri)
        mlflow.set_experiment(experiment_name='projeto_cd4ml_exemplo')

        X, y = self.data.seperate_x_and_y()
        train_model = TrainModel(X = X, y = y)

        with mlflow.start_run():

            trained_model = train_model.get_trained_model()

            # predicted_qualities = lr.predict(test_x)
            y_pred = train_model.predict()
            X_train = train_model.get_X_train()
            y_test = train_model.get_y_test()

            (accuracy, f1, roc_auc) = self.eval_metrics(y_test, y_pred)

            # print("Elasticnet model (alpha={:f}, l1_ratio={:f}):".format(alpha, l1_ratio))
            # print("  RMSE: %s" % rmse)
            # print("  MAE: %s" % mae)
            # print("  R2: %s" % r2)

            # mlflow.log_param("n_estimators", n_estimators)
            # mlflow.log_param("criterion", criterion)
            # mlflow.log_param("random_state", random_state)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("f1", f1)
            mlflow.log_metric("roc_auc", roc_auc)

            signature = infer_signature(X_train, y_test)

            # tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            # Model registry does not work with file store
            # if tracking_url_type_store != "file":
            #     # Register the model
            #     # There are other ways to use the Model Registry, which depends on the use case,
            #     # please refer to the doc for more information:
            #     # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            
            mlflow.sklearn.log_model(trained_model, 
                                    "model",
                                    registered_model_name="RandomForestClassifierBreastCancerModel",
                                    signature=signature)
            # else:
            # mlflow.sklearn.log_model(classifier, "model") #, signature=signature)

            mlflow.log_artifacts("data", artifact_path="data")