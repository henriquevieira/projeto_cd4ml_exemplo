import os

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from urllib.parse import urlparse
from sklearn.ensemble import RandomForestClassifier


from src.data.download_dataset import Download
from src.models.train_model import TrainModel
from src.register.register_model import RegisterModel

# Download data

absolute_path = os.path.abspath('')
data_raw_path = os.path.join(absolute_path, 'data/raw')

download = Download(destination_path = data_raw_path)
response = download.download()
data_path = response[0]

# load data
train_model = TrainModel(path = data_path)
train_model.load_data()

# processing data

# splitting data
X_train, X_test, y_train, y_test = train_model.split_data()

# training model
# classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
# classifier.fit(X_train, y_train)

# # metrics
# y_pred = classifier.predict(X_test)

# Tracking

remote_server_uri = "http://192.168.68.52:12000/" # this value has been replaced
mlflow.set_tracking_uri(uri=remote_server_uri)
mlflow.set_experiment(experiment_name='projeto_cd4ml_exemplo')

tags = {
        "Projeto": "Tutorial CD4ML",
        "team": "Ciencia de dados",
        "dataset": "Breast Cancer"
       }

n_estimators = 10
criterion = 'entropy'
random_state = 0


rg = RegisterModel()

with mlflow.start_run():

    classifier = RandomForestClassifier(n_estimators = n_estimators, criterion = criterion, random_state = random_state)
    classifier.fit(X_train, y_train)

    # predicted_qualities = lr.predict(test_x)
    y_pred = classifier.predict(X_test)

    (accuracy, f1, roc_auc) = rg.eval_metrics(y_test, y_pred)

    # print("Elasticnet model (alpha={:f}, l1_ratio={:f}):".format(alpha, l1_ratio))
    # print("  RMSE: %s" % rmse)
    # print("  MAE: %s" % mae)
    # print("  R2: %s" % r2)

    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("criterion", criterion)
    mlflow.log_param("random_state", random_state)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1", f1)
    mlflow.log_metric("roc_auc", roc_auc)

    predictions = classifier.predict(X_train)
    signature = infer_signature(X_train, predictions)

    # tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

    # Model registry does not work with file store
    # if tracking_url_type_store != "file":
    #     # Register the model
    #     # There are other ways to use the Model Registry, which depends on the use case,
    #     # please refer to the doc for more information:
    #     # https://mlflow.org/docs/latest/model-registry.html#api-workflow
    
    mlflow.sklearn.log_model(classifier, 
                                 "model", 
                                 registered_model_name="RandomForestClassifierBreastCancerModel",
                                 signature=signature)
    # else:
    # mlflow.sklearn.log_model(classifier, "model") #, signature=signature)

    mlflow.log_artifacts("data", artifact_path="data")


