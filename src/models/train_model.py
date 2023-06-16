
import json

import pandas as pd
from sklearn.model_selection import train_test_split

from src.models.loads import Load

class TrainModel():

    def __init__(self, algorithm_name, params_filepath, X, y, test_size = 0.20, random_state = 0):
        self.algorithm_name = algorithm_name
        self.params_filepath = params_filepath
        self.X = X
        self.y = y
        self.test_size = test_size
        self.random_state = random_state

        self.load = Load(algorithm_name = self.algorithm_name)
        self.X_train, self.X_test, self.y_train, self.y_test = self.spliting_data()
        self.algorithm, self.metrics = self.load_algorithm_and_metrics()
        self.params = self.load_parameters()
        self.model = self.training()

    def spliting_data(self):

        X_train, X_test, y_train, y_test = train_test_split(self.X, 
                                                            self.y, 
                                                            test_size = self.test_size, 
                                                            random_state = self.random_state)

        return X_train, X_test, y_train, y_test

    def load_parameters(self):
        params = self.load.load_parameters(params_filepath = self.params_filepath)
        return params

    def load_algorithm_and_metrics(self):
        
        # TODO REMOVE HARDCODE
        # algorithm_name = "RandomForestClassifier"
        
        algorithm, metrics = self.load.load_algorithm_and_metrics()
        return algorithm, metrics

    def training(self):

        # TODO REMOVE HARDCODE
        # params_directory = 'src/models/algorithms/RandomForestClassifier/default.json'
        model = self.algorithm(self.params, self.X_train, self.y_train)
        model.fit()

        return model

    def get_trained_model(self):
        return self.model

    def get_X_train(self):
        return self.X_train

    def get_X_test(self):
        return self.X_test

    def get_y_train(self):
        return self.y_train

    def get_y_test(self):
        return self.y_test

    def predict(self, X_test = None):

        if not X_test:
            X_test = self.X_test

        model = self.get_trained_model()
        return model.predict(X_test)

    def eval_metrics(self, actual, pred):

        metrics = self.metrics(actual, pred)
        return metrics.eval_metrics()

