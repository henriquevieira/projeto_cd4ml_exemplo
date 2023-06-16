
import json

import pandas as pd
from sklearn.model_selection import train_test_split

from src.models.load_model import Model

class TrainModel():

    def __init__(self, X, y, test_size = 0.20, random_state = 0):
        self.X = X
        self.y = y
        self.test_size = test_size
        self.random_state = random_state

        self.X_train, self.X_test, self.y_train, self.y_test = self.spliting_data()
        self.model = self.training()

    def spliting_data(self):

        X_train, X_test, y_train, y_test = train_test_split(self.X, 
                                                            self.y, 
                                                            test_size = self.test_size, 
                                                            random_state = self.random_state)

        return X_train, X_test, y_train, y_test

    def load_parameters(self, params_directory):
        try:
            with open(params_directory, 'r') as file:
                return json.load(file)
        except Exception as e:
            print("ERROR LOAD PARAMETERS")
            print(e)

    def training(self):

        # TODO REMOVE HARDCODE
        algorithm_name = "RandomForestClassifier"
        params_directory = 'src/models/algorithms/RandomForestClassifier/default.json'

        params = self.load_parameters(params_directory)
        m = Model(algorithm_name = algorithm_name)
        algorithm = m.load_model()

        model = algorithm(params,  self.X_train, self.y_train)
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