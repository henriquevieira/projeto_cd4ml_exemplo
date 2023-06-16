from sklearn.ensemble import RandomForestClassifier

class Model:

    def __init__(self, parameters,  X_train, y_train):
        self.parameters = parameters
        self.model = self.get_model()
        self.X_train = X_train
        self.y_train = y_train

    def get_model(self):
        params = self.parameters
        model = RandomForestClassifier(**params)
        return model

    def fit(self):
        self.model.fit(self.X_train, self.y_train)

    def predict(self, y_test):
        y_pred = self.model.predict(y_test)
        return y_pred