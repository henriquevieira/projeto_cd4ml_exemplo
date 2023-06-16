import pandas as pd

class LoadData():

    def __init__(self, path):
        self.path = path
        self.data = None

    def get_data(self):

        if self.data:
            return self.data
        else:
            try:
                print('LOAD DATA')
                self.data = self.load_data_from_path()
                return self.data
            except Exception as e:
                print('ERROR TO LOAD DATA')
                print(e)

    def load_data_from_path(self):
        data = pd.read_csv(self.path, sep=';')
        return data

    def seperate_x_and_y(self):
        data = self.get_data()
        X = data.drop('diagnose', axis=1)
        y = data['diagnose']

        return X, y
