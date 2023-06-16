import os
import glob

class Model:

    def __init__(self, algorithm_name):
        self.algorithm_dir = 'algorithms'
        self.algorithm_name = algorithm_name

    def list_algorithm_names(self):

        algorithms_names = glob.glob(self.algorithm_dir)
        return algorithms_names

    def find_algorithm(self):
        algorithm_path = os.path.join(self.algorithm_dir, self.algorithm_name)
        return os.path.exists(algorithm_path)

    def load_model(self):

        model = None
        try:
            if self.find_algorithm:
                print("FIND ALGORITHM")
                name = "Model"
                # package = "src.models.algorithms.{}.model".format(self.algorithm_name)
                package = "src.models.algorithms.RandomForestClassifier.model"
                print(package)
                model = getattr(__import__(package, fromlist=[name]), name)
                return model
        except Exception as e:
            print("ERROR MODEL NOT FOUND")
            print(e)