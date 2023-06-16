import os
import glob
import json

class Load:

    def __init__(self, algorithm_name):
        self.algorithm_dir = 'src/models/algorithms'
        self.algorithm_name = algorithm_name

    def list_algorithm_names(self):

        algorithms_names = glob.glob(self.algorithm_dir)
        return algorithms_names

    def find_algorithm(self):
        algorithm_path = os.path.join(self.algorithm_dir, self.algorithm_name)
        return os.path.exists(algorithm_path)

    def load_class(self, class_name, package_name):
        package = "src.models.algorithms.{}.{}".format(self.algorithm_name, package_name)
        print(package)
        model = getattr(__import__(package, fromlist=[class_name]), class_name) # from x import y
        return model

    def load_algorithm_and_metrics(self):

        model = None
        try:
            if self.find_algorithm:
                print("FIND ALGORITHM")
                model = self.load_class(class_name = "Model", package_name = 'model')
                metrics = self.load_class(class_name = "Metrics", package_name = 'metrics')
                
                return model, metrics
        except Exception as e:
            print("ERROR MODEL NOT FOUND")
            print(e)

    def load_parameters(self, params_filepath):
        try:
            if self.find_algorithm:
                params_directory = os.path.join(self.algorithm_dir, self.algorithm_name, params_filepath)
                with open(params_directory, 'r') as file:
                    return json.load(file)
        except Exception as e:
            print("ERROR LOAD PARAMETERS")
            print(e)