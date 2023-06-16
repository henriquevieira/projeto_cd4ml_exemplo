import yaml

class PipelineConfig:

    def __init__(self):
        self.pipeline_file = "src/pipeline/pipeline_config.yaml"
        self.config = self.load_pipeline_configuration()

    def load_pipeline_configuration(self):
        try:
            with open(self.pipeline_file, "r") as file:
                config = yaml.load(file, Loader=yaml.FullLoader)
                return config
        except Exception as e:
            print("ERROR PIPELINE CONFIG FILE NOT FOUND")
            print(e)

    def get_config(self, config_name):
        try:
            config_value = self.config[config_name]
            return config_value
        except Exception as e:
            print("ERROR CONFIG NAME NOT FOUND")
            print(e)