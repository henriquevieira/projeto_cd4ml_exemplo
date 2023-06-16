import os

from src.data.download_dataset import Download
from src.data.load_data import LoadData
from src.models.train_model import TrainModel
from src.register.register_model import RegisterModel

# Download data

absolute_path = os.path.abspath('')
data_raw_path = os.path.join(absolute_path, 'data/raw')

download = Download(destination_path = data_raw_path)
response = download.download()
data_path = response[0]

# load data
data = LoadData(path = data_path)

# processing data

# training
# TODO verify mlflow necessity

# register model
rg = RegisterModel(data = data)

rg.do_register()




