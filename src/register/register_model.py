
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

class RegisterModel:

    def __init__(self):
        pass

    def eval_metrics(self, actual, pred):
        accuracy = accuracy_score(actual, pred)
        f1 = f1_score(actual, pred)
        roc_auc = roc_auc_score(actual, pred)
        
        return accuracy, f1, roc_auc