from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

class Metrics:

    def __init__(self, actual, pred):
        self.actual = actual
        self.pred = pred

    def eval_metrics(self):
        metrics = {
                    "accuracy" : accuracy_score(self.actual, self.pred),
                    "f1" : f1_score(self.actual, self.pred),
                    "roc_auc" : roc_auc_score(self.actual, self.pred),
                }

        return metrics