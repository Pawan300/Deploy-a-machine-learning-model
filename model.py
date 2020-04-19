import pickle as pkl

import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score


class ml_model:
    def __init__(self):
        self.model = None
        self.data = None
        self.X_train = None
        self.X_test = None
        self.Y_train = None
        self.Y_test = None

    def load_data(self):
        wine = load_wine()
        self.data = pd.DataFrame(
            data=np.c_[wine["data"], wine["target"]],
            columns=wine["feature_names"] + ["target"],
        )

    def print_data(self):
        print(self.data.head())

    def prepare_data(self):
        self.X_train = self.data[:-20]
        self.X_test = self.data[-20:]

        self.Y_train = self.X_train.target
        self.Y_test = self.X_test.target

        self.X_train = self.X_train.drop("target", 1)
        self.X_test = self.X_test.drop("target", 1)

    def classifier(self):
        self.model = tree.DecisionTreeClassifier()
        self.model = self.model.fit(self.X_train, self.Y_train)

    def prediction(self):
        Y_pred = self.model.predict(self.X_test)
        print("accuracy_score: %.2f" % accuracy_score(self.Y_test, Y_pred))

    def save_model(self):
        pkl.dump(self.model, open("final_prediction.pickle", "wb"))


if __name__ == "__main__":
    demo = ml_model()
    demo.load_data()
    demo.print_data()
    demo.prepare_data()
    demo.classifier()
    demo.prediction()
    demo.save_model()
