from abstract_stage import AbstractStage
from helpers import dicts_to_df, format_feature_df
from helpers.classification import Classification
from helpers.feature_sets import get_all_feature_sets

import collections
import pandas
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import os

def classification_experiment(df, network_models, features_collection, printing=False):
    Y = df["Model"]

    models = {
        "svm": {
            "model": make_pipeline(StandardScaler(), SVC()),  # kernel="rbf", cache_size=500
            "cv_grid": {
                "svc__C": [10 ** exp for exp in range(5)],
                "svc__gamma": [10 ** -exp for exp in range(5)]
            }
        },
        # {
        #     "model": DummyClassifier,
        #     "params": {"strategy": "most_frequent"}
        # }
    }

    accuracies = pandas.DataFrame()

    cv_best_misclassified_masks = collections.defaultdict(dict)
    model_masks = {}

    for network_model in network_models:
        if printing:
            print("Model", network_model)
        network_model_mask = (df["Model"] == network_model) | (df["Model"] == "real-world")
        model_masks[network_model] = network_model_mask

        for features_name, features in sorted(features_collection.items()):
            if printing:
                print("Feature set", features_name)
            X = df[features]
            best_accuracy = 0
            for model_name, model in models.items():
                c = Classification(
                    X.loc[network_model_mask],
                    Y.loc[network_model_mask],
                    **model
                )
                cv_acc = c.results["cv"]["accuracy"]
                accuracies.loc[network_model, features_name] = cv_acc
                if cv_acc > best_accuracy:
                    best_accuracy = cv_acc
                    cv_best_misclassified_masks[features_name][network_model] = c.results["cv"]["mask"]
    else:
        return (accuracies, model_masks, cv_best_misclassified_masks)


class Classifier(AbstractStage):
    _stage = "4-classification_results"

    def __init__(self, features):
        super(Classifier, self).__init__()
        self.features = features

    def _execute(self):
        print('making output path')
        os.makedirs(self._stagepath + "accuracies/", exist_ok=True)
        df = dicts_to_df(self.features)
        format_feature_df(df)

        df_real = df[df["Model"] == "real-world"]
        print(collections.Counter(df["Model"]))

        small_avg_degree = df_real["Centrality.Degree.Location.Arithmetic Mean"] <= 30

        filters = {
            "all": [True] * len(df_real),
            "avg-degree-le-30": small_avg_degree,
            "avg-degree-gt-30": ~small_avg_degree,
            "socfb":     df_real["Type"] == "socfb",
            "not-socfb": df_real["Type"] != "socfb"
        }

        format_str = "{:20}{:>5}"
        network_models = sorted(set(df["Model"])-set(["real-world"]))

        for filtername, filterdf in sorted(filters.items()):
            graphs = sorted(df_real[filterdf]["Graph"])
            print(format_str.format(filtername, len(graphs)))

            features_collection = get_all_feature_sets(df, graphs)
            sub_df = df.loc(axis=0)[:, graphs, :]
            accuracies, model_masks, cv_best_misclassified_masks = \
                classification_experiment(
                    sub_df,
                    network_models,
                    features_collection,
                    printing=False)
            accuracies.to_csv(
                self._stagepath + "accuracies/" + filtername + ".csv",
                header=True,
                index_label="features"
            )
