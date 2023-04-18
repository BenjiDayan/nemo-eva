import numpy

from sklearn.model_selection import \
    cross_val_predict, GridSearchCV, permutation_test_score


class Classification:
    def __init__(self, X, Y, model, cv_grid=None):
        assert numpy.isfinite(X).all().all()

        cv = 3

        if cv_grid:
            # print(f'model: {model}, cv_grid: {cv_grid}, scoring: accuracy, cv: {cv}, refit: False')
            grid_search = GridSearchCV(
                model, cv_grid, scoring="accuracy", cv=cv, refit=False)
            grid_search.fit(X, Y)
            self.best_params = grid_search.best_params_
            model.set_params(**self.best_params)
        else:
            self.best_params = None

        self.results = {
            "cv":   {"prediction": cross_val_predict(model, X, Y, cv=cv)}
        }

        for result in self.results.values():
            mask = (Y != result["prediction"])
            result["mask"] = mask
            result["accuracy"] = sum(~mask)/len(mask)
