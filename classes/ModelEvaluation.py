import pandas as pd
import numpy as np
import pickle
from sklearn.dummy import DummyRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, TimeSeriesSplit, GridSearchCV
from sklearn.impute import KNNImputer
from sklearn.linear_model import SGDRegressor


class ModelEvaluation:
    def __init__(self) -> None:
        # self.train_X, self.train_y, _, _ = self.train_test_time_split(model_data)
        self.time_series_split = TimeSeriesSplit()
        pass

    def train_test_time_split(
        self, model_data: pd.DataFrame, combined=False
    ) -> (
        tuple[pd.DataFrame, pd.DataFrame]
        | tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
    ):
        """
        Split data for modelling into train and test.

        Parameters:
        ----------
        model_data: dataframe to train models on
        combined: bool (True: keep train and test as full dataframes)

        return:

            tuple[Dataframe, Dataframe] or tuple[DataFrame, DataFrame, Series, Series]
        """
        train_data: pd.DataFrame = model_data.drop(
            model_data.loc["2022"].index, inplace=False
        )
        test_data: pd.DataFrame = model_data.loc["2022"]

        if combined:
            return train_data, test_data

        train_X: pd.DataFrame = train_data.drop("Total Trafikkmengde", axis=1)
        test_X: pd.DataFrame = test_data.drop("Total Trafikkmengde", axis=1)
        train_y: pd.Series = train_data["Total Trafikkmengde"]
        test_y: pd.Series = test_data["Total Trafikkmengde"]

        return train_X, test_X, train_y, test_y

    def cross_validate(self, X, y, estimator, parameters=None) -> tuple:
        """
        Cross validates a given model and returns that model
        trained on the whole training set
        """
        gscv = GridSearchCV(
            estimator,
            parameters,
            cv=self.time_series_split,
            verbose=3,
            scoring="neg_mean_squared_error",
            n_jobs=4
        )
        gscv.fit(X, y)
        best_model = gscv.best_estimator_
        best_score = np.sqrt(-gscv.best_score_)
        best_params = gscv.best_params_
        print(f"Mean score of best estimator: {best_score}")
        print(f"With parameters: {best_params}\n")
        return best_model, best_score

    def save_model(self, model, model_name) -> None:
        """
        Save model to file with specified name
        """
        with open(f"models/{model_name}.pickle", "wb") as f:
            pickle.dump(model, file=f)

    def load_trained_model(self, model_name):
        """
        Load trained model from file

        return: model
        """
        with open(f"models/{model_name}.pickle", "rb") as f:
            model = pickle.load(f)

        return model

    def support_vector(self, X, y, save_to_file=False) -> tuple:
        parameters = {
            "kernel": ["linear", "poly", "rbf"],
            # "degree": [3, 4, 5],
        }

        svr = SVR()

        print("Running svr")

        model, score = self.cross_validate(X, y, svr, parameters)

        if save_to_file:
            self.save_model(model, "SVR")

        return model, score

    def k_neighbors(self, X, y, save_to_file=False) -> tuple:
        parameters = {"n_neighbors": [1, 5, 10, 20, 30, 40, 50]}

        model = KNeighborsRegressor()
        print("Running KNN Regressor")
        model, score = self.cross_validate(X, y, model, parameters)

        if save_to_file:
            self.save_model(model, "KNNR")

        return model, score

    def random_forest(self, X, y, save_to_file=False) -> tuple:
        parameters = {"n_estimators": [100, 150]}
        model = RandomForestRegressor()

        print("Running RandomForestRegressor")
        model, score = self.cross_validate(X, y, model, parameters)
        if save_to_file:
            self.save_model(model, "RandomForestRegressor")
        return model, score

    def dummy_regressor(self, X, y, save_to_file=False) -> tuple:
        parameters = {"strategy": ["mean", "median"]}

        model = DummyRegressor()
        print("Running Dummy Regressor")
        model, score = self.cross_validate(X, y, model, parameters)
        if save_to_file:
            self.save_model(model, "DummyRegressor")
        return model, score
