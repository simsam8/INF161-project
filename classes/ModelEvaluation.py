import pandas as pd
import pickle
import time
from sklearn.dummy import DummyRegressor
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import ElasticNet


class ModelEvaluation:
    """
    Class for evaluating different models and choosing the best one
    """

    def __init__(self, random_state=None) -> None:
        """
        Sets the internal random state and initializes
        a TimeSeriesSplit for cross validation.

        Parameters:
        ----------
        random_state: internal random state
        """
        self.random_state = random_state
        self.time_series_split = TimeSeriesSplit()
        pass

    def train_test_time_split(
        self, model_data: pd.DataFrame, test_year_start="2021", combined=False
    ) -> (
        tuple[pd.DataFrame, pd.DataFrame]
        | tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
    ):
        """
        Split data for modelling into train and test.
        Data from 2021 and onward is used as test set.

        Parameters:
        ----------
        model_data: dataframe to train models on
        test_year_start: year to use as starting point for test set
        combined: bool (True: keep train and test as full dataframes, not X and y)

        return:

            tuple[Dataframe, Dataframe] or tuple[DataFrame, DataFrame, Series, Series]
        """
        train_data: pd.DataFrame = model_data.drop(
            model_data.loc[test_year_start:].index, inplace=False
        )
        test_data: pd.DataFrame = model_data.loc[test_year_start:]

        if combined:
            return train_data, test_data

        train_X: pd.DataFrame = train_data.drop("Total Trafikkmengde", axis=1)
        test_X: pd.DataFrame = test_data.drop("Total Trafikkmengde", axis=1)
        train_y: pd.Series = train_data["Total Trafikkmengde"]
        test_y: pd.Series = test_data["Total Trafikkmengde"]

        return train_X, test_X, train_y, test_y

    def cross_validate(self, X, y, estimator: Pipeline, parameters=None) -> tuple:
        """
        Cross validates a given model using TimeSeriesSplit
        and returns that model trained on the whole training set.

        Parameters:
        X: feature columns
        y: target column
        estimator: estimator to use in grid search
        parameters: model parameters to search

        return: best model, best score, validation time
        """
        gscv = GridSearchCV(
            estimator,
            parameters,
            cv=self.time_series_split,
            verbose=3,
            scoring="neg_root_mean_squared_error",
            n_jobs=10,
        )
        start_time = time.time()
        gscv.fit(X, y)
        end_time = time.time()
        best_model = gscv.best_estimator_
        best_score = -gscv.best_score_
        best_params = gscv.best_params_
        validation_time = end_time - start_time
        print(f"Mean score of best estimator: {best_score}")
        print(f"With parameters: {best_params}\n")
        print(f"Validation time: {validation_time}")
        return best_model, best_score, validation_time

    def evaluate_best_model(self, X, y) -> tuple[Pipeline, float]:
        """
        Runs cross validation on each model.
        Chooses the model with lowest rmse.

        Parameters:
        ----------
        X: feature columns
        y: target column

        return: model, validation score
        """
        models = {
            "Dummy": self.dummy_regressor(X, y),
            # "KNN": self.k_neighbors(X, y),
            "RandomForest": self.random_forest(X, y),
            "MLPR": self.mlp(X, y),
            "SVR": self.support_vector(X, y),
            # "ElasticNet": self.elastic_net(X, y),
        }
        [
            print(f"name: {name},\t\trmse: {val[1]},\t\tval_time: {val[2]}")
            for name, val in models.items()
        ]
        best_model_key = min(models, key=lambda x: models[x][1])
        model = models[best_model_key][0]
        val_score = models[best_model_key][1]
        return model, val_score

    def save_model(self, model: Pipeline, model_name: str) -> None:
        """
        Save model to file with specified name

        Parameters:
        ----------
        model: model to be saved
        model_name: name of saved model
        """
        with open(f"models/{model_name}.pickle", "wb") as f:
            pickle.dump(model, file=f)

    def load_trained_model(self, model_name: str) -> Pipeline:
        """
        Loads trained model by name from models folder

        Parameters:
        ----------
        model_name: name of saved model (exclude file extension)

        return: model
        """
        with open(f"models/{model_name}.pickle", "rb") as f:
            model = pickle.load(f)

        return model

    def support_vector(self, X, y) -> tuple:
        """
        Create a pipeline and run cross validation with:
        Support Vector Machine Regressor

        Parameters:
        ----------
        X: feature columns
        y: target column

        return: best model, best score, validation time
        """
        parameters = [
            {
                "model__kernel": ["linear", "rbf"],
                "model__C": [1, 10, 0.1, 0.01],
            },
            {
                "model__kernel": ["poly"],
                "model__degree": [3, 4, 5],
                "model__C": [1, 10, 0.1, 0.01],
            },
        ]

        model = SVR()
        pipe = self.create_pipe(model)

        print("Running svr")

        return self.cross_validate(X, y, pipe, parameters)

    def k_neighbors(self, X, y) -> tuple:
        """
        Create a pipeline and run cross validation with:
        KNearest Neighbors Regressor

        Parameters:
        ----------
        X: feature columns
        y: target column

        return: best model, best score, validation time
        """
        parameters = {
            "model__n_neighbors": range(1, 102, 2),
            "model__weights": ["uniform", "distance"],
        }

        model = KNeighborsRegressor()
        pipe = self.create_pipe(model)
        print("Running KNN Regressor")
        return self.cross_validate(X, y, pipe, parameters)

    def random_forest(self, X, y) -> tuple:
        """
        Create a pipeline and run cross validation with:
        Random Forest Regressor

        Parameters:
        ----------
        X: feature columns
        y: target column

        return: best model, best score, validation time
        """
        parameters = {
            "model__n_estimators": [100, 150, 200, 250],
            # "max_features": ["sqrt", "log2", 1],
            # "min_samples_split": [1.0, 2, 5, 10],
        }
        model = RandomForestRegressor(random_state=self.random_state)
        pipe = self.create_pipe(model)

        print("Running RandomForestRegressor")
        return self.cross_validate(X, y, pipe, parameters)

    def mlp(self, X, y) -> tuple:
        """
        Create a pipeline and run cross validation with:
        Multi Layer Perceptron Regresor

        Parameters:
        ----------
        X: feature columns
        y: target column

        return: best model, best score, validation time
        """
        parameters = {
            "model__hidden_layer_sizes": [(50,), (100,), (50, 50), (30, 30, 10)],
            "model__alpha": [0.001, 0.01],
            "model__activation": ["tanh", "relu"],
        }
        model = MLPRegressor(
            random_state=self.random_state, shuffle=False, max_iter=2000
        )
        pipe = self.create_pipe(model)

        print("Running MLPRegressor")
        return self.cross_validate(X, y, pipe, parameters)

    def elastic_net(self, X, y) -> tuple:
        """
        Create a pipeline and run cross validation with:
        Multi Layer Perceptron Regressor

        Parameters:
        ----------
        X: feature columns
        y: target column

        return: best model, best score, validation time
        """
        parameters = {
            "model__l1_ratio": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "model__selection": ["cyclic", "random"],
        }
        model = ElasticNet(random_state=self.random_state)
        pipe = self.create_pipe(model)

        print("Running Elastic net")
        return self.cross_validate(X, y, pipe, parameters)

    def dummy_regressor(self, X, y) -> tuple:
        """
        Create a pipeline and run cross validation with:
        Dummy regressor

        Parameters:
        ----------
        X: feature columns
        y: target column

        return: best model, best score, validation time
        """
        parameters = {"model__strategy": ["mean", "median"]}

        model = DummyRegressor()
        pipe = self.create_pipe(model)
        print("Running Dummy Regressor")

        return self.cross_validate(X, y, pipe, parameters)

    def create_pipe(self, model) -> Pipeline:
        """
        Creates a pipeline with the given model.
        Uses KNNimputer and StandardScaler

        Parameters:
        ----------
        model: model

        return: ready pipeline
        """
        pipe = Pipeline(
            [
                ("imputer", KNNImputer()),
                ("scaler", StandardScaler()),
                ("model", model),
            ]
        )
        return pipe
