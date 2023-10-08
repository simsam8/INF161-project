import pandas as pd


class ModelEvaluation:
    def __init__(self) -> None:
        # self.train_X, self.train_y, _, _ = self.train_test_time_split(model_data)
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
        combined: bool (True: further split train and test into X and y)

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
