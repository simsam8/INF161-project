import pandas as pd


class FeatureEngineering:
    """
    Class for feature engineering
    """

    def __init__(self) -> None:
        pass

    def remove_columns(self, columns: list[str], df: pd.DataFrame):
        """
        Removes given columns from dataframe.

        Parameters:
        ----------
        columns: columns to drop
        df: dataframe

        return: dataframe
        """
        return df.drop(columns, axis=1)

    def create_time_based_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates time based features from datetime index

        Parameters:
        ----------
        df: dataframe

        return: DataFrame
        """
        df["hour"] = df.index.hour
        df["day"] = df.index.dayofweek
        df["month"] = df.index.month
        df["year"] = df.index.year
        return df
