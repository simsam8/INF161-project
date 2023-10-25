import pandas as pd
import numpy as np


class FeatureEngineering:
    """
    Class for feature engineering
    """
    def __init__(self) -> None:
        pass

    def encode_cyclical(
        self, df: pd.DataFrame, col: str, max_val: int
    ) -> pd.DataFrame:
        """
        Encodes a cyclical feature.

        Parameters:
        ----------
        data: DataFrame
        col: feature column
        max_val: max value of cycle

        return: modified DataFrame
        """
        df[col + "_sin"] = np.sin(2 * np.pi * df[col] / max_val)
        df[col + "_cos"] = np.cos(2 * np.pi * df[col] / max_val)
        df = df.drop(col, axis=1)
        return df

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
