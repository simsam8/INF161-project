import matplotlib.pyplot as plt
import seaborn as sb
import plotly.express as px
import pandas as pd


class DataExploration:
    """
    Class for data exploration
    """
    def __init__(self) -> None:
        pass

    def correlation_matrix(self, df: pd.DataFrame) -> None:
        """
        Plots correlation matrix of dataset

        Parameters:
        ----------
        df: dataframe
        """
        sb.heatmap(data=df.corr())
        plt.tight_layout()
        plt.show()

    def traffic_by_day(self, df: pd.DataFrame) -> None:
        """
        Total traffic for each weekday

        Parameters:
        ----------
        df: dataframe
        """
        fig = px.histogram(
            df,
            x="day",
            y="Total Trafikkmengde",
            barmode="group",
            histfunc="avg",
            orientation="v",
        )
        fig.show()

    def hourly_traffick_by_day(self, df: pd.DataFrame) -> None:
        """
        Total traffic for each hour of the day

        Parameters:
        ----------
        df: dataframe
        """
        fig = px.histogram(
            df,
            x="hour",
            y="Total Trafikkmengde",
            barmode="group",
            histfunc="avg",
            orientation="v",
        )
        fig.show()

    def monthly_traffick_by_year(self, df: pd.DataFrame) -> None:
        """
        Total traffic every month, shows for each year

        Parameters:
        ----------
        df: dataframe
        """
        fig = px.histogram(
            df,
            x="month",
            y="Total Trafikkmengde",
            barmode="group",
            orientation="v",
            histfunc="avg",
            color="year",
        )

        fig.show()

    def traffic_by_year(self, df: pd.DataFrame) -> None:
        """
        Total amount of traffic each year

        Parameters:
        ----------
        df: dataframe
        """
        fig = px.histogram(
            df,
            x="year",
            y="Total Trafikkmengde",
            barmode="group",
            orientation="v",
            histfunc="avg",
        )
        fig.show()

    def scatter_traffic_temperature(self, df: pd.DataFrame) -> None:
        """
        Scatter plot of temperature and traffic amount

        Parameters:
        ----------
        df: dataframe
        """
        fig = px.scatter(
            df,
            y="Total Trafikkmengde",
            x="Lufttemperatur",
        )
        fig.update_layout(autotypenumbers="convert types")
        fig.show()

    def scatter_traffic_wind(self, df: pd.DataFrame) -> None:
        """
        Scatter plot of wind strength and traffic amount

        Parameters:
        ----------
        df: dataframe
        """
        fig = px.scatter(
            df,
            y="Total Trafikkmengde",
            x="Vindstyrke",
        )
        fig.update_layout(autotypenumbers="convert types")
        fig.show()
