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

    def correlation_matrix(self, dataframe) -> None:
        """
        Plots correlation matrix of dataset
        """
        sb.heatmap(data=dataframe.corr())
        plt.tight_layout()
        plt.show()

    def traffic_by_day(self, dataframe: pd.DataFrame) -> None:
        """
        Total traffic for each weekday
        """
        fig = px.histogram(
            dataframe,
            x="day",
            y="Total Trafikkmengde",
            barmode="group",
            histfunc="sum",
            orientation="v",
        )
        fig.show()

    def hourly_traffick_by_day(self, dataframe: pd.DataFrame) -> None:
        """
        Total traffic for each hour of the day
        """
        fig = px.histogram(
            dataframe,
            x="hour",
            y="Total Trafikkmengde",
            barmode="group",
            histfunc="sum",
            orientation="v",
        )
        fig.show()

    def monthly_traffick_by_year(self, dataframe: pd.DataFrame) -> None:
        """
        Total traffic every month, shows for each year
        """
        fig = px.histogram(
            dataframe,
            x="month",
            y="Total Trafikkmengde",
            barmode="group",
            orientation="v",
            color="year",
        )

        fig.show()

    def traffic_by_year(self, dataframe: pd.DataFrame) -> None:
        """
        Total amount of traffic each year
        """
        fig = px.histogram(
            dataframe,
            x="year",
            y="Total Trafikkmengde",
            barmode="group",
            orientation="v",
        )
        fig.show()

    def scatter_traffic_temperature(self, dataframe: pd.DataFrame) -> None:
        """
        Scatter plot of temperature and traffic amount
        """
        fig = px.scatter(
            dataframe,
            y="Total Trafikkmengde",
            x="Lufttemperatur",
        )
        fig.update_layout(autotypenumbers="convert types")
        fig.show()

    def scatter_traffic_wind(self, dataframe: pd.DataFrame) -> None:
        """
        Scatter plot of wind strength and traffic amount
        """
        fig = px.scatter(
            dataframe,
            y="Total Trafikkmengde",
            x="Vindstyrke",
        )
        fig.update_layout(autotypenumbers="convert types")
        fig.show()
