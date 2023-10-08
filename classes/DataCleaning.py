import pandas as pd
import numpy as np
import os


class DataCleaning:
    """
    Class for preprocessing data
    """
    def __init__(self) -> None:
        pass

    def get_dataset(self, data_folder: str) -> pd.DataFrame:
        traffic_data = os.path.join(data_folder, "trafikkdata.csv")
        traffic_df = self.clean_traffic_data(traffic_data)
        weather_df = self.clean_weather_data(data_folder)
        combined = self.combine_data(traffic_df, weather_df)
        feature_engineered = self.create_features(combined)
        return feature_engineered

    def clean_traffic_data(self, filepath: str) -> pd.DataFrame:
        """
        Cleans traffic data.
        Removes unnecessary columns and sets datetime index.
        Creates a column with "Total Trafikkmengde"

        return: Dataframe
        """

        # Leser traffikkdata,
        # bruker midlertidige kolonne navn,
        # og bruker regex for flere seperatorer
        temp_col_names = [str(i) for i in range(24)]
        raw_df = pd.read_csv(
            filepath, names=temp_col_names, sep=r";|\|", engine="python"
        )

        # Setter faktiske kolonne navn
        raw_df.columns = raw_df.iloc[0]
        raw_df = raw_df.iloc[1:]

        # Fjerner unødvendige kolonner
        to_drop = [
            "Trafikkregistreringspunkt",
            "Navn",
            "Vegreferanse",
            "Fra",
            "Til",
            "Til tidspunkt",
            "Dekningsgrad (%)",
            "Antall timer total",
            "Antall timer inkludert",
            "Antall timer ugyldig",
            "Lengdekvalitetsgrad (%)",
            "Ikke gyldig lengde",
            "< 5,6m",
            ">= 5,6m",
            "5,6m - 7,6m",
            "7,6m - 12,5m",
            "12,5m - 16,0m",
            ">= 16,0m",
            "16,0m - 24,0m",
            ">= 24,0m",
        ]
        trafikk_df = raw_df.drop(columns=to_drop)

        # Henter rader med 'Totalt' i kolonne 'Felt' og fjerner resten.
        # Dropper 'Felt' kolonnen og lager kolonne for total trafikkmengde
        trafikk_df = trafikk_df.where(trafikk_df["Felt"] == "Totalt", inplace=False)
        trafikk_df = trafikk_df.drop(columns=["Felt"])
        trafikk_df = trafikk_df.rename(columns={"Trafikkmengde": "Total Trafikkmengde"})
        trafikk_df = trafikk_df[trafikk_df["Dato"].notna()]

        # Setter datatype
        trafikk_df["Total Trafikkmengde"] = trafikk_df["Total Trafikkmengde"].replace(
            "-", np.nan
        )

        # Lager en datetime kolonne
        trafikk_df["Datetime"] = pd.to_datetime(
            trafikk_df["Dato"].astype(str)
            + " "
            + trafikk_df["Fra tidspunkt"].astype(str)
        )

        # Dropper duplikater der klokken blir stilt tilbake
        trafikk_df = trafikk_df.drop_duplicates(["Datetime"], keep="first")

        # Dropper dato og tidspunkt
        # Setter Datetime kolonne som index til dataset
        trafikk_df = trafikk_df.drop(columns=["Dato", "Fra tidspunkt"])
        trafikk_df.set_index("Datetime", inplace=True)

        return trafikk_df

    def clean_weather_data(self, data_folder: str) -> pd.DataFrame:
        """
        Combines all weather data into one dataframe.
        Cleans and resamples data into 1H intervals.

        return: DataFrame
        """

        # Henter filsti til værdata
        work_dir = os.getcwd()
        data_dir = os.path.join(work_dir, data_folder)
        csv_files = [
            f"{data_folder}/{f}"
            for f in os.listdir(data_dir)
            if f.startswith("Florida")
        ]

        # Setter sammen til ett datasett
        data_frames = []
        for f in csv_files:
            data_frames.append(pd.read_csv(f))

        df = pd.concat(data_frames)

        # Kombinerer kolonnene Dato og Tid, og sorterer etter dato
        df["Datetime"] = pd.to_datetime(
            df["Dato"].astype(str) + " " + df["Tid"].astype(str)
        )
        df = df.drop(columns=["Dato", "Tid"])
        df.set_index("Datetime", inplace=True)
        df = df.sort_values(["Datetime"])

        # Setter manglende verdier til Nan
        df = df.replace(9999.99, np.nan)

        # Setter negative verdier til 0 i globalstråling
        df["Globalstraling"] = df["Globalstraling"].clip(lower=0)

        # Dropper kolonnen relativluftfuktighet
        # mesteparten av radene har manglende verdier
        # df["Relativ luftfuktighet"] = df["Relativ luftfuktighet"].replace("", np.nan)
        df = df.drop(columns=["Relativ luftfuktighet"])

        # Resampler værdata til 1t intervaller
        df_resampled = df.resample("H").mean()
        return df_resampled

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates new features from existing features

        return: DataFrame
        """
        df["hour"] = df.index.hour
        df["day"] = df.index.dayofweek
        df["month"] = df.index.month
        df["year"] = df.index.year
        return df

    def combine_data(
        self, trafikk_df: pd.DataFrame, weather_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Combines traffic and weather data

        return: DataFrame
        """

        df = weather_df.merge(trafikk_df, right_index=True, left_index=True)
        return df
