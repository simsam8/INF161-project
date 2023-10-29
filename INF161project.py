from classes.DataCleaning import DataCleaning
from classes.ModelEvaluation import ModelEvaluation
from classes.FeatureEngineering import FeatureEngineering
from sklearn.metrics import mean_squared_error
import pandas as pd


seed = 100

# Vasker data
clean_data = DataCleaning()
df = clean_data.get_dataset("raw_data")

# Fjerner unødvendige kolonner
fe = FeatureEngineering()
drop_columns = ["year"]
df = fe.remove_columns(drop_columns, df)

# Deler opp i modellerings- og prediksjonsdata
data2023 = df.loc["2023"]
training_data = df.drop(df.loc["2023"].index, inplace=False)

# dropper rader med nan verdier i Total Trafikkmengde
training_data = training_data[training_data["Total Trafikkmengde"].notna()]


me = ModelEvaluation(random_state=seed)

# Deler inn i trening og test set
X_train, X_test, y_train, y_test = me.train_test_time_split(training_data)

print(f"train shape: {X_train.shape}")
print(f"test shape: {X_test.shape}")

# Velger beste modell
model, val_rmse = me.evaluate_best_model(X_train, y_train)

print(f"Best model: {model}")
print(f"Best model validation rmse: {val_rmse}")


# Tester beste modell
prediction = model.predict(X_test)
test_rmse = round(mean_squared_error(y_test, prediction, squared=False), 2)
print(f"Best model test rmse: {test_rmse}")

# Lagrer resultatene til log
with open("program_log.txt", "a") as f:
    f.write("\nFINAL MODEL\n\n")
    f.write(f"Model: {model}\n")
    f.write(f"Validation rmse: {val_rmse}\n")
    f.write(f"Test rmse: {test_rmse}")

me.save_model(model, "model")


# predikerer 2023 data og lagrer prediksjon
data2023 = data2023.drop("Total Trafikkmengde", axis=1)
final_prediction = model.predict(data2023)


index_2023 = data2023.index.to_frame(index=False)

prediksjon_col = pd.DataFrame(final_prediction, columns=["Prediksjon"])
prediksjon_col["Prediksjon"] = prediksjon_col["Prediksjon"].round()

index_2023["Dato"] = pd.to_datetime(index_2023["Datetime"]).dt.date
index_2023["Tid"] = pd.to_datetime(index_2023["Datetime"]).dt.time
index_2023 = index_2023.drop("Datetime", axis=1)
prediction_frame = pd.concat([index_2023, prediksjon_col], axis=1)

prediction_frame.to_csv("predictions.csv")
