from classes.DataCleaning import DataCleaning
from classes.ModelEvaluation import ModelEvaluation
from classes.FeatureEngineering import FeatureEngineering
from sklearn.metrics import mean_squared_error
import pandas as pd


seed = 100

# Get clean dataset
clean_data = DataCleaning()
df = clean_data.get_dataset("raw_data")

# Feature engineering if neccesary
fe = FeatureEngineering()
drop_columns = ["year"]
df = fe.remove_columns(drop_columns, df)
# df = fe.encode_cyclical(df, "hour", 23)
# df = fe.encode_cyclical(df, "month", 12)
# df = fe.encode_cyclical(df, "day", 6)

# Divide modelling data and prediction data
data2023 = df.loc["2023"]
training_data = df.drop(df.loc["2023"].index, inplace=False)
print(training_data.columns)

# drop rows with nan values in Total Trafikkmengde
training_data = training_data[training_data["Total Trafikkmengde"].notna()]


me = ModelEvaluation()

# Split into train and test
X_train, X_test, y_train, y_test = me.train_test_time_split(training_data)

# only when testing code
# -- remove later --
# X_train = X_train.loc["2020"]
# y_train = y_train.loc["2020"]

print(f"train shape: {X_train.shape}")
print(f"test shape: {X_test.shape}")

# Choose the best model
model, val_rmse = me.evaluate_best_model(X_train, y_train)

print(f"Best model validation rmse: {val_rmse}")


# Best model on test set
prediction = model.predict(X_test)
test_rmse = mean_squared_error(y_test, prediction, squared=False)
print(f"Best model test rmse: {test_rmse}")


me.save_model(model, "model")


# predict 2023 and save prediction
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
