from classes.DataCleaning import DataCleaning
from classes.DataExploration import DataExploration
from classes.ModelEvaluation import ModelEvaluation
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error
import numpy as np


clean_data = DataCleaning()

df = clean_data.get_dataset("raw_data")

data2023 = df.loc["2023"]
training_data = df.drop(df.loc["2023"].index, inplace=False)


me = ModelEvaluation()

X_train, X_test, y_train, y_test = me.train_test_time_split(training_data)

Ximputer = KNNImputer()
yimputer = KNNImputer()
trans_X = Ximputer.fit_transform(X_train)
trans_y = yimputer.fit_transform(np.reshape(y_train, (-1, 1)))

model, _ = me.k_neighbors(trans_X, np.ravel(trans_y), save_to_file=False)
print(_)

# model = me.load_trained_model("KNNR")


X_test = Ximputer.transform(X_test)

prediction = model.predict(X_test)

score = np.sqrt(mean_squared_error(y_test, prediction))


# print(model)
print(score)
