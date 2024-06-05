import numpy as np
import pandas as pd
from google.colab import files
from sklearn.ensemble import RandomForestRegressor

#Read the train data
train = pd.read_csv('train.csv')

#Pull data into target (y)
train_y = train.SalePrice
predictor_columns = ['LotArea', 'OverallQual', 'YearBuilt', 'TotRmsAbvGrd']

#Create training predictors data
train_X = train[predictor_columns]

regressor = RandomForestRegressor()
regressor.fit(train_X, train_y)

#Read the test data
test = pd.read_csv('test.csv')
test_X = test[predictor_columns]

#Make predictions
predicted_prices = regressor.predict(test_X)
print(predicted_prices)

results = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})
results.to_csv('results.csv', index=False)

files.download("results.csv")
