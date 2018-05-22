'''
Decision tree basic
'''

import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# Using pandas to get familiar with your data

# save filepath to variable for easier access
melb_file_path = '/home/insam/09_data/melb_data/melb_data.csv'

# read the data and store data in Dataframe titled melb
melb_data = pd.read_csv(melb_file_path)
# print a summary
#print(melb_data.describe())

# Selection and filetring data
#print(melb_data.columns)

# Selecting a single column
melb_price_Data = melb_data.Price
#print(melb_price_Data)

# SElecting a multiple columns
#columns_of_interest = ['Landsize','BuildingArea']
#two_columns_of_data = melb_data[columns_of_interest]
#print(two_columns_of_data.describe())

# Preprocessing
melb_data.fillna(melb_data.mean(), inplace = True)


# Choosing the prediction target
y = melb_data.Price

# Choosing predictors
melb_predictors = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea',
                        'YearBuilt', 'Lattitude', 'Longtitude']
X = melb_data[melb_predictors]


# Building your model

# Define model
melb_model = DecisionTreeRegressor()

# Fit model
melb_model.fit(X, y)

'''
print("making predictions for the following 5 houses")
print(X.head())
print("\n\nThe predictions are")
print(melb_model.predict(X.head()))
'''

# Model Validation
predictied_home_prices = melb_model.predict(X)
mean_absolute = mean_absolute_error(y, predictied_home_prices)
print(mean_absolute)


# Split data into train, valid
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)
# Define model
melb_model_new = DecisionTreeRegressor()
# Fit model
melb_model_new.fit(train_X,train_y)
# Get predicted prices on validation data
val_predictions = melb_model_new.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))

# Compare different models
