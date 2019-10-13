import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import  metrics

data = pd.read_csv('./iris.csv')
data.columns = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']

data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# values = data['petal-length']
#
# plt.figure(figsize=(7, 4))
# plt.tight_layout()
# plt.hist(values, bins=[1, 2, 3, 4], rwidth=0.98)


data_x = data.iloc[:, :3]   # the x values sepal-length and sepal-width. All rows, column 1 and column 2
data_y = data.iloc[:, 3]    # the y values : petal-length


scaler = StandardScaler()
scaler.fit(data_x)
data_x = scaler.transform(data_x)


# get the training set and test set
train_x, test_x,  train_y, test_y = train_test_split(data_x, data_y, test_size=0.33, shuffle=False)

lr = LinearRegression()     # create a linear regression model
lr.fit(train_x, train_y)

#   lr.intercept_   # get the intercept
#   lr.coef_        # get coefficients

predicted_y = lr.predict(test_x)

# print the actual and predicated side by side
df = pd.DataFrame({'Actual': np.array(test_y), 'Predicted': predicted_y})
print(df)

# plot a barchart of the values side by side
df.plot(figsize=(15, 7), kind='bar')
# plt.show() # shows the bar graph


meanAbsError = metrics.mean_absolute_error(np.array(test_y), predicted_y)
meanSqrdError = metrics.mean_squared_error(np.array(test_y), predicted_y)
rootMeanSqrdError = np.sqrt(meanSqrdError)


print('Mean absolute error = ', meanAbsError)
print('Mean squared error = ', meanSqrdError)
print('Root mean squared error = ', rootMeanSqrdError)
