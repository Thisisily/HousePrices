import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
# Setting the aesthetics for the plots
sns.set(style="whitegrid")

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Load the training dataset
train_file_path = '/kaggle/input/house-prices-advanced-regression-techniques/train.csv'
train_data = pd.read_csv(train_file_path)

# Display the first few rows of the training dataset
train_data.head()
# Checking the size of the dataset
dataset_shape = train_data.shape

# Basic statistical summary
statistical_summary = train_data.describe()

# Displaying dataset shape and statistical summary
dataset_shape, statistical_summary


# Identifying missing values
missing_values = train_data.isnull().sum()
missing_values = missing_values[missing_values > 0].sort_values(ascending=False)

# Display missing values
missing_values

# Percentage of missing values
missing_percentage = (train_data.isnull().sum() / len(train_data)) * 100
missing_data = pd.DataFrame({'Missing Ratio': missing_percentage})
missing_data = missing_data[missing_data['Missing Ratio'] > 0].sort_values(by='Missing Ratio', ascending=False)

# Drop columns with a high percentage of missing values (arbitrarily set at 80% for this analysis)
columns_to_drop = missing_data[missing_data['Missing Ratio'] > 80].index.tolist()
train_data_cleaned = train_data.drop(columns=columns_to_drop)

# Impute the rest of the missing values
# For simplicity, we'll impute numerical features with the median and categorical features with the mode
for column in missing_data.index:
    if column not in columns_to_drop:
        if train_data[column].dtype == 'object':
            train_data[column].fillna(train_data[column].mode()[0], inplace=True)
        else:
            train_data[column].fillna(train_data[column].median(), inplace=True)

# Checking the dataset after handling missing values
remaining_missing = train_data.isnull().sum()
remaining_missing = remaining_missing[remaining_missing > 0]

# Display columns dropped and any remaining missing values
columns_to_drop, remaining_missing

# Distribution of SalePrice
plt.figure(figsize=(10, 6))
sns.histplot(train_data['SalePrice'], kde=True)
plt.title('Distribution of SalePrice')
plt.xlabel('Sale Price')
plt.ylabel('Frequency')
plt.show()

# Filtering out only numerical columns for correlation calculation
numerical_data = train_data.select_dtypes(include=['int64', 'float64'])

# Calculating the correlation matrix on numerical data
corr_matrix_numerical = numerical_data.corr()

# One-hot encoding categorical variables
train_data_encoded = pd.get_dummies(train_data)

# Calculating the correlation matrix for the one-hot encoded data
corr_matrix_encoded = train_data_encoded.corr()

# Plotting the heatmap for the correlation matrix of the encoded dataset
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix_encoded, cmap='coolwarm')
plt.title('Correlation Matrix of Encoded Numerical Features')
plt.show()


# Displaying the top 10 features most correlated with SalePrice in the encoded data
top_corr_features_encoded = corr_matrix_encoded['SalePrice'].sort_values(ascending=False).head(11)
top_corr_features_encoded
# Extracting the top 10 features most correlated with SalePrice (excluding SalePrice itself)
top_features = corr_matrix_encoded['SalePrice'].sort_values(ascending=False).head(11)[1:]

# Creating a bar chart for these features
plt.figure(figsize=(10, 6))
sns.barplot(x=top_features.values, y=top_features.index)
plt.title('Top 10 Features Most Correlated with SalePrice')
plt.xlabel('Correlation Coefficient')
plt.ylabel('Features')
plt.show()



# Feature engineering: You can create new features or modify existing ones based on your insights
# Creating a new feature 'TotalSF' as a combination of basement, 1st and 2nd floor square feet
train_data['TotalSF'] = train_data['TotalBsmtSF'] + train_data['1stFlrSF'] + train_data['2ndFlrSF']

# Displaying the first few rows with the new feature
train_data[['TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'TotalSF']].head()

# One-hot encoding of categorical variables
train_data_encoded = pd.get_dummies(train_data)

# Display the first few rows of the encoded dataset
train_data_encoded.head()


from sklearn.preprocessing import StandardScaler

# Feature scaling
scaler = StandardScaler()

# Selecting numerical features to scale (excluding 'Id' and 'SalePrice')
numerical_features = train_data_encoded.select_dtypes(include=['number']).columns.drop(['Id', 'SalePrice'])

# Applying the scaler
train_data_encoded[numerical_features] = scaler.fit_transform(train_data_encoded[numerical_features])

# Display the first few rows of the scaled dataset
train_data_encoded[numerical_features].head()



from sklearn.model_selection import train_test_split

# Defining the target variable and features
X = train_data_encoded.drop(['SalePrice', 'Id'], axis=1)
y = np.log(train_data_encoded['SalePrice']) # Using the log transformation as per competition's evaluation metric

# Splitting the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Display the shape of the training and validation sets
X_train.shape, X_val.shape, y_train.shape, y_val.shape

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Initializing the Linear Regression model
lr_model = LinearRegression()

# Training the model
lr_model.fit(X_train, y_train)

# Predicting on the validation set
y_val_pred = lr_model.predict(X_val)

# Calculating RMSE on the validation set
rmse_val = np.sqrt(mean_squared_error(y_val, y_val_pred))

# Display the RMSE
rmse_val


from sklearn.ensemble import RandomForestRegressor

# Initializing the Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Training the model
rf_model.fit(X_train, y_train)

# Predicting on the validation set
y_val_pred_rf = rf_model.predict(X_val)

# Calculating RMSE on the validation set
rmse_val_rf = np.sqrt(mean_squared_error(y_val, y_val_pred_rf))

# Display the RMSE
rmse_val_rf

from sklearn.ensemble import GradientBoostingRegressor

# Initializing the Gradient Boosting Regressor
gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)

# Training the model
gb_model.fit(X_train, y_train)

# Predicting on the validation set
y_val_pred_gb = gb_model.predict(X_val)

# Calculating RMSE on the validation set
rmse_val_gb = np.sqrt(mean_squared_error(y_val, y_val_pred_gb))

# Display the RMSE
rmse_val_gb

from sklearn.model_selection import GridSearchCV

# Define the parameter grid for Gradient Boosting Regressor
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [3, 4, 5]
}

# Initializing the Gradient Boosting Regressor
gb_model_tuned = GradientBoostingRegressor(random_state=42)

# Grid Search with Cross-Validation
grid_search = GridSearchCV(estimator=gb_model_tuned, param_grid=param_grid, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')

# Fitting the grid search
grid_search.fit(X_train, y_train)

# Best parameters and score
best_params = grid_search.best_params_
best_score = np.sqrt(-grid_search.best_score_)

# Display the best parameters and corresponding score
best_params, best_score
# Retraining the Gradient Boosting Regressor with the optimized parameters
opt_gb_model = GradientBoostingRegressor(learning_rate=0.05, max_depth=3, n_estimators=300, random_state=42)

# Training the model
opt_gb_model.fit(X_train, y_train)

# Predicting on the validation set
y_val_pred_opt_gb = opt_gb_model.predict(X_val)

# Calculating RMSE on the validation set
rmse_val_opt_gb = np.sqrt(mean_squared_error(y_val, y_val_pred_opt_gb))

# Display the RMSE
rmse_val_opt_gb
