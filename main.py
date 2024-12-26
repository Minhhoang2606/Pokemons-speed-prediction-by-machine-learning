'''
Pokemon speed predictor using Linear Regression
Author: Henry Ha
'''
# Load the dataset and display the first few rows
import pandas as pd
pokemon_data = pd.read_csv("Pokemon.csv")
print(pokemon_data.head())

#TODO EDA

# Display general information and summary statistics
print(pokemon_data.info())
print(pokemon_data.describe())

import matplotlib.pyplot as plt
import seaborn as sns

# Plot histograms for key features
pokemon_data[['Speed', 'Attack', 'Defense']].hist(bins=20, figsize=(10, 6))
plt.suptitle("Distribution of Key Features")
plt.show()

# Scatter plot to visualize relationships
sns.scatterplot(data=pokemon_data, x='Attack', y='Speed', hue='Type 1', alpha=0.7)
plt.title("Relationship Between Attack and Speed")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()

# Correlation matrix and heatmap
correlation_matrix = pokemon_data.corr(numeric_only=True)
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
plt.title("Correlation Matrix")
plt.show()

# Boxplot for Speed by Type.1
plt.figure(figsize=(12, 6))
sns.boxplot(data=pokemon_data, x='Type 1', y='Speed', palette='Set2')
plt.title("Speed Distribution by Pokémon Type")
plt.xticks(rotation=45)
plt.show()

# Boxplot for Speed by Generation
sns.boxplot(data=pokemon_data, x='Generation', y='Speed', palette='Set3')
plt.title("Speed Distribution Across Generations")
plt.show()

# Boxplots for identifying outliers
pokemon_data[['Speed', 'Attack', 'Defense']].plot(kind='box', subplots=True, layout=(1, 3), figsize=(15, 5))
plt.suptitle("Boxplots to Identify Outliers")
plt.show()

#TODO Data preprocessing

# Handle missing values in the Type 2 column
pokemon_data['Type 2'].fillna('None', inplace=True)

from sklearn.preprocessing import StandardScaler

# Select numerical features for scaling
numerical_features = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']
scaler = StandardScaler()

# Scale the numerical features
pokemon_data[numerical_features] = scaler.fit_transform(pokemon_data[numerical_features])

# Perform one-hot encoding on categorical features
pokemon_data = pd.get_dummies(pokemon_data, columns=['Type 1', 'Type 2', 'Legendary'], drop_first=True)

# Drop the 'Name' column as it is not relevant for prediction
pokemon_data.drop(columns=['Name'], inplace=True)

# Separate the target variable and input features
X = pokemon_data.drop(columns=['Speed'])
y = pokemon_data['Speed']

#TODO Model development

from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LinearRegression

# Initialize the Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Predict speed for the training and testing sets
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

import joblib

# Save the trained model
joblib.dump(model, 'pokemon_speed_predictor.pkl')

#TODO Model evaluation

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Calculate evaluation metrics for the testing set
mae = mean_absolute_error(y_test, y_test_pred)
mse = mean_squared_error(y_test, y_test_pred)
r2 = r2_score(y_test, y_test_pred)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R²): {r2}")

import matplotlib.pyplot as plt

# Scatter plot of actual vs predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_test_pred, alpha=0.7, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='red')
plt.xlabel("Actual Speed")
plt.ylabel("Predicted Speed")
plt.title("Actual vs Predicted Speed")
plt.show()

# Plot residuals
residuals = y_test - y_test_pred
plt.figure(figsize=(8, 6))
plt.hist(residuals, bins=20, color='green', edgecolor='black')
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.title("Residuals Distribution")
plt.show()

from sklearn.model_selection import cross_val_score

# Perform 5-fold cross-validation
cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')

# Print cross-validation results
print(f"Cross-Validation R² Scores: {cv_scores}")
print(f"Mean R²: {cv_scores.mean():.4f}")
print(f"Standard Deviation of R²: {cv_scores.std():.4f}")

