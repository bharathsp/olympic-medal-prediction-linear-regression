import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
import pickle
import warnings
warnings.filterwarnings("ignore")

# Load the dataset containing athlete information and the NOC regions
athletes = pd.read_csv("athlete_events.csv")
noc = pd.read_csv("noc_regions.csv")

# Explore the dataset
athletes.head()

noc=noc[['NOC','region']]
noc.head()

athletes.info()

# Display statistical summary
athletes.describe()

# Check for missing values
athletes.isna().sum()

# Drop rows with missing values in the athletes dataset
athletes = athletes.dropna()

# Drop rows with missing values in the NOC dataset
noc = noc.dropna()

# Impute missing values in Age, Height, Weight columns with their respective means
imputer = SimpleImputer(strategy='mean')
athletes['Age'] = imputer.fit_transform(athletes[['Age']])
athletes['Height'] = imputer.fit_transform(athletes[['Height']])
athletes['Weight'] = imputer.fit_transform(athletes[['Weight']])

# Fill missing values in the Medal column with 'None'
athletes['Medal'].fillna('None', inplace=True)

# Merge the athletes dataset with the NOC regions dataset
athletes = athletes.merge(noc, on='NOC', how='left')

# Replace the 'NOC' column with 'region'
athletes['NOC'] = athletes['region']
athletes = athletes.drop(columns=['region'])

# Combine 'Team' and 'NOC' columns into 'Combined_Team'
athletes['Combined_Team'] = athletes['NOC'].fillna(athletes['Team'])
athletes = athletes.drop(columns=['Team', 'NOC'])

# Standardize the text data in columns: 'Combined_Team', 'City', 'Sport', 'Event'
athletes['Combined_Team'] = athletes['Combined_Team'].str.strip()
athletes['City'] = athletes['City'].str.strip()
athletes['Sport'] = athletes['Sport'].str.strip()
athletes['Event'] = athletes['Event'].str.strip()
athletes.head()

# Remove duplicate rows
athletes.drop_duplicates(inplace=True)

# Remove remaining duplicates based on specific columns
athletes = athletes.drop_duplicates(subset=['Name', 'Sex', 'Combined_Team', 'Event'], keep='first')

# Plotting box plots for Age, Height, Weight
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
sns.boxplot(x=athletes['Age'])
plt.title('Box Plot of Age')

plt.subplot(1, 3, 2)
sns.boxplot(x=athletes['Height'])
plt.title('Box Plot of Height')

plt.subplot(1, 3, 3)
sns.boxplot(x=athletes['Weight'])
plt.title('Box Plot of Weight')

plt.show()

# Plot histograms for Age, Height, Weight
athletes[['Age', 'Height', 'Weight']].hist(figsize=(15, 5))
plt.show()

# Use Z-scores to identify outliers
athletes['Age_Zscore'] = zscore(athletes['Age'])
athletes['Height_Zscore'] = zscore(athletes['Height'])
athletes['Weight_Zscore'] = zscore(athletes['Weight'])

# Remove outliers based on Z-score
athletes = athletes[(athletes['Age_Zscore'].abs() <= 3) & 
                    (athletes['Height_Zscore'].abs() <= 3) & 
                    (athletes['Weight_Zscore'].abs() <= 3)]

# Drop the Z-score columns
athletes = athletes.drop(columns=['Age_Zscore', 'Height_Zscore', 'Weight_Zscore'])

# Re-check for unrealistic values
print(athletes[(athletes['Age'] < 10) | (athletes['Age'] > 100)])
print(athletes[(athletes['Height'] < 100) | (athletes['Height'] > 250)])
print(athletes[(athletes['Weight'] < 30) | (athletes['Weight'] > 200)])

# Check unique values for each categorical column
print("\n\nUnique values in 'Sex':\n", athletes['Sex'].unique())
print("\n\nUnique values in 'Combined_Team':\n", athletes['Combined_Team'].unique())
print("\n\nUnique values in 'Medal':\n", athletes['Medal'].unique())
print("\n\nUnique values in 'Season':\n", athletes['Season'].unique())
print("\n\nUnique values in 'City':\n", athletes['City'].unique())
print("\n\nUnique values in 'Sport':\n", athletes['Sport'].unique())

# Display the frequency distribution for each categorical column
print(athletes['Sex'].value_counts())
print(athletes['Combined_Team'].value_counts())
print(athletes['Medal'].value_counts())
print(athletes['Season'].value_counts())
print(athletes['Year'].value_counts())
print(athletes['Sport'].value_counts())

# One-Hot Encoding for 'Sex' column
athletes = pd.get_dummies(athletes, columns=['Sex'])
athletes[['Sex_F', 'Sex_M']] = athletes[['Sex_F', 'Sex_M']].astype(int)

# Grouping data by Year and Combined_Team
grouped_df = athletes.groupby(['Year', 'Combined_Team']).agg({
    'Age': 'mean',
    'Height': 'mean',
    'Weight': 'mean',
    'Event': pd.Series.nunique,
    'Medal': 'count',
    'Sex_F': 'sum',
    'Sex_M': 'sum'
}).reset_index()

# Create a column for previous medal count
grouped_df = grouped_df.sort_values(by=['Combined_Team', 'Year'])
grouped_df['Prev_Medal'] = grouped_df.groupby('Combined_Team')['Medal'].shift(1).fillna(0)
grouped_df.rename(columns={'Combined_Team': 'Team'}, inplace=True)

grouped_df.tail(10)

# Calculate the correlation matrix
correlation_matrix = grouped_df[['Age', 'Weight', 'Height', 'Event', 'Medal', 
                                 'Sex_F', 'Sex_M', 'Prev_Medal']].corrwith(grouped_df['Medal'])

# Convert to DataFrame for easy plotting
correlation_df = correlation_matrix.to_frame().reset_index()
correlation_df.columns = ['Feature', 'Correlation with Medal']

# Plot the correlations using seaborn
plt.figure(figsize=(10, 6))
sns.barplot(data=correlation_df, x='Correlation with Medal', y='Feature', palette='coolwarm', orient='h')
plt.title('Correlation between Features and Medal')
plt.xlabel('Correlation')
plt.ylabel('Features')
plt.show()

# List of features to plot
features = ['Age', 'Weight', 'Height', 'Event', 'Sex_F', 'Sex_M', 'Prev_Medal']

# Create a grid of plots
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(18, 12))
axes = axes.flatten()  # Flatten the 2D array of axes to iterate easily

# Generate plots
for i, feature in enumerate(features):
    sns.regplot(x=feature, y='Medal', data=grouped_df, ax=axes[i], ci=None)
    axes[i].set_title(f'{feature} vs Medal')
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel('Medal')

# Remove unused subplot (if any)
for j in range(len(features), len(axes)):
    fig.delaxes(axes[j])

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()

# Drop columns with low correlation with the target variable
grouped_df = grouped_df[['Year', 'Team', 'Event', 'Sex_F', 'Sex_M', 'Prev_Medal', 'Medal']]
grouped_df.head()

# Prepare features and target variable
X = grouped_df.drop(columns=['Year', 'Team', 'Medal'])
y = grouped_df['Medal']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)
y_pred

# Adding predictions column to test set
test_set_with_predictions = pd.DataFrame(X_test, columns=X.columns)
test_set_with_predictions['Actual_Medal'] = y_test.values
test_set_with_predictions['Predicted_Medal'] = y_pred
test_set_with_predictions.reset_index(drop=True, inplace=True)
test_set_with_predictions.head(20)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Absolute Error: {mae:.2f}')
print(f'R-squared: {r2:.2f}')

# Model coefficients to understand feature importance
coefficients = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_})
coefficients

# Save the model and scaler to a pickle file
with open('olympic_medal_predictor.pkl', 'wb') as file:
    pickle.dump({'model': model, 'scaler': scaler}, file)

print("Model and scaler saved to 'olympic_medal_predictor.pkl'")

# Load the model and scaler from the pickle file
with open('olympic_medal_predictor.pkl', 'rb') as file:
    data = pickle.load(file)
    loaded_model = data['model']
    loaded_scaler = data['scaler']

# Function to get user input and make prediction
def get_user_input():
    event = int(input("Enter Number of Events: "))
    sex_f = int(input("Enter Number of Female Athletes: "))
    sex_m = int(input("Enter Number of Male Athletes: "))
    prev_medal = float(input("Enter Number of Previous Medals: "))

    # Create a DataFrame with the user input
    input_data = pd.DataFrame({
        'Event': [event],
        'Sex_F': [sex_f],
        'Sex_M': [sex_m],
        'Prev_Medal': [prev_medal]
    })

    # Preprocess input data (e.g., scaling)
    input_data_scaled = loaded_scaler.transform(input_data)

    # Make prediction using the loaded model
    prediction = loaded_model.predict(input_data_scaled)

    return prediction[0]

# Get prediction from user input
predicted_medal = get_user_input()
print(f'\nPredicted Medals: {predicted_medal:.0f}')

