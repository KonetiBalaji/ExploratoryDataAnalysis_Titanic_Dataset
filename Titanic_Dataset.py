# Titanic_Dataset.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv('./dataset/Titanic-Dataset.csv')

# Check initial data info and missing values
print(data.info())
print(data.isnull().sum())

# Data Cleaning
data['Age'] = data['Age'].fillna(data['Age'].median())
data.drop('Cabin', axis=1, inplace=True)
data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])

# Check and remove duplicates
print("Duplicates:", data.duplicated().sum())
data.drop_duplicates(inplace=True)

# Verify no missing values remain
print(data.isnull().sum())
print(data.shape)

# Statistical Summary
print(data.describe())

# Correlation Analysis (numeric columns)
print(data.select_dtypes(include=['number']).corr())

# Survival Rates by Gender and Class
print(data.groupby('Sex')['Survived'].mean())
print(data.groupby('Pclass')['Survived'].mean())

# Visualizations
sns.set(style='whitegrid')

# Survival Count by Gender
plt.figure()
sns.countplot(x='Sex', hue='Survived', data=data)
plt.title('Survival Count by Gender')
plt.savefig('./images/survival_gender.png')
plt.show()

# Survival Count by Passenger Class
plt.figure()
sns.countplot(x='Pclass', hue='Survived', data=data)
plt.title('Survival Count by Passenger Class')
plt.savefig('./images/survival_class.png')
plt.show()

# Age Distribution by Survival
plt.figure()
sns.histplot(data=data, x='Age', hue='Survived', bins=30, kde=True)
plt.title('Age Distribution by Survival')
plt.savefig('./images/age_distribution.png')
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(data.select_dtypes(include=['number']).corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.savefig('images/correlation_heatmap.png')
plt.show()

# Feature Engineering
data['Title'] = data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
data['Title'] = data['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
data['Title'] = data['Title'].replace('Mlle', 'Miss')
data['Title'] = data['Title'].replace('Ms', 'Miss')
data['Title'] = data['Title'].replace('Mme', 'Mrs')
data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
data['IsAlone'] = (data['FamilySize'] == 1).astype(int)

# Dynamically determine number of bins and labels for AgeBin
age_bins = pd.qcut(data['Age'], 5, retbins=True, duplicates='drop')[1]
age_labels = ['Very Young', 'Young', 'Middle', 'Senior', 'Very Senior'][:len(age_bins)-1]
data['AgeBin'] = pd.qcut(data['Age'], len(age_bins)-1, labels=age_labels, duplicates='drop')

data['FareBin'] = pd.qcut(data['Fare'], 5, labels=['Low', 'Low-Medium', 'Medium', 'Medium-High', 'High'])

# Advanced Visualizations
plt.figure(figsize=(12, 6))
sns.countplot(x='Title', hue='Survived', data=data)
plt.title('Survival Count by Title')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('./images/survival_title.png')
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(x='FamilySize', hue='Survived', data=data)
plt.title('Survival Count by Family Size')
plt.savefig('./images/survival_family_size.png')
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(x='AgeBin', hue='Survived', data=data)
plt.title('Survival Count by Age Bin')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('./images/survival_age_bin.png')
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(x='FareBin', hue='Survived', data=data)
plt.title('Survival Count by Fare Bin')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('./images/survival_fare_bin.png')
plt.show()

# Machine Learning Model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Prepare features for modeling
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Title', 'FamilySize', 'IsAlone']
X = pd.get_dummies(data[features])
y = data['Survived']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Feature Importance
feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
feature_importance = feature_importance.sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importance')
plt.tight_layout()
plt.savefig('./images/feature_importance.png')
plt.show()
