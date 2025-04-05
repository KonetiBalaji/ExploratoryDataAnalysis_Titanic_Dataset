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
