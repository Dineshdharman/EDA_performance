import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Load the dataset
df = pd.read_excel(r'C:\Users\HP\Desktop\task 1\student\student-mat.xlsx')

# Basic dataset checks
print(df.head())
print(df.shape)  # Fixed: Removed incorrect `df.shape()`
print(df.info())
print(df.nunique())
print(df.describe(include='all'))
print(df.isnull().sum())  # Checking for missing values

# Encode categorical variables using a more efficient method
categorical_cols = ['sex', 'school', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob',
                    'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities',
                    'nursery', 'higher', 'internet', 'romantic']

for col in categorical_cols:
    df[col] = df[col].astype('category').cat.codes  # Faster than LabelEncoder()

# Correlation Analysis
correlation = df.corr().abs()  # Used absolute correlation for ranking
grade_cols = ['G1', 'G2', 'G3']
all_correlations = []

for column in df.columns:
    if column not in grade_cols:
        avg_corr = (df[column].corr(df['G1']) +
                    df[column].corr(df['G2']) +
                    df[column].corr(df['G3'])) / 3
        all_correlations.append({'Feature': column, 'Avg_Correlation': abs(avg_corr)})

# Create DataFrame and sort by correlation
corr_df = pd.DataFrame(all_correlations).sort_values('Avg_Correlation', ascending=False)

print("\nFeatures ranked by correlation strength with grades:")
for idx, row in corr_df.iterrows():
    print(f"{row['Feature']}: {row['Avg_Correlation']:.3f}")

# Multicollinearity Check using VIF
X = df.drop(columns=['G1', 'G2', 'G3'])  # Exclude target variables
vif_data = pd.DataFrame()
vif_data['Feature'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

print("\nVariance Inflation Factor (VIF) to check multicollinearity:")
print(vif_data)

# Top features based on correlation with grades
top_features = ['failures', 'Medu', 'higher', 'Fedu', 'goout', 'schoolsup', 
                'studytime', 'age', 'traveltime', 'reason']

# Correlation Heatmap for Top Features
plt.figure(figsize=(12, 8))
sns.heatmap(df[top_features + ['G1', 'G2', 'G3']].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Top Features with Grades')
plt.show()

# Scatter Plots for Numerical Features vs G3
numerical_features = ['failures', 'Medu', 'Fedu', 'age', 'traveltime', 'studytime']
for feature in numerical_features:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=df[feature], y=df['G3'])
    plt.title(f'Scatter Plot of {feature} vs G3')
    plt.xlabel(feature)
    plt.ylabel('G3')
    plt.show()

# Box Plots for Categorical Features vs G3
categorical_features = ['higher', 'schoolsup', 'reason', 'goout']
for feature in categorical_features:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df[feature], y=df['G3'])
    plt.title(f'Box Plot of {feature} vs G3')
    plt.xlabel(feature)
    plt.ylabel('G3')
    plt.show()

# Distribution of Target Variable (G3)
plt.figure(figsize=(8, 5))
sns.histplot(df['G3'], bins=20, kde=True, color='blue')
plt.title("Distribution of Final Grade (G3)")
plt.xlabel("G3 Score")
plt.ylabel("Frequency")
plt.show()
