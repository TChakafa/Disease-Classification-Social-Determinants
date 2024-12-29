#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report


# In[2]:


# Load the dataset
data = pd.read_csv('Health.csv')


# In[3]:


# Define features and targets
X = data[['Age', 'Educational Level', 'Sex', 'Housing Stability', 'Water Quality', 'Air Quality', 'Access to Primary Care']]
y = data[['Disease', 'Risk Level']]


# In[4]:


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[5]:


# Preprocessing for categorical variables
categorical_features = ['Educational Level', 'Sex', 'Housing Stability', 'Water Quality', 'Air Quality', 'Access to Primary Care']
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_features)
    ],
    remainder='passthrough'  # Keep numerical features as is
)


# In[6]:


# Create a pipeline with preprocessing and model
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', MultiOutputClassifier(RandomForestClassifier(random_state=42)))
])


# In[7]:


# Fit the model
pipeline.fit(X_train, y_train)


# In[11]:


# Function to classify user input
def classify_user_input():
    print("Please enter the following information:")
    
    # Collect user input
    age = float(input("Age: "))
    educational_level = input("Educational Level (Not Applicable, Primary, Secondary, Tertiary): ")
    sex = input("Sex (Male or Female): ")
    housing_stability = input("Housing Stability (Stable or Unstable): ")
    water_quality = input("Water Quality (Poor, Fair, Good): ")
    air_quality = input("Air Quality (Poor, Fair, Good): ")
    access_to_primary_care = input("Access to Primary Care (Yes or No): ")

    # Create a DataFrame from the input
    user_input = pd.DataFrame({
        'Age': [age],
        'Educational Level': [educational_level],
        'Sex': [sex],
        'Housing Stability': [housing_stability],
        'Water Quality': [water_quality],
        'Air Quality': [air_quality],
        'Access to Primary Care': [access_to_primary_care]
    })

    # Make predictions
    predictions = pipeline.predict(user_input)
    
    # Output the predictions
    disease, risk_level = predictions[0]
    print(f"Predicted Disease: {disease}, Predicted Risk Level: {risk_level}")

# Call the function to classify user input
classify_user_input()


# In[10]:


# Evaluate the model on the test set (optional)
#predictions = pipeline.predict(X_test)
#print(classification_report(y_test, predictions, target_names=['Disease', 'Risk Level']))


# In[13]:


def plot_variable_contributions(data):
    variable_counts = data['Disease'].value_counts()
    plt.figure(figsize=(8, 5))
    plt.pie(variable_counts, labels=variable_counts.index, autopct='%1.1f%%', startangle=90)
    plt.title('Variable Contributions to Disease Causation')
    plt.axis('equal')  # Equal aspect ratio ensures that pie chart is circular.
    plt.show()

plot_variable_contributions(data)


# In[14]:


def plot_risk_levels(data):
    risk_counts = data['Risk Level'].value_counts()
    colors = {'Low': 'green', 'Medium': 'yellow', 'High': 'red'}
    plt.figure(figsize=(8, 5))
    plt.pie(risk_counts, labels=risk_counts.index, colors=[colors[risk] for risk in risk_counts.index], autopct='%1.1f%%', startangle=90)
    plt.title('Classification Risk Levels')
    plt.axis('equal')
    plt.show()

plot_risk_levels(data)


# In[15]:


def plot_correlation_heatmap(data):
    # Convert categorical variables to numerical for correlation analysis
    numerical_data = pd.get_dummies(data, drop_first=True)
    plt.figure(figsize=(10, 8))
    correlation_matrix = numerical_data.corr()
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, linewidths=0.5)
    plt.title('Correlation Analysis Heatmap')
    plt.show()

plot_correlation_heatmap(data)


# In[16]:


import joblib

# After training your model
joblib.dump(pipeline, 'model.joblib')  # Save the model


# In[ ]:




