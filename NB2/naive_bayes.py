import pandas as pd
from collections import defaultdict


data = pd.read_csv('placement_data.csv')  

# Selecting features and target variable
X = data[['Department', 'Gender', 'CGPA', 'DSA', 'Projects']]
y = data['Placement'].apply(lambda x: 1 if x == 'Yes' else 0)  # Convert to binary (1 for Yes, 0 for No)

# Print the first 10 tuples of the dataset
print("First 10 records of the dataset:")
print(data.head(10))

# Function to calculate prior probabilities
def calculate_prior(y):
    return y.value_counts(normalize=True).to_dict()

# Function to calculate likelihoods
def calculate_likelihoods(X, y):
    likelihoods = defaultdict(lambda: defaultdict(lambda: {'yes': 0.0, 'no': 0.0}))
    
    for column in X.columns:
        for feature_value in X[column].unique():
            prob_yes = len(X[(X[column] == feature_value) & (y == 1)]) / len(y[y == 1]) if len(y[y == 1]) > 0 else 0.0
            prob_no = len(X[(X[column] == feature_value) & (y == 0)]) / len(y[y == 0]) if len(y[y == 0]) > 0 else 0.0
            likelihoods[column][feature_value] = {'yes': prob_yes, 'no': prob_no}
    
    return likelihoods

# Function to predict using Naive Bayes
def predict_naive_bayes(priors, likelihoods, X_new):
    prob_yes = priors[1]  # Prior probability for placement (Yes)
    prob_no = priors[0]  # Prior probability for placement (No)
    
    for feature, value in X_new.items():
        if value in likelihoods[feature]:  # Ensure the value exists in likelihoods
            prob_yes *= likelihoods[feature][value]['yes']
            prob_no *= likelihoods[feature][value]['no']
        else:
            prob_yes *= 0.0001  # Small smoothing factor for unseen values
            prob_no *= 0.0001
    
    return 'Yes' if prob_yes > prob_no else 'No'

# Calculate priors and likelihoods
priors = calculate_prior(y)
likelihoods = calculate_likelihoods(X, y)

# Collect new student information
def get_new_student_info():
    department = input("Enter Department (Comps, IT, AIDS, EXTC): ")
    gender = input("Enter Gender (Male/Female): ")
    cgpa = input("Enter CGPA (<8, >=8, >9.5): ")
    dsa = input("Enter DSA (yes/no): ")
    projects = input("Enter Projects (yes/no): ")
    
    return {
        'Department': department, 
        'Gender': gender, 
        'CGPA': cgpa, 
        'DSA': dsa, 
        'Projects': projects
    }

# Get new student info
new_student = get_new_student_info()

# Make prediction for new student
prediction = predict_naive_bayes(priors, likelihoods, new_student)
print(f"Predicted Placement status for you: {prediction}")
