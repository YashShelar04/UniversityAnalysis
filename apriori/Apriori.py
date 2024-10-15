import numpy as np
import pandas as pd
from apyori import apriori

# Load the dataset (assuming you have the data in a CSV file)
store_data = pd.read_csv('apriori.csv')

# Take only the first 50 rows
store_data = store_data.head(50)

# Display the first few rows of the dataset (optional)
print(store_data.head(10))

# Get the column names (which are the subject names)
subject_names = list(store_data.columns)

# Preprocess the data for Apriori (converting DataFrame to a list of lists)
# Each row will be converted into a list of subjects where the value is 1
records = []
for index, row in store_data.iterrows():
    transaction = [subject_names[i] for i in range(len(row)) if row.iloc[i] == 1]  # Using iloc for positional access
    records.append(transaction)

# Applying the Apriori algorithm with desired parameters
association_rules = apriori(records, min_support=0.0045, min_confidence=0.2, min_lift=3, min_length=2)
association_results = list(association_rules)

# Print the number of rules generated
print(f"Total number of association rules: {len(association_results)}")

# Loop through the association results to extract and print rules and their metrics
for item in association_results:
    # First index of the inner list contains the base item and add item
    pair = item[0]
    items = [x for x in pair]
    print(f"Rule: {items[0]} -> {items[1]}")

    # Second index of the inner list contains support
    print(f"Support: {item[1]}")

    # Third index contains the confidence and lift
    print(f"Confidence: {item[2][0][2]}")
    print(f"Lift: {item[2][0][3]}")
    print("=====================================")
