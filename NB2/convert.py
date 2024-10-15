import pandas as pd

# Define the file path to your dataset
file_path = 'adm_data.csv'

# Read the dataset
df = pd.read_csv(file_path)

# Binarize 'Chance of Admit' column (threshold: 0.5)
df['Chance of Admit '] = df['Chance of Admit '].apply(lambda x: 1 if x >= 0.5 else 0)

# Save the updated dataset back to the CSV file
df.to_csv(file_path, index=False)

print(f"Updated dataset saved to: {file_path}")
