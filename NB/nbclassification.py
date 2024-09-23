import pandas as pd
from collections import defaultdict

data = pd.read_csv('bank_dataset.csv')

X = data[['age', 'job', 'marital_status', 'balance']]
y = data['term_deposit']

X.loc[:, 'job'] = X['job'].astype('category')
X.loc[:, 'marital_status'] = X['marital_status'].astype('category')

def calculate_prior(y):
    return y.value_counts(normalize=True).to_dict()

def calculate_likelihoods(X, y):
    likelihoods = defaultdict(lambda: defaultdict(lambda: {'yes': 0.0, 'no': 0.0}))
    for column in X.columns:
        for feature_value in X[column].unique():
            prob_yes = len(X[(X[column] == feature_value) & (y == 'yes')]) / len(y[y == 'yes']) if len(y[y == 'yes']) > 0 else 0.0
            prob_no = len(X[(X[column] == feature_value) & (y == 'no')]) / len(y[y == 'no']) if len(y[y == 'no']) > 0 else 0.0
            likelihoods[column][feature_value] = {'yes': prob_yes, 'no': prob_no}
    return likelihoods

def predict_naive_bayes(priors, likelihoods, X_new):
    prob_yes = priors['yes']
    prob_no = priors['no']
    for feature, value in X_new.items():
        prob_yes *= likelihoods[feature][value]['yes']
        prob_no *= likelihoods[feature][value]['no']
    return 'yes' if prob_yes > prob_no else 'no'

priors = calculate_prior(y)
likelihoods = calculate_likelihoods(X, y)

job_options = ['admin', 'blue-collar', 'entrepreneur', 'retired', 'student', 'technician']
marital_status_options = ['single', 'married', 'divorced']

def get_new_customer_info():
    print("Select your job from the following options:")
    for idx, job in enumerate(job_options, start=1):
        print(f"{idx}. {job}")
    
    job_choice = int(input("Enter the number corresponding to your job: ")) - 1
    job = job_options[job_choice]

    print("\nSelect your marital status from the following options:")
    for idx, status in enumerate(marital_status_options, start=1):
        print(f"{idx}. {status}")

    marital_status_choice = int(input("Enter the number corresponding to your marital status: ")) - 1
    marital_status = marital_status_options[marital_status_choice]

    age = int(input("Enter your age: "))
    balance = float(input("Enter your balance: "))

    return {'age': age, 'job': job, 'marital_status': marital_status, 'balance': balance}

new_customer = get_new_customer_info()

prediction = predict_naive_bayes(priors, likelihoods, new_customer)
print(f"Predicted class for the new customer: {prediction}")
