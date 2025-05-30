import numpy as np
import pandas as pd

admissions = pd.read_csv('data_1.csv')

# Make dummy variables for rank
data = pd.concat([admissions, pd.get_dummies(admissions['rank'], prefix='rank')], axis=1)
data = data.drop('rank', axis=1)

# Standarize features
for field in ['gre', 'gpa']:
    mean, std = data[field].mean(), data[field].std()
    data.loc[:, field] = (data[field] - mean) / std
    
# Split off random 10% of the data for testing
np.random.seed(21)
sample = np.random.choice(data.index, size=int(len(data) * 0.9), replace=False)
data, test_data = data.iloc[sample], data.drop(sample)

# Split into features and targets
# features, targets = data.drop('admit', axis=1), data['admit']
# features_test, targets_test = test_data.drop('admit', axis=1), test_data['admit']

# At the end of data_prep.py, convert to NumPy arrays explicitly
features = np.array(data.drop('admit', axis=1))
targets = np.array(data['admit'])
features_test = np.array(test_data.drop('admit', axis=1))
targets_test = np.array(test_data['admit'])