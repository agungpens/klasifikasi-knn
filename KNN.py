import pandas as pd

# Load dataset
dataset = pd.read_csv('milk.csv')

# Split dataset into train_data and train_label
train_data = dataset.drop('Grade', axis=1)
train_label = dataset['Grade']

# Show train_data and train_label
print(train_data)
print(train_label)

from sklearn.preprocessing import MinMaxScaler

# Normalize train_data using MinMaxScaler
scaler = MinMaxScaler()
train_data_normalized = scaler.fit_transform(train_data)

# Show normalized train_data
print(train_data_normalized)


from sklearn.neighbors import KNeighborsClassifier

# Define input data test
test_data = [[7, 50, 1, 1, 1, 0, 245]]

# Normalize test_data using MinMaxScaler
test_data_normalized = scaler.transform(test_data)

# Define k values
k_values = [1, 2, 3, 4, 5, 6, 7]

# Perform k-NN classification for each k value
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_data_normalized, train_label)
    prediction = knn.predict(test_data_normalized)
    print('k =', k, '->', prediction)
