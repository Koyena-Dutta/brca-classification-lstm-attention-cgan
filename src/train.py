from sklearn.model_selection import train_test_split

def prepare_lstm_data(X, y):
    X = X.reshape((X.shape[0], 1, X.shape[1]))
    return train_test_split(X, y, test_size=0.2, random_state=42)