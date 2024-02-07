from sklearn.model_selection import train_test_split as train_tests

class DataLoader:
    def __init__(self, documents, scores, test_size=0.2, random_state=None):
        self.documents = documents
        self.scores = scores
        self.test_size = test_size
        self.random_state = random_state

    def load_data(self):
        return self.documents, self.scores

    def train_test_splits(self):
        X_train, X_test, y_train, y_test = train_tests(self.documents, self.scores, test_size=self.test_size, random_state=self.random_state)
        return X_train, X_test, y_train, y_test
