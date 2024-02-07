import tensorflow as tf
from sklearn.model_selection import train_test_split

class TOSModel():
    def __init__(self, documents):
        self.model = None
        self.vocab_size = None
        self.documents = documents
        self.num_epochs = 10
        self.batch_size = 32
        self.embedding_dim = 128

    def update_input_dim(self):
        for document in self.documents:
            tokens = document.tokens
            self.vocab_size = max(self.vocab_size, len(document.vocab))
            

    def create_model(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Embedding(input_dim=self.vocab_size, output_dim=embedding_dim),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        self.model.compile(optimizer="adam", loss="mse", metrics=['accuracy'])
    
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train, epochs=self.num_epochs, batch_size=self.batch_size)
    
    def evaluate(self, X_test, y_test):
        loss, accuracy = self.model.evaluate(X_test, y_test)
        print(f"Test Loss: {loss}")
        print(f"Test Accuracy: {accuracy}")

    def predict(self, documents: Document):
        tokenized_documents = []
        for document in documents:
            tokenized_documents.append(document.tokenize())
        

