import datetime
import torch
import numpy as np

class CombinedModel:
    def __init__(self, model1, model2, model1_name, model2_name):
        """
        :param model1: The PyTorch model that generates embeddings for a given text.
        :param model2: The sklearn classifier model.
        :param model1_name: Name of the first model.
        :param model2_name: Name of the second model.
        """
        self.embedding_model = model1
        self.classifier_model = model2
        self.model_names = (model1_name, model2_name)
        self.creation_date = datetime.datetime.now().strftime('%Y-%m-%d')

    def top_predictions(self, text):
        """
        Returns the top 5 predictions and their probabilities for the input text.
        
        :param text: The input text to be processed.
        :return: Dictionary containing top 5 predictions and their probabilities.
        """
        embedding_numpy = self._get_embedding(text)
        probabilities = self.classifier_model.predict_proba(embedding_numpy)[0]
        
        # Get top 5 prediction indices
        top_indices = np.argsort(probabilities)[-5:][::-1]
        
        # Convert indices to class labels
        top_classes = self.classifier_model.classes_[top_indices]
        
        # Get corresponding probabilities
        top_probabilities = probabilities[top_indices]
        
        return dict(zip(top_classes, top_probabilities))

    def _get_embedding(self, text):
        input_tensor = self.text_to_tensor(text)
        self.embedding_model.eval()
        with torch.no_grad():
            embedding = self.embedding_model(input_tensor)
        return embedding.cpu().numpy()

    def text_to_tensor(self, text):
        """
        Convert text to tensor for PyTorch model input. You may need to implement this.
        
        :param text: The input text.
        :return: Tensor representation of the text.
        """
        pass

# Example usage:
# Assuming you've loaded model
