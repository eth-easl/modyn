from typing import Any

from torch import nn


class FastText:
    def __init__(self, model_configuration: dict[str, Any], device: str, amp: bool) -> None:
        self.model = TextClassifier(model_configuration)
        self.model.to(device)


class TextClassifier(nn.Module):
    # pylint: disable=too-many-instance-attributes
    def __init__(self, model_configuration: dict[str, Any]):
        super(TextClassifier, self).__init__()
        vocab_size = model_configuration["vocab_size"]
        embedding_dim = model_configuration["embedding_dim"]
        hidden_dim = model_configuration["hidden_dim"]
        num_class = model_configuration["num_class"]
        word_embeddings = model_configuration["word_embeddings"]

        self.embedding = nn.EmbeddingBag(vocab_size, embedding_dim, sparse=True)
        self.embedding.weights = nn.Parameter(word_embeddings, requires_grad=False)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_class)
        self.softmax = nn.Softmax()

    def forward(self, data):
        embedded_sent = self.embedding(data)
        x = self.fc1(embedded_sent)
        x = self.fc2(x)
        x = self.softmax(x)
        return x
