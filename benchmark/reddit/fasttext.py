import torch
import torch.nn as nn

# TODO(#28): Integrate into Modyn: Inherit from trainer_server.internal.PytorchTrainer
class TextClassifier(nn.Module):
	def __init__(self, vocab_size, embed_dim, hidden_dim, num_class, word_embeddings):
		super(TextClassifier, self).__init__()
		self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
		self.embedding.weights = nn.Parameter(word_embeddings, requires_grad=False)
		self.fc1 = nn.Linear(embed_dim, hidden_dim)
		self.fc2 = nn.Linear(hidden_dim, num_class)
		self.softmax = nn.Softmax()

	def forward(self, x):
		embedded_sent = self.embedding(x)
		x = self.fc1(embedded_sent)
		x = self.fc2(x)
		x = self.softmax(x)
		return x