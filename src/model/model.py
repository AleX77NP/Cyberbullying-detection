from torch import nn
import torch


# Custom Neural Network for detection & classification
class TweetClassificationModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_categories, p_dropout=0.2):
        super(TweetClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embedding_dim, sparse=False)
        self.dropout = nn.Dropout(p_dropout)
        self.fc = nn.Linear(embedding_dim, num_categories)
        self.__init_weights()

    def __init_weights(self):
        init_range = 0.5
        self.embedding.weight.data.uniform_(-init_range, init_range)
        self.fc.weight.data.uniform_(-init_range, init_range)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        out = self.dropout(embedded)
        return self.fc(out)


# Evaluate model on evaluation data
def evaluate_model(model, eval_dataloader, criterion):
    model.eval()
    total_val_acc, total_val_count = 0, 0

    with torch.no_grad():
        for _, (label, text, offsets) in enumerate(eval_dataloader):
            predicted_label = model(text, offsets)
            loss = criterion(predicted_label, label)
            total_val_acc += (predicted_label.argmax(1) == label).sum().item()
            total_val_count += label.size(0)
    return total_val_acc / total_val_count


# Predict category for new tweet
def predict(model, text, text_pipeline):
    with torch.no_grad():
        text = torch.tensor(text_pipeline(text))
        output = model(text, torch.tensor([0]))
        return output.argmax(1).item()
