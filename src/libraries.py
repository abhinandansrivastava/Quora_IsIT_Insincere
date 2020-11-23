import torch
import torch.optim as optim
from torchtext import data
import spacy
nlp = spacy.load('en_core_web_sm')

SEED = 42
torch.manual_seed(SEED)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def binary_accuracy(preds, y):
    # round predictions to the closest integer
    rounded_preds = torch.round(preds)

    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc


def predict(sentence):
    global model,nlp
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]  #tokenize the sentence
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]          #convert to integer sequence
    length = [len(indexed)]                                    #compute no. of words
    tensor = torch.LongTensor(indexed).to(device)              #convert to tensor
    tensor = tensor.unsqueeze(1).T                             #reshape in form of batch,no. of words
    length_tensor = torch.LongTensor(length)                   #convert to tensor
    prediction = model(tensor, length_tensor)                  #prediction
    return prediction.item()
