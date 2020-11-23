import torch
from model import classifier
import random
from torchtext import data
import torch.optim as optim
import torch.nn as nn
import spacy
nlp = spacy.load('en_core_web_sm')


def predict(sentence):
    global model
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]  #tokenize the sentence
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]          #convert to integer sequence
    length = [len(indexed)]                                    #compute no. of words
    tensor = torch.LongTensor(indexed).to(device)              #convert to tensor
    tensor = tensor.unsqueeze(1).T                             #reshape in form of batch,no. of words
    length_tensor = torch.LongTensor(length)                   #convert to tensor
    prediction = model(tensor, length_tensor)                  #prediction
    return prediction.item()


SEED = 42
BATCH_SIZE = 64
torch.manual_seed(SEED)
embedding_dim = 100
num_hidden_nodes = 32
num_output_nodes = 1
num_layers = 2
bidirection = True
dropout = 0.2

TEXT = data.Field(tokenize = 'spacy',batch_first = True,include_lengths = True)
LABEL = data.LabelField(dtype = torch.float,batch_first = True)
fields = [(None, None), ('text',TEXT),('label', LABEL)]


training_data = data.TabularDataset(path = 'quora.csv',format = 'csv',
            fields = fields, skip_header = True)


train_data, valid_data = training_data.split(split_ratio = 0.8,
            random_state = random.seed(SEED))


TEXT.build_vocab(train_data,min_freq = 3, vectors = "glove.6B.100d")
size_of_vocab = len(TEXT.vocab)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model = classifier(size_of_vocab,embedding_dim,num_hidden_nodes,
                   num_output_nodes,num_layers,bidirectional = True,
                   dropout = dropout)

model = model.to(device)

path = '/home/grp-gpu-nc/DBS/Untitled Folder/EntityExtraction/BiLSTM/saved_weights.pt'
model.load_state_dict(torch.load(path))
model.eval()

data_list = ["Are there any sports that you don't like?",
             "Why Indian girls go crazy about marrying Shri. Rahul Gandhi ji?"]

for text_data in data_list:
    print(predict(text_data))
