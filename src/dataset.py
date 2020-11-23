import torch
import random
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
from model import classifier
from engine import train,evaluate
from torchtext import data
from libraries import count_parameters,binary_accuracy

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

# Load the Datasets

training_data = data.TabularDataset(path = 'quora.csv',format = 'csv',
            fields = fields, skip_header = True)

train_data, valid_data = training_data.split(split_ratio = 0.8,
            random_state = random.seed(SEED))


# initialize the embeddings
TEXT.build_vocab(train_data,min_freq = 3, vectors = "glove.6B.100d")
LABEL.build_vocab(train_data)


print("Size of TEXT Vocab:",len(TEXT.vocab))
print("Size of LABEL Vocab:",len(LABEL.vocab))

# check whether cuda is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load an iterator
train_iterator, valid_iterator = data.BucketIterator.splits(
    (train_data,valid_data),
    batch_size = BATCH_SIZE,
    sort_key = lambda x : len(x.text),
    sort_within_batch = True,
    device = device
)

size_of_vocab = len(TEXT.vocab)
model = classifier(size_of_vocab,embedding_dim,num_hidden_nodes,
                   num_output_nodes,num_layers,bidirectional = True,
                   dropout = dropout)

print(f'The model has {count_parameters(model):,} trainable parameters')

# Initialize the pretrained embedding
pretrained_embeddings = TEXT.vocab.vectors
model.embedding.weight.data.copy_(pretrained_embeddings)

print(pretrained_embeddings.shape)

# define optimizer and loss
optimizer = optim.Adam(model.parameters())
criterion = nn.BCELoss()

#push to cuda if available
model = model.to(device)
criterion = criterion.to(device)

N_EPOCHS = 10
best_valid_loss = float('inf')

for epoch in tqdm(range(N_EPOCHS)):

    # train the model
    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)

    # evaluate the model
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)

    # save the best model
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'saved_weights.pt')

    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')
