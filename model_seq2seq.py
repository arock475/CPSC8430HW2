import pickle

import torch.optim as optim
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import os
import json
import re
from torch.utils.data import DataLoader, Dataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def preprocess():
    path = './MLDS_hw2_1_data/'
    with open(path + 'training_label.json', 'r') as f:
        file = json.load(f)

    words = {}
    for d in file:
        for s in d['caption']:
            ws = re.sub('[.!,;?]', ' ', s).split()
            for word in ws:
                word = word.replace('.', '') if '.' in word else word
                if word in words:
                    words[word] += 1
                else:
                    words[word] = 1
    other_tok = [('<PAD>', 0), ('<SOS>', 1), ('<EOS>', 2), ('<UNK>', 3)]
    indexToWord = {index + len(other_tok): word for index, word in enumerate(words)}
    wordToIndex = {word: index + len(other_tok) for index, word in enumerate(words)}
    for token, index in other_tok:
        indexToWord[index] = token
        wordToIndex[token] = index

    return indexToWord, wordToIndex, words


def sentence_split(sentence, word_dict, wordToIndex):
    sentence = re.sub(r'[.!,;?]', ' ', sentence).split()
    for i in range(len(sentence)):
        if sentence[i] not in word_dict:
            sentence[i] = 3
        else:
            sentence[i] = wordToIndex[sentence[i]]
    sentence.insert(0, 1)
    sentence.append(2)
    return sentence


def get_labels(labels, word_dict, wordToIndex):
    label_json = './MLDS_hw2_1_data/' + labels
    labs = []
    with open(label_json, 'r') as f:
        label = json.load(f)
    for d in label:
        for s in d['caption']:
            s = sentence_split(s, word_dict, wordToIndex)
            labs.append((d['id'], s))
    return labs


def get_features(features):
    feats = {}
    training_feats = './MLDS_hw2_1_data/' + features
    files = os.listdir(training_feats)
    i = 0
    for file in files:
        print(i)
        i += 1
        value = np.load(os.path.join(training_feats, file))
        feats[file.split('.npy')[0]] = value
    return feats


def collate(data):
    data.sort(key=lambda x: len(x[1]), reverse=True)
    feats, captions = zip(*data)
    feats = torch.stack(feats, 0)

    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    return feats, targets, lengths


class training_data(Dataset):
    def __init__(self, features, labels, word_dict, wordToIndex):
        self.features = features
        self.labels = labels
        self.word_dict = word_dict
        self.feats = get_features(features)
        self.wordToIndex = wordToIndex
        self.data_pair = get_labels(labels, word_dict, wordToIndex)

    def __len__(self):
        return len(self.data_pair)

    def __getitem__(self, idx):
        assert (idx < self.__len__())
        feat_file, sentence = self.data_pair[idx]
        data = torch.Tensor(self.feats[feat_file])
        data += torch.Tensor(data.size()).random_(0, 2000) / 10000.
        return torch.Tensor(data), torch.Tensor(sentence)


class test_data(Dataset):
    def __init__(self, test_data_path):
        self.feats = []
        files = os.listdir(test_data_path)
        for file in files:
            key = file.split('.npy')[0]
            value = np.load(os.path.join(test_data_path, file))
            self.feats.append([key, value])

    def __len__(self):
        return len(self.feats)

    def __getitem__(self, idx):
        return self.feats[idx]


class attention(nn.Module):
    def __init__(self, hidden_size):
        super(attention, self).__init__()

        self.hidden_size = hidden_size
        self.linear1 = nn.Linear(2 * hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, hidden_size)
        self.to_weight = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden_state, encoder_outputs):
        batch_size, seq_len, feat_n = encoder_outputs.size()
        hidden_state = hidden_state.view(batch_size, 1, feat_n).repeat(1, seq_len, 1)
        matching_inputs = torch.cat((encoder_outputs, hidden_state), 2).view(-1, 2 * self.hidden_size)

        x = self.linear1(matching_inputs)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.linear4(x)
        attention_weights = self.to_weight(x)
        attention_weights = attention_weights.view(batch_size, seq_len)
        attention_weights = F.softmax(attention_weights, dim=1)
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)

        return context


class encoderRNN(nn.Module):
    def __init__(self):
        super(encoderRNN, self).__init__()

        self.compress = nn.Linear(4096, 512)
        self.dropout = nn.Dropout(0.35)
        self.lstm = nn.LSTM(512, 512, batch_first=True)

    def forward(self, input):
        batch_size, seq_len, feat_n = input.size()
        input = input.view(-1, feat_n)
        input = self.compress(input)
        input = self.dropout(input)
        input = input.view(batch_size, seq_len, 512)

        output, t = self.lstm(input)
        hidden_state, context = t[0], t[1]
        return output, hidden_state


class decoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, vocab_size, word_dim):
        super(decoderRNN, self).__init__()

        self.hidden_size = 512
        self.output_size = output_size
        self.vocab_size = vocab_size
        self.word_dim = word_dim

        self.embedding = nn.Embedding(output_size, 1024)
        self.dropout = nn.Dropout(0.3)
        self.lstm = nn.LSTM(hidden_size + word_dim, hidden_size, batch_first=True)
        self.attention = attention(hidden_size)
        self.to_final_output = nn.Linear(hidden_size, output_size)

    def forward(self, encoder_last_hidden_state, encoder_output, targets=None):
        _, batch_size, _ = encoder_last_hidden_state.size()

        decoder_current_hidden_state = None if encoder_last_hidden_state is None else encoder_last_hidden_state
        decoder_current_hidden_state = decoder_current_hidden_state.to(device)
        decoder_cxt = torch.zeros(decoder_current_hidden_state.size())
        decoder_cxt = decoder_cxt.to(device)
        decoder_current_input_word = Variable(torch.ones(batch_size, 1)).long()
        decoder_current_input_word = decoder_current_input_word.to(device)
        seq_logProb = []
        seq_predictions = []

        targets = self.embedding(targets)
        _, seq_len, _ = targets.size()

        for i in range(seq_len - 1):
            current_input_word = self.embedding(decoder_current_input_word).squeeze(1)

            context = self.attention(decoder_current_hidden_state, encoder_output)
            lstm_input = torch.cat([current_input_word, context], dim=1).unsqueeze(1).to(device)
            lstm_output, hidden_output = self.lstm(lstm_input, (decoder_current_hidden_state, decoder_cxt))
            decoder_current_hidden_state = hidden_output[0]
            logprob = self.to_final_output(lstm_output.squeeze(1))
            seq_logProb.append(logprob.unsqueeze(1))
            decoder_current_input_word = logprob.unsqueeze(1).max(2)[1]

        seq_logProb = torch.cat(seq_logProb, dim=1)
        seq_predictions = seq_logProb.max(2)[1]
        return seq_logProb, seq_predictions

    def infer(self, encoder_last_hidden_state, encoder_output):
        _, batch_size, _ = encoder_last_hidden_state.size()
        decoder_current_hidden_state = None if encoder_last_hidden_state is None else encoder_last_hidden_state
        decoder_current_input_word = Variable(torch.ones(batch_size, 1)).long()
        decoder_current_input_word = decoder_current_input_word.to(device)
        decoder_c = torch.zeros(decoder_current_hidden_state.size())
        seq_logProb = []
        seq_predictions = []
        assumption_seq_len = 28

        for i in range(assumption_seq_len - 1):
            current_input_word = self.embedding(decoder_current_input_word).squeeze(1)
            context = self.attention(decoder_current_hidden_state, encoder_output)
            lstm_input = torch.cat([current_input_word, context], dim=1).unsqueeze(1)
            lstm_input = lstm_input.to(device)
            decoder_current_hidden_state = decoder_current_hidden_state.to(device)
            decoder_c = decoder_c.to(device)
            lstm_output, hidden_out = self.lstm(lstm_input, (decoder_current_hidden_state, decoder_c))
            decoder_current_hidden_state = hidden_out[0]
            logprob = self.to_final_output(lstm_output.squeeze(1))
            seq_logProb.append(logprob.unsqueeze(1))
            decoder_current_input_word = logprob.unsqueeze(1).max(2)[1]

        seq_logProb = torch.cat(seq_logProb, dim=1)
        seq_predictions = seq_logProb.max(2)[1]
        return seq_logProb, seq_predictions


class MODELS(nn.Module):
    def __init__(self, encoder, decoder):
        super(MODELS, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, feats, mode, target_sentences=None, tr_steps=None):
        seq_logProb = None
        seq_predictions = None
        encoder_outputs, encoder_last_hidden_state = self.encoder(feats)
        encoder_outputs, encoder_last_hidden_state = encoder_outputs.to(device), encoder_last_hidden_state.to(device)
        if mode == 'train':
            seq_logProb, seq_predictions = self.decoder(encoder_last_hidden_state=encoder_last_hidden_state, encoder_output=encoder_outputs, targets=target_sentences)
        elif mode == 'inference':
            seq_logProb, seq_predictions = self.decoder.infer(encoder_last_hidden_state=encoder_last_hidden_state, encoder_output=encoder_outputs)
        return seq_logProb, seq_predictions


def calculate_loss(loss_fn, x, y, lengths):
    batch_size = len(x)
    predict_cat = None
    groundT_cat = None
    flag = True

    for batch in range(batch_size):
        predict = x[batch]
        ground_truth = y[batch]
        seq_len = lengths[batch] - 1

        predict = predict[:seq_len]
        ground_truth = ground_truth[:seq_len]
        if flag:
            predict_cat = predict
            groundT_cat = ground_truth
            flag = False
        else:
            predict_cat = torch.cat((predict_cat, predict), dim=0)
            groundT_cat = torch.cat((groundT_cat, ground_truth), dim=0)

    loss = loss_fn(predict_cat, groundT_cat)

    return loss


def train(model, epoch, loss_fn, parameters, optimizer, train_loader):
    model.train()
    print(epoch)

    for batch_idx, batch in enumerate(train_loader):
        feats, ground_truths, lengths = batch
        feats, ground_truths = feats.to(device), ground_truths.to(device)
        feats, ground_truths = Variable(feats), Variable(ground_truths)

        optimizer.zero_grad()
        seq_logProb, seq_predictions = model(feats, target_sentences=ground_truths, mode='train', tr_steps=epoch)
        ground_truths = ground_truths[:, 1:]
        loss = calculate_loss(loss_fn, seq_logProb, ground_truths, lengths)
        loss.backward()
        optimizer.step()


def test(test_loader, model, indexToWord):
    model.eval()
    sentence = []

    for batch_idx, batch in enumerate(test_loader):
        index, feats = batch
        feats = feats.to(device)
        index, feats = index, Variable(feats).float()

        seq_logProb, seq_predictions = model(feats, mode='inference')
        test_predictions = seq_predictions

        result = [[indexToWord[x.item()] if indexToWord[x.item()] != '<UNK>' else 'something' for x in s] for s in test_predictions]
        result = [' '.join(s).split('<EOS>')[0] for s in result]
        results = zip(index, result)
        for r in results:
            sentence.append(r)
    return sentence


def main():
    indexToWord, wordToIndex, word_dict = preprocess()
    with open('indexToWord.pickle', 'wb') as handle:
        pickle.dump(indexToWord, handle, protocol=pickle.HIGHEST_PROTOCOL)
    features = '/training_data/feat'
    labels = 'training_label.json'
    train_dataset = training_data(features, labels, word_dict, wordToIndex)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True, num_workers=1,
                                  collate_fn=collate)

    epochs = 50

    encoder = encoderRNN()
    decoder = decoderRNN(512, len(indexToWord) + 4, len(indexToWord) + 4, 1024)
    model = MODELS(encoder=encoder, decoder=decoder)

    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    parameters = model.parameters()
    optimizer = optim.Adam(parameters, lr=0.0001)

    for epoch in range(epochs):
        train(model, epoch + 1, loss_fn, parameters, optimizer, train_dataloader)

    torch.save(model, "{}/{}.h5".format('SavedModel', 'model3'))
    print("Training Completed")

    # create test output for test data
    #torch.load('./SavedModel/model2.h5')
    test_features_filepath = './MLDS_hw2_1_data/testing_data/feat'
    dataset = test_data(test_features_filepath)
    testing_loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=1)

    model = model.to(device)
    test_preds = test(testing_loader, model, indexToWord)

    with open('outputFinal.txt', 'w') as f:
        for id, pred in test_preds:
            f.write('{},{}\n'.format(id, pred))


if __name__ == "__main__":
    main()
