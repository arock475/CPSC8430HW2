from random import random

import torch.optim as optim
import time
import torch
import torch.nn as nn
import json
import pickle
import time
import re
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
from tqdm import trange

EPOCHS = 10
BATCH_SIZE = 128

import numpy as np
import torch


def synthetic_data(Nt=2000, tf=80 * np.pi):
    '''
    create synthetic time series dataset
    : param Nt:       number of time steps
    : param tf:       final time
    : return t, y:    time, feature arrays
    '''

    t = np.linspace(0., tf, Nt)
    y = np.sin(2. * t) + 0.5 * np.cos(t) + np.random.normal(0., 0.2, Nt)

    return t, y


def train_test_split(t, y, split=0.8):
    '''

    split time series into train/test sets

    : param t:                      time array
    : para y:                       feature array
    : para split:                   percent of data to include in training set
    : return t_train, y_train:      time/feature training and test sets;
    :        t_test, y_test:        (shape: [# samples, 1])

    '''

    indx_split = int(split * len(y))
    indx_train = np.arange(0, indx_split)
    indx_test = np.arange(indx_split, len(y))

    t_train = t[indx_train]
    y_train = y[indx_train]
    y_train = y_train.reshape(-1, 1)

    t_test = t[indx_test]
    y_test = y[indx_test]
    y_test = y_test.reshape(-1, 1)

    return t_train, y_train, t_test, y_test


def windowed_dataset(y, input_window=5, output_window=1, stride=1, num_features=1):
    '''
    create a windowed dataset

    : param y:                time series feature (array)
    : param input_window:     number of y samples to give model
    : param output_window:    number of future y samples to predict
    : param stide:            spacing between windows
    : param num_features:     number of features (i.e., 1 for us, but we could have multiple features)
    : return X, Y:            arrays with correct dimensions for LSTM
    :                         (i.e., [input/output window size # examples, # features])
    '''

    L = y.shape[0]
    num_samples = (L - input_window - output_window) // stride + 1

    X = np.zeros([input_window, num_samples, num_features])
    Y = np.zeros([output_window, num_samples, num_features])

    for ff in np.arange(num_features):
        for ii in np.arange(num_samples):
            start_x = stride * ii
            end_x = start_x + input_window
            X[:, ii, ff] = y[start_x:end_x, ff]

            start_y = stride * ii + input_window
            end_y = start_y + output_window
            Y[:, ii, ff] = y[start_y:end_y, ff]

    return X, Y


def numpy_to_torch(Xtrain, Ytrain, Xtest, Ytest):
    '''
    convert numpy array to PyTorch tensor

    : param Xtrain:                           windowed training input data (input window size, # examples, # features); np.array
    : param Ytrain:                           windowed training target data (output window size, # examples, # features); np.array
    : param Xtest:                            windowed test input data (input window size, # examples, # features); np.array
    : param Ytest:                            windowed test target data (output window size, # examples, # features); np.array
    : return X_train_torch, Y_train_torch,
    :        X_test_torch, Y_test_torch:      all input np.arrays converted to PyTorch tensors

    '''

    X_train_torch = torch.from_numpy(Xtrain).type(torch.Tensor)
    Y_train_torch = torch.from_numpy(Ytrain).type(torch.Tensor)

    X_test_torch = torch.from_numpy(Xtest).type(torch.Tensor)
    Y_test_torch = torch.from_numpy(Ytest).type(torch.Tensor)

    return X_train_torch, Y_train_torch, X_test_torch, Y_test_torch

def preprocess():
    with open("/Users/aaronspears/Documents/2024Spring/CPSC8430/HW2/MLDS_hw2_1_data/training_label.json", 'r') as f:
        data = json.load(f)

    word_dict = {}
    for d in data:
        for c in d["caption"]:
            sentence = re.sub(r'[^\w\s]', '', c).split()
            for word in sentence:
                if word in word_dict:
                    word_dict[word] += 1
                else:
                    word_dict[word] = 1
    RNN_tokens = [('<PAD>', 0), ('<SOS>', 1), ('<EOS>', 2), ('<UNK>', 3)]
    i2w = {i + len(RNN_tokens): w for i, w in enumerate(word_dict)}
    w2i = {w: i + len(RNN_tokens) for i, w in enumerate(word_dict)}
    for token, index in RNN_tokens:
        i2w[index] = token
        w2i[token] = index

    return i2w, w2i, word_dict

def getTrainingFeatures():
    feature_data = {}
    folder_path = "/Users/aaronspears/Documents/2024Spring/CPSC8430/HW2/MLDS_hw2_1_data/training_data/feat"
    files = os.listdir(folder_path)
    for file in files:
        value = np.load(os.path.join(folder_path, file))
        feature_data[file.split('.npy')[0]] = value
    return feature_data

def s_split(sentence, word_dict, w2i):
    sentence = re.sub(r'[.!,;?]', ' ', sentence).split()
    for i in range(len(sentence)):
        if sentence[i] not in word_dict:
            sentence[i] = 3
        else:
            sentence[i] = w2i[sentence[i]]
    sentence.insert(0, 1)
    sentence.append(2)
    return sentence


def annotate(label_file, word_dict, w2i):
    annotated_caption = []
    with open(label_file, 'r') as f:
        label = json.load(f)
    for d in label:
        for s in d['caption']:
            s = s_split(s, word_dict, w2i)
            annotated_caption.append((d['id'], s))
    return annotated_caption

class trainingData(Dataset):
    def __init__(self, label_file, files_dir, word_dict, w2i):
        self.label_file = label_file
        self.files_dir = files_dir
        self.word_dict = word_dict
        self.avi = getTrainingFeatures()
        self.w2i = w2i
        self.data_pair = annotate(label_file, word_dict, w2i)

    def __len__(self):
        return len(self.data_pair)

    def __getitem__(self, idx):
        assert (idx < self.__len__())
        avi_file_name, sentence = self.data_pair[idx]
        data = torch.Tensor(self.avi[avi_file_name])
        data += torch.Tensor(data.size()).random_(0, 2000) / 10000.
        return torch.Tensor(data), torch.Tensor(sentence)


class encoderModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers = 1):
        super(encoderModel, self).__init__()
        self.hidden = None
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # define LSTM layer
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)

    def forward(self, x_input):
        lstm_out, self.hidden = self.lstm(x_input.view(x_input.shape[0], x_input.shape[1], self.input_size))

        return lstm_out, self.hidden

    def init_hidden(self, batch_size):
        '''
        initialize hidden state
        : param batch_size:    x_input.shape[1]
        : return:              zeroed hidden state and cell state
        '''

        return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                torch.zeros(self.num_layers, batch_size, self.hidden_size))

class decoderModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers = 1):
        super(decoderModel, self).__init__()
        self.hidden = None
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers)
        self.linear = nn.Linear(hidden_size, input_size)
    def forward(self, x_input, encoder_hidden_states):
        lstm_out, self.hidden = self.lstm(x_input.unsqueeze(0), encoder_hidden_states)
        output = self.linear(lstm_out.squeeze(0))

        return output, self.hidden


class encodeDecodeModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(encodeDecodeModel, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.encoder = encoderModel(input_size = input_size, hidden_size = hidden_size)
        self.decoder = decoderModel(input_size = input_size, hidden_size = hidden_size)

    def train_model(self, input, target, epochs, target_len, batch_size = BATCH_SIZE,
                    training_prediction = 'recursive', teacher_forcing_ratio = 0.5, learning_rate = 0.001,
                    dynamic_tf = False):
        losses = np.full(epochs, np.nan)

        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        # calculate number of batch iterations
        n_batches = int(input.shape[1] / batch_size)

        with trange(epochs) as tr:
            for it in tr:

                batch_loss = 0.
                batch_loss_tf = 0.
                batch_loss_no_tf = 0.
                num_tf = 0
                num_no_tf = 0

                for b in range(n_batches):
                    # select data
                    input_batch = input[:, b: b + batch_size, :]
                    target_batch = target[:, b: b + batch_size, :]

                    # outputs tensor
                    outputs = torch.zeros(target_len, batch_size, input_batch.shape[2])

                    # initialize hidden state
                    encoder_hidden = self.encoder.init_hidden(batch_size)

                    # zero the gradient
                    optimizer.zero_grad()

                    # encoder outputs
                    encoder_output, encoder_hidden = self.encoder(input_batch)

                    # decoder with teacher forcing
                    decoder_input = input_batch[-1, :, :]  # shape: (batch_size, input_size)
                    decoder_hidden = encoder_hidden

                    if training_prediction == 'recursive':
                        # predict recursively
                        for t in range(target_len):
                            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                            outputs[t] = decoder_output
                            decoder_input = decoder_output

                    if training_prediction == 'teacher_forcing':
                        # use teacher forcing
                        if random() < teacher_forcing_ratio:
                            for t in range(target_len):
                                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                                outputs[t] = decoder_output
                                decoder_input = target_batch[t, :, :]

                        # predict recursively
                        else:
                            for t in range(target_len):
                                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                                outputs[t] = decoder_output
                                decoder_input = decoder_output

                    if training_prediction == 'mixed_teacher_forcing':
                        # predict using mixed teacher forcing
                        for t in range(target_len):
                            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                            outputs[t] = decoder_output

                            # predict with teacher forcing
                            if random() < teacher_forcing_ratio:
                                decoder_input = target_batch[t, :, :]

                            # predict recursively
                            else:
                                decoder_input = decoder_output

                    # compute the loss
                    loss = criterion(outputs, target_batch)
                    batch_loss += loss.item()

                    # backpropagation
                    loss.backward()
                    optimizer.step()

                # loss for epoch
                batch_loss /= n_batches
                losses[it] = batch_loss

                # dynamic teacher forcing
                if dynamic_tf and teacher_forcing_ratio > 0:
                    teacher_forcing_ratio = teacher_forcing_ratio - 0.02

                    # progress bar
                tr.set_postfix(loss="{0:.3f}".format(batch_loss))

        return losses

    def predict(self, input_tensor, target_len):

        '''
        : param input_tensor:      input data (seq_len, input_size); PyTorch tensor
        : param target_len:        number of target values to predict
        : return np_outputs:       np.array containing predicted values; prediction done recursively
        '''

        # encode input_tensor
        input_tensor = input_tensor.unsqueeze(1)  # add in batch size of 1
        encoder_output, encoder_hidden = self.encoder(input_tensor)

        # initialize tensor for predictions
        outputs = torch.zeros(target_len, input_tensor.shape[2])

        # decode input_tensor
        decoder_input = input_tensor[-1, :, :]
        decoder_hidden = encoder_hidden

        for t in range(target_len):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            outputs[t] = decoder_output.squeeze(0)
            decoder_input = decoder_output

        np_outputs = outputs.detach().numpy()

        return np_outputs

def main():
    #i2w, w2i, word_dict = preprocess()
    #label_file = "/Users/aaronspears/Documents/2024Spring/CPSC8430/HW2/MLDS_hw2_1_data/training_label.json"
    #files_dir = "/Users/aaronspears/Documents/2024Spring/CPSC8430/HW2/MLDS_hw2_1_data/training_data/feat"
    #train_dataset = trainingData(label_file, files_dir, word_dict, w2i)
    #train_dataloader = DataLoader(dataset = train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    #encoder = encoderModel()
    #decoder = decoderModel(512, len(i2w) + 4, len(i2w) + 4, 1024, 0.3)

    iw = 80
    ow = 20
    s = 5

    t, y = synthetic_data()
    t_train, y_train, t_test, y_test = train_test_split(t, y, split=0.8)

    Xtrain, Ytrain = windowed_dataset(y_train, input_window=iw, output_window=ow, stride=s)
    Xtest, Ytest = windowed_dataset(y_test, input_window=iw, output_window=ow, stride=s)

    X_train, Y_train, X_test, Y_test = numpy_to_torch(Xtrain, Ytrain, Xtest, Ytest)

    # specify model parameters and train
    model = encodeDecodeModel(input_size=X_train.shape[2], hidden_size=15)
    loss = model.train_model(X_train, Y_train, epochs=50, target_len=ow, batch_size=5,
                             training_prediction='mixed_teacher_forcing', teacher_forcing_ratio=0.6, learning_rate=0.01,
                             dynamic_tf=False)

if __name__ == "__main__":
    main()