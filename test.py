import sys
import torch
import json
from Final import test_data, test, MODELS, encoderRNN, decoderRNN, attention
from torch.utils.data import DataLoader
from MLDS_hw2_1_data.bleu_eval import BLEU
import pickle

model = torch.load('./SavedModel/model3.h5')
filepath = './MLDS_hw2_1_data/testing_data/feat'
dataset = test_data('{}'.format(sys.argv[1]))
testing_loader = DataLoader(dataset, batch_size=32, shuffle=True)

with open('indexToWord.pickle', 'rb') as handle:
    i2w = pickle.load(handle)
#model = model.cuda()

ss = test(testing_loader, model, i2w)

with open(sys.argv[2], 'w') as f:
    for id, s in ss:
        f.write('{},{}\n'.format(id, s))


test = json.load(open('./MLDS_hw2_1_data/testing_label.json'))
output = sys.argv[2]
result = {}
with open(output,'r') as f:
    for line in f:
        line = line.rstrip()
        comma = line.index(',')
        test_id = line[:comma]
        caption = line[comma+1:]
        result[test_id] = caption
bleu=[]
for item in test:
    score_per_video = []
    captions = [x.rstrip('.') for x in item['caption']]
    score_per_video.append(BLEU(result[item['id']],captions,True))
    bleu.append(score_per_video[0])
average = sum(bleu) / len(bleu)
print("BLEU score = " + str(average))