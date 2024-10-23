import itertools
import os
import enum
import gc

from typing import Optional
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

from disvoice.prosody import Prosody  # from prosody import Prosody
from loss_functions import AngularPenaltySMLoss  # AM_Softmax
# from aamsoftmax import LossFunction
from glob import glob

import torch
import torch.nn as nn
import numpy as np
import torchaudio
import torch.nn.functional as F
import torch.optim as optim
from logzero import logger
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
from sklearn.metrics import recall_score as recall
from sklearn.metrics import precision_score as precision
from sklearn.metrics import f1_score as fscore
from sklearn.metrics import balanced_accuracy_score as weightacc
from sklearn.metrics import confusion_matrix as confusion
from sklearn.metrics import accuracy_score as accuracy

# In[1]: Define objects and variables to run through whole code
timesteps = 350
checkpoint = './checkpoint'
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
src_path = input("please enther the absolout path for ShEMO wav files on your system\n(e.g. '/home/ShEMO/wav/' )\n")
if len(glob(f"{src_path}/*.wav")) == 0 and len(glob(f"{src_path}/*.pt")) == 0:
    logger.error(f"{src_path} is not correct, there isn't any wav or pt file in\n\t{src_path}/*.wav")
    exit()

if len(glob(f"{src_path}/*.pt")) > 0:
    con = input("Do you want to remove pt files and generate them again(y/n)?")
    if con == "y":
        os.system(f'rm {src_path}/*.pt')
else:
    con = "y"

chpoint = input("Please Enter the name of checkpoint you want to fine-tune on (located in checkpoint dir)\n") or "best_model000.pth"

data_dir = Path(src_path)


# Map emotions to categories
class generate_label(enum.IntEnum):
    A = 0  # ang
    S = 1  # sad
    H = 2  # hap
    N = 3  # neu
    F = 4  # fear
    others = 5


# In[2]: Initialize prosody and wav2vec 2 model and  Extract and Save features as pt files in wav dirs
if con == "y":
    prosody = Prosody()
    bundle = torchaudio.pipelines.WAV2VEC2_BASE
    print("Sample Rate:", bundle.sample_rate)
    model = bundle.get_model().to(device)
    ses = defaultdict(list)

    for f, g in enumerate(['M03', 'M12', 'F08', 'F03', 'M42', 'F06', 'M08',
                           'M02', 'M31', 'M43', 'M26', 'M45', 'M28', 'M24',
                           'F18', 'M56', 'M35', 'M04', 'F16', 'F31', 'F04',
                           'F13', 'F14', 'M29', 'M07', 'M40', 'F29', 'M15',
                           'M52', 'F28', 'M30', 'F12', 'M01', 'M47', 'M27',
                           'M48', 'F15', 'M17', 'F09', 'M20', 'M10', 'F26',
                           'F30', 'M33', 'F17', 'M36', 'M44', 'F21', 'F05',
                           'F25', 'F20', 'M51', 'F10', 'M37', 'M11', 'M19',
                           'M50', 'M23', 'M05', 'M16', 'M38', 'F23', 'F22',
                           'M32', 'M09', 'M46', 'M39', 'F02', 'F24', 'M41',
                           'M49', 'M34', 'M54', 'M13', 'M14', 'M18', 'F27',
                           'M21', 'M06', 'F01', 'F07', 'M55', 'F19', 'M25',
                           'M53', 'M22', 'F11']):
        ses[f] = list(data_dir.glob(f'{g}*.wav'))
        print(len(ses[f]))
        for wav_file in tqdm(ses[f]):
            waveform, sample_rate = torchaudio.load(wav_file)
            waveform = waveform.to(device)
            if sample_rate != bundle.sample_rate:
                waveform = torchaudio.functional.resample(waveform,
                                                          sample_rate,
                                                          bundle.sample_rate)
            with torch.inference_mode():
                #    conv_feats, _ = model.feature_extractor(waveform, waveform.shape[1])
                features, _ = model.extract_features(waveform)
                features = torch.cat((features[0].detach().cpu(),
                                      features[8].detach().cpu()), 0)
                torch.save(features, os.path.join(str(wav_file)[:-4]+'v.pt'))
            features = torch.nan_to_num(prosody.extract_features_file(
                str(wav_file), static=True, plots=False, fmt="torch"), nan=0)
            torch.save(features, os.path.join(str(wav_file)[:-4]+'p.pt'))


# In[3]: Read processed data from system
def read_ShEMO(audio_indexes, emotion_steps, is_training, filter_num):
    sample_num = 6000
    pernums_sample = np.arange(sample_num)
    sample_label = torch.empty((sample_num, 1), dtype=torch.int8)
    sample_label_pf = torch.empty((sample_num, 1), dtype=torch.int8)
    sample_data = torch.empty((sample_num, 2, timesteps, filter_num), dtype=torch.float32)
    sample_pros = torch.empty((sample_num, 103), dtype=torch.float32)

    snum = 0
    sample_num = 0
    sample_emt = {'H': 0, 'A': 0, 'N': 0, 'S': 0}

    for filename in tqdm(audio_indexes):
        emotion = filename.split('/')[-1][3]
        if (emotion in ['H', 'A', 'N', 'S']):
            featv1 = torch.load(filename)
            featv2 = featv1[1:2, :]
            featv1 = featv1[0:1, :]
            featp = torch.load(filename[:-4]+'p.pt').squeeze(0)
            time = featv1.shape[1]
            if (time <= timesteps):
                em = generate_label[emotion].value
                pernums_sample[snum] = 1
                sample_label_pf[snum] = em
                feat1_ = featv1.clone()
                feat2_ = featv2.clone()
                while featv1.shape[1] < timesteps:
                    featv1 = torch.cat([featv1, feat1_], dim=1)
                    featv2 = torch.cat([featv2, feat2_], dim=1)
                sample_data[sample_num, 0, :, :] = featv1[0, :timesteps, :]
                sample_data[sample_num, 1, :, :] = featv2[0, :timesteps, :]                
                sample_pros[sample_num, :] = featp
                sample_label[sample_num] = em
                sample_emt[emotion] = sample_emt[emotion] + 1
                sample_num = sample_num + 1
            else:
                frames = divmod(time-timesteps, emotion_steps[emotion])[0] + 1
                em = generate_label[emotion].value
                pernums_sample[snum] = frames
                sample_label_pf[snum] = em
                for i in range(frames):
                    begin = emotion_steps[emotion]*i
                    end = begin + timesteps
                    sample_data[sample_num, 0, :, :] = featv1[0, begin:end, :]
                    sample_data[sample_num, 1, :, :] = featv2[0, begin:end, :]
                    sample_pros[sample_num, :] = featp
                    sample_label[sample_num] = em
                    sample_emt[emotion] = sample_emt[emotion] + 1
                    sample_num = sample_num + 1
            snum = snum + 1

    sample_label = sample_label[:sample_num]
    sample_data = sample_data[:sample_num, :, :, :]

    sample_pros = sample_pros[:sample_num, :]
    pernums_sample = pernums_sample[:snum]
    sample_label_pf = sample_label_pf[:snum]

    ## use training min/max normalization values
    # if is_training:
    #     arr = np.arange(sample_num)
    #     np.random.shuffle(arr)
    #     min_ = sample_pros.min(0)[0]
    #     max_ = sample_pros.max(0)[0]
    #     torch.save({'min_':min_, 'max_':max_}, ".norm_metric.pt")
    #     sample_pros = (sample_pros-min_)/(max_-min_)
    #     sample_data = sample_data[arr]
    #     sample_pros = sample_pros[arr]
    #     sample_label = sample_label[arr]
    # else:
    #     norm_metrics = torch.load(".norm_metric.pt")
    #     max_ = norm_metrics.get('max_')
    #     min_ = norm_metrics.get('min_')
    #     sample_pros = (sample_pros-min_)/(max_-min_)

    return sample_data, sample_pros, sample_label, sample_label_pf, pernums_sample, sample_emt


## Can set different number for each fold
emotion_steps = [{'H': 100, 'A': timesteps, 'N': timesteps, 'S': 100}]*10  # F9
em_steps_test = [{'H': timesteps, 'A': timesteps, 'N': timesteps, 'S': timesteps}]*10

ses = defaultdict(list)
for f, g in enumerate(['M03', 'M12', 'F08', 'F03', 'M42', 'F06', 'M08', 'M02',
                       'M31', 'M43', 'M26', 'M45', 'M28', 'M24', 'F18', 'M56',
                       'M35', 'M04', 'F16', 'F31', 'F04', 'F13', 'F14', 'M29',
                       'M07', 'M40', 'F29', 'M15', 'M52', 'F28', 'M30', 'F12',
                       'M01', 'M47', 'M27', 'M48', 'F15', 'M17', 'F09', 'M20',
                       'M10', 'F26', 'F30', 'M33', 'F17', 'M36', 'M44', 'F21',
                       'F05', 'F25', 'F20', 'M51', 'F10', 'M37', 'M11', 'M19',
                       'M50', 'M23', 'M05', 'M16', 'M38', 'F23', 'F22', 'M32',
                       'M09', 'M46', 'M39', 'F02', 'F24', 'M41', 'M49', 'M34',
                       'M54', 'M13', 'M14', 'M18', 'F27', 'M21', 'M06', 'F01',
                       'F07', 'M55', 'F19', 'M25', 'M53', 'M22', 'F11']):
    ses[f] = list(data_dir.glob(f'{g}*v.pt'))


# In[4]: Define data loader and sampler
class data_emo(Dataset):
    def __init__(self, data, pros, label) -> None:
        super().__init__()
        label_dict = defaultdict(list)
        self.data, self.pros, self.label = data, pros, label
        for i, item in enumerate(label):
            item = int(item)
            label_dict[item].append(i)
        self.label_dict = label_dict

    def __len__(self):
        return self.label.shape[0]

    def __getitem__(self, index):
        return self.data[index], self.pros[index], self.label[index]


class balanced_sampler(Sampler):
    def __init__(self, data_source, nPerEmotion):
        self.label_dict = data_source.label_dict
        self.nPerEmotion = nPerEmotion
        
    def __iter__(self):
        dictkeys = list(self.label_dict.keys())
        dictkeys.sort()
        iter_list = []
        key0 = dictkeys[0]
        data0 = self.label_dict[key0]
        pIndex0 = np.random.permutation(data0)
        key1 = dictkeys[1]
        data1 = self.label_dict[key1]
        pIndex1 = np.random.permutation(data1)
        key2 = dictkeys[2]
        data2 = self.label_dict[key2]
        pIndex2 = np.random.permutation(data2)
        key3 = dictkeys[3]
        data3 = self.label_dict[key3]
        pIndex3 = np.random.permutation(data3)
        for i in range(max(len(pIndex0), len(pIndex1), len(pIndex2), len(pIndex3))):
            iter_list.append(np.random.permutation([pIndex0[i % len(pIndex0)],
                                                    pIndex1[i % len(pIndex1)],
                                                    pIndex2[i % len(pIndex2)],
                                                    pIndex3[i % len(pIndex3)]]))
        return iter(itertools.chain.from_iterable([iter for iter in iter_list]))

    def __len__(self):
        return len(self.data_source)


# In[5]: Define models
class model_emo(nn.Module):
    def __init__(self, num_classes=4, Di1=16, Di2=32, Drc=32, Fc1=64, Fc2=64):
        super(model_emo, self).__init__()
        self.cred = nn.Conv1d(Di2, Drc, 1)
        self.relu = nn.LeakyReLU(0.2)
        # fully connected layers
        self.fc1 = nn.Linear(Di1 * Drc, Fc1)
        self.fc2 = nn.Linear(Fc1, Fc2)
        self.layer_norm = nn.LayerNorm(Fc2, elementwise_affine=False)
        self.W = nn.Parameter(torch.FloatTensor(num_classes, Fc2).
                              uniform_(-0.25, 0.25).to(device),
                              requires_grad=True)
        self.softmax = nn.Softmax(dim=1)

    def Linear(self, input):
        return F.linear(self.layer_norm(input), F.normalize(self.W))

    def forward(self, x):
        x = self.cred(x.transpose(2, 1))
        layer1 = self.relu(self.fc1(x.view(x.shape[0], -1)))
        layer2 = self.relu(self.fc2(layer1))
        Ylogits = self.Linear(layer2)
        Ylogits = self.softmax(Ylogits)
        return Ylogits


class MultiheadAttentionKQV(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int,
                 out_dim: int = None, dropout: float = 0.0):
        super().__init__()
        head_dim = embed_dim // num_heads
        if head_dim * num_heads != embed_dim:
            raise ValueError(f"`embed_dim ({embed_dim})` is not divisible by `num_heads ({num_heads})`")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = torch.nn.Dropout(dropout)
        self.head_dim = head_dim

        self.scaling = self.head_dim ** -0.5

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        if out_dim is not None:
            self.out_proj = nn.Linear(embed_dim, out_dim, bias=True)
        else:
            self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(self, xk: torch.Tensor, xq: torch.Tensor, xv: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,) -> torch.Tensor:
        batch_size, channels, length, embed_dim = xk.size()
        shape = (batch_size, channels*length, self.num_heads, self.head_dim)
        shape_ = (batch_size, self.num_heads, self.head_dim)
        q = self.q_proj(xq).view(*shape_)
        k = self.k_proj(xk).view(*shape).permute(0, 1, 3, 2)  # B, nH, Hd
        v = self.v_proj(xv).view(*shape)
        weights = self.scaling * (q.unsqueeze(1) @ k)  # B, nH
        if attention_mask is not None:
            weights += attention_mask

        weights = torch.nn.functional.softmax(weights, dim=-1)
        weights = self.dropout(weights)
        output = weights @ v  # B, nH, Hd
        output = output.reshape(batch_size, channels, length, embed_dim)
        output = self.out_proj(output)
        return output


class model_emo_mh(nn.Module):
    def __init__(self, Di2=32, Fc2=64):
        super(model_emo_mh, self).__init__()
        self.mhatt = MultiheadAttentionKQV(embed_dim=int((Di2-16)/2),
                                           num_heads=8, out_dim=Fc2)
        self.conv = nn.Sequential(nn.Conv2d(2, 4, [11, 17], stride=[5, 2]))
        self.pros1 = nn.Sequential(
            nn.Linear(103, 128),
            nn.Sigmoid())
        self.pros2 = nn.Sequential(
            nn.Linear(128, int((Di2-16)/2)),
            nn.LeakyReLU(0.2))

    def Linear(self, input):
        return F.linear(self.layer_norm(input), F.normalize(self.W))

    def forward(self, x, p):
        p1 = self.pros1(p)
        p_ = self.pros2(p1)
        c1 = self.conv(x)
        att = self.mhatt(c1, p_, c1)
        std, avg = torch.std_mean(att, unbiased=False, dim=2)
        seqr = torch.cat((avg, std), dim=2)
        seqrw = seqr * p1.unsqueeze(1)
        return att, seqrw.view(seqrw.shape[0], -1)


# In[6]:
freeze = 0
load = 1
learning_rate = 1e-3
clip = 0
verbos = 0


def train(train_loader, test_loader, test_label,
          pernums_test, test_count, chpath='', fnum=''):
    y_pred_test = np.empty((test_count, 4), dtype=np.float32)
    tnum = pernums_test.shape[0]
    y_test = np.empty((tnum, 4), dtype=np.float32)
    ##########tarin model###########

    def init_weights(m):
        if type(m) is torch.nn.Linear:
            m.weight.data.normal_(0.0, 0.1)
            if m.bias is not None:
                m.bias.data.fill_(0.1)
        elif type(m) is torch.nn.Conv2d:
            m.weight.data.normal_(0.0, 0.1)
            if m.bias is not None:
                m.bias.data.fill_(0.1)

    model_ = model_emo_mh(Di2=768, Fc2=64)
    model_.apply(init_weights)
    model_ = model_.to(device)
    criterion = AngularPenaltySMLoss(64*2*4, 4, loss_type='cosface')
    criterion = criterion.cuda()

    if load:
        model_dict = model_.state_dict()
        state_dict = {}
        criteriondict = {}
        saved_model = torch.load(os.path.join("checkpoint", chpath))
        state_dict = saved_model['state_dict']
        criteriondict = saved_model['classifier']
        criterion.load_state_dict(criteriondict)
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict} # k not in ['W']}
        model_dict.update(pretrained_dict)
        model_.load_state_dict(model_dict)

    if freeze:
        for param in model_.parameters():
            param.requires_grad = False
        optimizer = optim.Adam(criterion.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=5e-4)
    else:
        optimizer = optim.Adam([
            {'params': list(model_.parameters())},
            {'params': list(criterion.parameters()), 'lr': 1e-2}],
            lr=1e-2, betas=(0.9, 0.999), weight_decay=5e-4)

    num_epoch = 20
    model_.train()
    for epoch in range(num_epoch):
        for train_data_, train_pros_, train_label_ in train_loader:
            inputs = train_data_.to(device)
            prosody = train_pros_.to(device)
            targets = train_label_.to(device)
            optimizer.zero_grad()
            feats, outputs = model_(inputs, prosody)
            loss = criterion(outputs, targets)[0]
            loss.backward()
            if clip:
                torch.nn.utils.clip_grad_norm_(model_.parameters(), clip)
            optimizer.step()

    index = 0
    model_.eval()
    with torch.no_grad():
        for test_data_, test_pros_, Test_label_ in test_loader:
            inputs = test_data_.to(device)
            prosody = test_pros_.to(device)
            targets = Test_label_.to(device)
            feats, outputs = model_(inputs, prosody)
            loss, outputs = criterion(outputs, targets)
            y_pred_test[index:index+targets.shape[0], :] = outputs.cpu().detach().numpy()
            index += targets.shape[0]

    index = 0

    for s in range(tnum):
        y_test[s, :] = np.max(y_pred_test[index:index+pernums_test[s], :], 0)
        index = index + pernums_test[s]

    test_rec = recall(test_label, np.argmax(y_test, 1), average='macro')
    test_wac = weightacc(test_label, np.argmax(y_test, 1))
    test_pre = precision(test_label, np.argmax(y_test, 1), average='macro')
    test_conf = confusion(test_label, np.argmax(y_test, 1))
    test_acc = accuracy(test_label, np.argmax(y_test, 1))
    test_fscore = fscore(test_label, np.argmax(y_test, 1), average='macro')

    torch.save({"state_dict": model_.state_dict(), "classifier": criterion.state_dict()}, os.path.join(checkpoint, chpath[:-4]+str(fnum)+"_shemo.pth"))

    return test_acc, test_pre, test_wac, test_rec, test_fscore, test_conf


total_ac = []
total_pr = []
total_wa = []
total_re = []
total_fs = []
total_co = []

for f_, fold in enumerate(range(0, 87, 9)):
    train_idx = [str(fi_) for fo in range(87) if fo not in
                 [fold, fold+1, fold+2, fold+3, fold+4, fold+5, fold+6, fold+7, fold+8]
                 for fi_ in ses[fo]]
    test_idx = [str(fi_) for fo in
                [ses[fold], ses[fold+1], ses[fold+2], ses[fold+3], ses[fold+4],
                 ses[fold+5], ses[fold+6], ses[fold+7], ses[fold+8]]
                for fi_ in fo]
    train_data, train_pros, train_label, train_label_pf, train_sample, train_emt = \
        read_ShEMO(audio_indexes=train_idx, emotion_steps=emotion_steps[fold],
                   is_training=True, filter_num=768)
    test_data, test_pros, test_label, test_label_pf, test_sample, test_emt = \
        read_ShEMO(audio_indexes=test_idx, emotion_steps=em_steps_test[fold],
                   is_training=False, filter_num=768)
    train_label = train_label.squeeze(1)
    test_label = test_label.squeeze(1)
    test_label_pf = test_label_pf.squeeze(1)

    batch_size = 32
    train_dataset = data_emo(train_data, train_pros, train_label.to(torch.long))
    emo_sampler = balanced_sampler(train_dataset, 4)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              sampler=emo_sampler, num_workers=0,
                              pin_memory=False, drop_last=False)
    test_loader = DataLoader(data_emo(test_data, test_pros, test_label.to(torch.long)),
                             batch_size=batch_size, shuffle=False, sampler=None,
                             batch_sampler=None, num_workers=0, pin_memory=False,
                             drop_last=False)

    test_ac, test_pr, test_wa, test_re, test_fs, test_conf = \
        train(train_loader, test_loader, test_label=test_label_pf,
              pernums_test=test_sample, test_count=test_label.shape[0],
              chpath=chpoint, fnum=f_)
    print("Result of fold %d; test_Acc: %3.4g test_precesion: %3.4g test_WeightedAcc: %3.4g test_recall: %3.4g test_fscore: %3.4g" % (fold, test_ac, test_pr, test_wa, test_re, test_fs))
    total_ac += [test_ac]
    total_pr += [test_pr]
    total_wa += [test_wa]
    total_re += [test_re]
    total_fs += [test_fs]
    total_co += [test_conf]

    del train_data, train_label, train_label_pf, train_sample, train_emt, test_data, test_label, test_label_pf, test_sample, test_emt, train_loader, test_loader
    gc.collect()


print("\nAVG Acc Val Test")
print(np.average(total_ac, axis=0))

print("\nAVG Precision Val Test")
print(np.average(total_pr, axis=0))

print("\nAVG Recall Val Test")
print(np.average(total_re, axis=0))

print("\nAVG WAcc Val Test")
print(np.average(total_wa, axis=0))

print("\nAVG Fscore Val Test")
print(np.average(total_fs, axis=0))

print("\nAVG Confusion Matrix Test")
print(np.average(total_co, axis=0))
