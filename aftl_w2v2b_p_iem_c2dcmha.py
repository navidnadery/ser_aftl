from typing import Optional
import itertools
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import os
import torch
import torchaudio
from torch import nn
from disvoice.prosody import Prosody
from contrastive_center_loss import ContrastiveCenterLoss
from loss_functions import AngularPenaltySMLoss#, AM_Softmax
# from aamsoftmax import LossFunction
from glob import glob
from logzero import logger
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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
src_path = input("please enther the absolout path for IEMOCAP Sessions on your system\n(e.g. '/home/IEMOCAP' if wav files are in /home/IEMOCAP/Session*/sentences/wav/Ses0*/*.wav )\n")
if len(glob(f"{src_path}/Session*/sentences/wav/Ses0*/*.wav")) == 0 or len(glob(f"{src_path}/Session*/sentences/wav/Ses0*/*.pt")) == 0:
    logger.error(f"{src_path} is not correct, there isn't any wav or pt file in\n\t{src_path}/Session*/sentences/wav/Ses0*/*.wav")
    exit()
if len(glob(f"{src_path}/Session*/sentences/wav/Ses0*/*.pt")) > 0:
    con = input("Do you want to remove pt files and generate them again(y/n)?")
if con == "y":
    os.system(f'rm "{src_path}/Session*/sentences/wav/Ses0*/*.pt"')

data_dir = Path(src_path)

## Map emotions to categories
import enum
class generate_label(enum.IntEnum):
    ang = 0
    sad = 1
    hap = 2
    exc = 2
    neu = 3
    fear = 4
    others = 5
# In[2]: Initialize prosody and wav2vec 2 model and  Extract and Save features as pt files in wav dirs
if con == "y":
    prosody = Prosody()
    bundle = torchaudio.pipelines.WAV2VEC2_BASE
    print("Sample Rate:", bundle.sample_rate)
    model = bundle.get_model().to(device)

    ses = defaultdict(list)
    f = 0
    for i in range(1,6):
        for g in ['M', 'F']:
            ses[f] = list(data_dir.glob(f'Session{i}/sentences/wav/Ses0{i}{g}_impro*/*.wav'))
            print(len(ses[f]))
            i_ = 0
            for wav_file in tqdm(ses[f]):
                i_ +=1
                waveform, sample_rate = torchaudio.load(wav_file)
                waveform = waveform.to(device)
                if sample_rate != bundle.sample_rate:
                    waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)
                with torch.inference_mode():
                    #    conv_feats, _ = model.feature_extractor(waveform, waveform.shape[1])
                    features, _ = model.extract_features(waveform)
                    features = torch.cat((features[0].detach().cpu(), features[8].detach().cpu()), 0)
                    torch.save(features, os.path.join(str(wav_file)[:-4]+'v.pt'))
                features = torch.nan_to_num(prosody.extract_features_file(str(wav_file), static=True, plots=False, fmt="torch"), nan=0)
                torch.save(features, os.path.join(str(wav_file)[:-4]+'p.pt'))
            f += 1

# In[3]: Read processed data from system
def read_IEMOCAP(audio_indexes, emotion_steps, is_training, filter_num):
    sample_num = 3000
    pernums_sample = np.arange(sample_num)
    
    sample_label = torch.empty((sample_num, 1), dtype=torch.int8)
    sample_label_pf = torch.empty((sample_num, 1), dtype=torch.int8)
    sample_data = torch.empty((sample_num, 2, timesteps, filter_num), dtype=torch.float32)
    sample_pros = torch.empty((sample_num, 103), dtype=torch.float32)

    snum = 0
    sample_num = 0
    sample_emt = {'hap':0, 'ang':0, 'neu':0, 'sad':0 , 'exc':0}
    processed = []

    for filename in tqdm(audio_indexes):
        if filename.split('/')[4] not in processed:
            parts = filename.split('/')
            emotdir = '/'.join(parts[:-4]) + '/dialog/EmoEvaluation/' + parts[-2] + '.txt'
            processed.append(parts[-2])
            emot_map = {}
            with open(emotdir,'r') as emot_to_read:
                while True:
                    line = emot_to_read.readline()
                    if not line:
                        break
                    if(line[0] == '['):
                        t = line.split()
                        emot_map[t[3]] = t[4]
        wavname = filename.split("/")[-1][:-4]
        emotion = emot_map[wavname]
        if(emotion in ['hap','ang','neu','sad','exc']):
            featv1 = torch.load(filename)
            featv2 = featv1[1:2,:]
            featv1 = featv1[0:1,:]
            featp = torch.load(filename[:-4]+'p.pt').squeeze(0)
            time = featv1.shape[1]
            if(time <= timesteps):
                em = generate_label[emotion].value
                pernums_sample[snum] = 1
                sample_label_pf[snum] = em
                feat1_ = featv1.clone()
                feat2_ = featv2.clone()
                while featv1.shape[1] < timesteps:
                    featv1 = torch.cat([featv1, feat1_], dim = 1)
                    featv2 = torch.cat([featv2, feat2_], dim = 1)
                sample_data[sample_num,0,:,:] = featv1[0,:timesteps, :]
                sample_data[sample_num,1,:,:] = featv2[0,:timesteps, :]
                
                sample_pros[sample_num,:] = featp
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
                    sample_data[sample_num,0,:,:] = featv1[0,begin:end,:]
                    sample_data[sample_num,1,:,:] = featv2[0,begin:end,:]
                    
                    sample_pros[sample_num,:] = featp
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
    if is_training:
        arr = np.arange(sample_num)
        np.random.shuffle(arr)
        min_ = sample_pros.min(0)[0]
        max_ = sample_pros.max(0)[0]
        torch.save({'min_':min_, 'max_':max_}, ".norm_metric.pt")
        sample_pros = (sample_pros-min_)/(max_-min_)
        sample_data = sample_data[arr]
        sample_pros = sample_pros[arr]
        sample_label = sample_label[arr]
    else:
        norm_metrics = torch.load(".norm_metric.pt")
        max_ = norm_metrics.get('max_')
        min_ = norm_metrics.get('min_')
        sample_pros = (sample_pros-min_)/(max_-min_)

    return sample_data, sample_pros, sample_label, sample_label_pf, pernums_sample, sample_emt


## Can set different number for each fold
emotion_steps = [{'hap':100, 'ang':45, 'neu':timesteps, 'sad':45 , 'exc':100}]*10 # F9
em_steps_test = [{'hap':timesteps, 'ang':timesteps, 'neu':timesteps, 'sad':timesteps , 'exc':timesteps}]*10 # F9

ses = defaultdict(list)
f = 0
for i in range(1,6):
  for g in ['M', 'F']:
    ses[f] = list(data_dir.glob(f'Session{i}/sentences/wav/Ses0{i}{g}_impro*/*v.pt'))
    f += 1


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
        self.label_dict         = data_source.label_dict
        self.nPerEmotion        = nPerEmotion
        
    def __iter__(self):
        dictkeys = list(self.label_dict.keys())
        dictkeys.sort()
        iter_list = []
        key0 = dictkeys[0]
        data0    = self.label_dict[key0]
        pIndex0 = np.random.permutation(data0)
        key1 = dictkeys[1]
        data1    = self.label_dict[key1]
        pIndex1 = np.random.permutation(data1)
        key2 = dictkeys[2]
        data2    = self.label_dict[key2]
        pIndex2 = np.random.permutation(data2)
        key3 = dictkeys[3]
        data3    = self.label_dict[key3]
        pIndex3 = np.random.permutation(data3)
        for i in range(max(len(pIndex0), len(pIndex1), len(pIndex2), len(pIndex3))):
            iter_list.append(np.random.permutation([pIndex0[i%len(pIndex0)], pIndex1[i%len(pIndex1)], pIndex2[i%len(pIndex2)], pIndex3[i%len(pIndex3)]]))
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
        self.W = nn.Parameter(torch.FloatTensor(num_classes, Fc2).uniform_(-0.25, 0.25).to(device), requires_grad=True)
        self.softmax = nn.Softmax(dim=1)

    def Linear(self, input):
      return F.linear(self.layer_norm(input), F.normalize(self.W))

    def forward(self, x):
        x = self.cred(x.transpose(2,1))
        layer1 = self.relu(self.fc1(x.view(x.shape[0], -1)))
        layer2 = self.relu(self.fc2(layer1))
        Ylogits = self.Linear(layer2)
        Ylogits = self.softmax(Ylogits)
        return Ylogits


class MultiheadAttentionKQV(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        out_dim: int = None,
        dropout: float = 0.0,
    ):
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

    def forward(
        self,
        xk: torch.Tensor,
        xq: torch.Tensor,
        xv: torch.Tensor,
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
        self.mhatt = MultiheadAttentionKQV(embed_dim = int((Di2-16)/2), num_heads = 8, out_dim = Fc2)
        self.conv = nn.Sequential(nn.Conv2d(2, 4, [11,17], stride=[5,2]))
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
learning_rate = 1e-3
clip = 0
verbos = 0
def train(train_loader, valid_loader, test_loader, valid_label, test_label, pernums_valid, pernums_test, valid_count, test_count, fi, ri):
    model_name = f'best_model{fi+ri}.pth'
    y_pred_valid = np.empty((valid_count, 4),dtype=np.float32)
    y_pred_test = np.empty((test_count, 4),dtype=np.float32)
    vnum = pernums_valid.shape[0]
    tnum = pernums_test.shape[0]
    y_valid = np.empty((vnum, 4), dtype=np.float32)
    y_test = np.empty((tnum, 4), dtype=np.float32)
    best_valid_ac = 0
    ##########tarin model###########
    def init_weights(m):
        if type(m) == torch.nn.Linear:
            m.weight.data.normal_(0.0, 0.1)
            if m.bias is not None:
                m.bias.data.fill_(0.1)
        elif type(m) == torch.nn.Conv2d:
            m.weight.data.normal_(0.0, 0.1)
            if m.bias is not None:
                m.bias.data.fill_(0.1)

    model_ = model_emo_mh(Di2=768, Fc2=64)
    model_.apply(init_weights)
    model_ = model_.to(device)
    optimizer = optim.Adam(model_.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=5e-4)

    center_loss = ContrastiveCenterLoss(dim_hidden=68*64*4, num_classes=4,
                                            lambda_c=1, use_cuda=device)
    center_loss = center_loss.cuda()
    optimizer_c = torch.optim.SGD(center_loss.parameters(), lr=0.5)

    #criterion = torch.nn.CrossEntropyLoss()
    criterion = AngularPenaltySMLoss(64*2*4, 4, loss_type='cosface') #loss_type='cosface')
    # criterion1 = AM_Softmax(64*2*4, 4, 0.4, 30)
    # criterion2 = LossFunction(64*2*4, 4, 0.4, 30)
    criterion = criterion.cuda()
    # criterion1 = criterion1.cuda()
    # criterion2 = criterion2.cuda()
    
    num_epoch = 50

    for epoch in range(num_epoch):
        model_.train()
        for train_data_, train_pros_, train_label_ in train_loader:
            inputs = train_data_.to(device)
            prosody = train_pros_.to(device)
            targets = train_label_.to(device)
            optimizer_c.zero_grad()
            optimizer.zero_grad()
            feats, outputs = model_(inputs, prosody)
            loss = criterion(outputs, targets)[0] + center_loss(targets.to(torch.int64), feats)
            loss.backward()
            if clip:
                torch.nn.utils.clip_grad_norm_(model_.parameters(), clip)
            optimizer.step()
            optimizer_c.step()
            
        index = 0
        cost_valid = 0
        model_.eval()
        with torch.no_grad():
            for valid_data_, valid_pros_, Valid_label_ in valid_loader:
                inputs = valid_data_.to(device)
                prosody= valid_pros_.to(device)
                targets = Valid_label_.to(device)
                feats, outputs = model_(inputs, prosody)
                loss, outputs = criterion(outputs, targets)
                y_pred_valid[index:index+targets.shape[0], :] = outputs.cpu().detach().numpy()
                loss = (loss + center_loss(targets.to(torch.int64), feats)).cpu().detach().numpy()
                index += targets.shape[0]
            cost_valid = cost_valid + np.sum(loss)
        model_.train()
        index = 0
        cost_valid = cost_valid/len(y_pred_valid)
        
        for s in range(vnum):
            y_valid[s,:] = np.max(y_pred_valid[index:index+pernums_valid[s],:], 0)
            index = index + pernums_valid[s]
        
          # compute evaluated results
        valid_rec_uw = recall(valid_label, np.argmax(y_valid, 1), average='micro')
        valid_wac_uw = weightacc(valid_label, np.argmax(y_valid, 1))
        valid_pre_uw = precision(valid_label, np.argmax(y_valid, 1), average='micro')
        valid_conf = confusion(valid_label, np.argmax(y_valid, 1))
        valid_acc_uw = accuracy(valid_label, np.argmax(y_valid, 1))
        valid_fscore = fscore(valid_label, np.argmax(y_valid, 1), average='micro')

        
        if valid_acc_uw > best_valid_ac:
            best_valid_re = valid_rec_uw
            best_valid_wa = valid_wac_uw
            best_valid_conf = valid_conf
            best_valid_ac = valid_acc_uw
            best_valid_pr = valid_pre_uw
            best_valid_fs = valid_fscore
            if not os.path.isdir(checkpoint):
                os.mkdir(checkpoint)
            torch.save({"state_dict": model_.state_dict(), "classifier": criterion.state_dict()}, os.path.join(checkpoint, model_name))

        
        if verbos:
            # print results
            print ("*****************************************************************")
            print ("Epoch: %05d" %(epoch+1))
            # print ("Training cost: %2.3g" %tcost)
            # print ("Training accuracy: %3.4g" %tracc)
            print ("Valid cost: %2.3g" %cost_valid)
            print ("Valid_Recall: %3.4g" %valid_rec_uw)
            print ("Best valid_RE: %3.4g" %best_valid_re)
            print ("Valid_Accuracy: %3.4g" %valid_acc_uw)
            print ("Best valid_Acc: %3.4g" %best_valid_ac)
            print ('Valid Confusion Matrix:["ang","sad","hap","neu"]')
            print (valid_conf)
            print ('Best Valid Confusion Matrix:["ang","sad","hap","neu"]')
            print (best_valid_conf)
            print ("*****************************************************************" )

    saved_model = torch.load(os.path.join(checkpoint, model_name))
    statedict = saved_model['state_dict']
    criteriondict = saved_model['classifier']
    criterion.load_state_dict(criteriondict)
    model_.load_state_dict(statedict)
    index = 0
    cost_valid = 0
    model_.eval()
    with torch.no_grad():
        for test_data_, test_pros_, Test_label_ in test_loader:
            inputs = test_data_.to(device)
            prosody= test_pros_.to(device)
            targets = Test_label_.to(device)
            feats, outputs = model_(inputs, prosody)
            loss, outputs = criterion(outputs, targets)
            y_pred_test[index:index+targets.shape[0], :] = outputs.cpu().detach().numpy()
            loss = (loss + center_loss(targets.to(torch.int64), feats)).cpu().detach().numpy()
            index += targets.shape[0]
        cost_valid = cost_valid + np.sum(loss)
    index = 0
    cost_valid = cost_valid/len(y_pred_test)
    
    for s in range(tnum):
        y_test[s,:] = np.max(y_pred_test[index:index+pernums_test[s],:], 0)
        index = index + pernums_test[s]
    
      # compute evaluated results
    test_rec = recall(test_label, np.argmax(y_test, 1), average='micro')
    test_conf = confusion(test_label, np.argmax(y_test, 1))
    test_ac = accuracy(test_label, np.argmax(y_test, 1))
    test_pr = precision(test_label, np.argmax(y_test, 1), average='micro')
    test_wa = weightacc(test_label, np.argmax(y_test, 1))
    test_fscore = fscore(test_label, np.argmax(y_test, 1), average='micro')

    if verbos:
        print ("Test_Recall: %3.4g" %test_rec)
        print ("Test_Accuracy: %3.4g" %test_ac)
        print ('Test Confusion Matrix:["ang","sad","hap","neu"]')
        print (test_conf)

    return best_valid_ac, best_valid_pr, best_valid_wa, best_valid_re, best_valid_fs, test_ac, test_pr, test_wa, test_rec, test_fscore, best_valid_conf, test_conf

# In[7]:

import gc

total_acval = []
total_actst = []
total_prval = []
total_prtst = []
total_waval = []
total_watst = []
total_reval = []
total_retst = []
total_fsval = []
total_fstst = []
total_conval = []
total_contst = []
for fold in range(10):
    eval_idx  = [str(fi_) for fi_ in ses[fold]]
    test_idx  = [str(fi_) for fi_ in ses[(fold+1)%9]]
    train_idx = [str(fi_) for fo in range(10) if fo not in [fold, (fold+1)%9] for fi_ in ses[fo]]
    train_data, train_pros, train_label, train_label_pf, train_sample, train_emt =       read_IEMOCAP(audio_indexes = train_idx, emotion_steps = emotion_steps[fold], is_training = True, filter_num = 768)
    test_data, test_pros, test_label, test_label_pf, test_sample, test_emt =       read_IEMOCAP(audio_indexes = test_idx, emotion_steps = em_steps_test[fold], is_training = False, filter_num = 768)
    eval_data, eval_pros, eval_label, eval_label_pf, eval_sample, eval_emt =       read_IEMOCAP(audio_indexes = eval_idx, emotion_steps = em_steps_test[fold], is_training = False, filter_num = 768)
    train_label = train_label.squeeze(1)
    test_label = test_label.squeeze(1)
    test_label_pf = test_label_pf.squeeze(1)
    eval_label = eval_label.squeeze(1)
    eval_label_pf = eval_label_pf.squeeze(1)

    batch_size = 32
    train_dataset = data_emo(train_data, train_pros, train_label.to(torch.long))
    emo_sampler = balanced_sampler(train_dataset, 4)

    train_loader = DataLoader(train_dataset, batch_size=  batch_size, sampler=emo_sampler , num_workers=0, pin_memory=False, drop_last=False)
    test_loader = DataLoader(data_emo(test_data, test_pros, test_label.to(torch.long))          , batch_size=batch_size, shuffle=False, sampler=None, batch_sampler=None          , num_workers=0, pin_memory=False, drop_last=False)
    eval_loader = DataLoader(data_emo(eval_data, eval_pros, eval_label.to(torch.long))          , batch_size=batch_size, shuffle=False, sampler=None, batch_sampler=None          , num_workers=0, pin_memory=False, drop_last=False)

    valid_acc = []
    valid_inf = []
    test_inf = []
    valid_confusion = []
    test_confusion = []
    for i in range(10):
        best_valid_ac, best_valid_pr, best_valid_wa, best_valid_re, best_valid_fs, test_ac, test_pr, test_wa, test_rec, test_fs, best_valid_conf, test_conf = train(train_loader, eval_loader, test_loader ,valid_label = eval_label_pf, test_label = test_label_pf, pernums_valid = eval_sample, pernums_test = test_sample, valid_count = eval_label.shape[0], test_count = test_label.shape[0], fi=str(fold), ri=str(i).zfill(2))
        valid_acc.append(best_valid_ac)
        valid_inf.append((best_valid_pr, best_valid_wa, best_valid_re, best_valid_fs))
        test_inf.append((test_ac, test_pr, test_wa, test_rec, test_fs))
        valid_confusion.append(best_valid_conf)
        test_confusion.append(test_conf)
        
    idx_best = np.argmax(valid_acc)
    print("Best of fold %d; valid_Acc: %3.4g valid_precesion: %3.4g valid_WeightedAcc: %3.4g valid_recall: %3.4g valid_fscore: %3.4g %d" %(fold, valid_acc[idx_best], *valid_inf[idx_best], idx_best))
    print("Best of fold %d; test_Acc: %3.4g test_precesion: %3.4g test_WeightedAcc: %3.4g test_recall: %3.4g test_fscore: %3.4g" %(fold, *test_inf[idx_best]))
    total_acval.append(sum(valid_acc)/len(valid_acc))
    total_prval.append(sum([valid_inf[id_][0] for id_ in range(len(valid_inf))])/len(valid_inf))
    total_waval.append(sum([valid_inf[id_][1] for id_ in range(len(valid_inf))])/len(valid_inf))
    total_reval.append(sum([valid_inf[id_][2] for id_ in range(len(valid_inf))])/len(valid_inf))
    total_fsval.append(sum([valid_inf[id_][3] for id_ in range(len(valid_inf))])/len(valid_inf))
    total_actst.append(sum([test_inf[id_][0] for id_ in range(len(test_inf))])/len(test_inf))
    total_prtst.append(sum([test_inf[id_][1] for id_ in range(len(test_inf))])/len(test_inf))
    total_watst.append(sum([test_inf[id_][2] for id_ in range(len(test_inf))])/len(test_inf))
    total_retst.append(sum([test_inf[id_][3] for id_ in range(len(test_inf))])/len(test_inf))
    total_fstst.append(sum([test_inf[id_][4] for id_ in range(len(test_inf))])/len(test_inf))
    total_conval.append(valid_confusion[idx_best])
    total_contst.append(test_confusion[idx_best])
    del train_data, train_label, train_label_pf, train_sample, train_emt
    del test_data, test_label, test_label_pf, test_sample, test_emt 
    del eval_data, eval_label, eval_label_pf, eval_sample, eval_emt
    del train_loader, test_loader, eval_loader
    
    gc.collect()


print("\nAVG Acc Val Test")
print(sum(total_acval)/len(total_acval))
print(sum(total_actst)/len(total_actst))

print("\nAVG Precision Val Test")
print(sum(total_prval)/len(total_prval))
print(sum(total_prtst)/len(total_prtst))

print("\nAVG Recall Val Test")
print(sum(total_reval)/len(total_reval))
print(sum(total_retst)/len(total_retst))

print("\nAVG WAcc Val Test")
print(sum(total_waval)/len(total_waval))
print(sum(total_watst)/len(total_watst))

print("\nAVG Fscore Val Test")
print(sum(total_fsval)/len(total_fsval))
print(sum(total_fstst)/len(total_fstst))

print("\nBest Confusion Valid")
print(np.average(total_conval, axis=0))
print("\nBest Confusion Test")
print(np.average(total_contst, axis=0))
