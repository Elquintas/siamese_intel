import os
import sys
import classes
import hparams
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
import pandas as pd
from torch.autograd import Variable
from astropy.stats import median_absolute_deviation
torch.manual_seed(42)



def load_files(arg1,arg2):
    
    phoneme=np.loadtxt(arg1,delimiter=',',dtype=np.float32)
    
    ref = pd.read_csv(arg2)
    ref = ref.values[:,0]
    final = []
    for i in range(len(ref)):
        #data = np.loadtxt('./data_new/'+ref[i],delimiter=',',dtype=np.float32)
        data = np.loadtxt('./data_mfccs/'+ref[i],delimiter=',',dtype=np.float32)
        final.append(data)

    return phoneme, final


def compare(test_phone, ref_list, model):

    set_sigmoid = []

    for i in range(len(ref_list)):

        # PAIR-WISE COMPARISON
        #x1 = test_phone.reshape(1,np.shape(test_phone)[0],np.shape(test_phone)[1])
        
        x1 = test_phone
        x2 = ref_list[i].reshape(1,np.shape(ref_list[i])[0],np.shape(ref_list[i])[1])

        len_x2 = [np.shape(ref_list[i])[0]]
        
        x2 = torch.nn.utils.rnn.pack_padded_sequence(x2,len_x2,batch_first=True)
        
        
        
        x1 = x1.cuda()
        x2 = x2.cuda()

        model.zero_grad() 
        h = model.init_hidden(1)
        h1 = h.data
        h2 = h.data

        h1 = h1.cuda()
        h2 = h2.cuda()
        
        
        #out,h1,h2,res = model(x1.to(device).float(),x2.to(device).float(),h1,h2)
        out,h1,h2,res = model(x2.to(device).float(),x1.to(device).float(),h1,h2)
        #out = res.cpu().detach().numpy()
        out = out.cpu().detach().numpy()
        #res = res.cpu().detach().numpy()
        #print(res)
        set_sigmoid.append(out)

    
    if np.mean(set_sigmoid) > hparams.sigmoid_threshold:
        #print('negative, sim: {}'.format(np.mean(set_sigmoid)))
        return 1, len(ref_list), np.mean(set_sigmoid)
    if np.mean(set_sigmoid) < hparams.sigmoid_threshold:
        #print('positive, sim: {}'.format(np.mean(set_sigmoid)))
        return 0, len(ref_list), np.mean(set_sigmoid)


def phonetic_median_threshold(avg_sim_all, phoneme):

    if(phoneme=="p"):
        threshold = 0.09 #0.09 
    if(phoneme=="t"):
        threshold = 0.14 #0.14
    if(phoneme=="k"):
        threshold = 0.10 #0.11
    if(phoneme=="b"):
        threshold = 0.09 #0.08
    if(phoneme=="d"):
        threshold = 0.11 #0.13
    if(phoneme=="g"):
        threshold = 0.30 #0.31
    if(phoneme=="S"):
        threshold = 0.36 #0.36
    if(phoneme=="f"):
        threshold = 0.98 #0.98
    if(phoneme=="s"):
        threshold = 0.08 #0.05
    if(phoneme=="v"):
        threshold = 0.69 #0.77
    if(phoneme=="Z"):
        threshold = 0.84 #0.84
    if(phoneme=="m"):
        threshold = 0.44 #0.49
    if(phoneme=="n"):
        threshold = 0.96 #0.97
    if(phoneme=="l"):
        threshold = 0.46 #0.23
    if(phoneme=="R"):
        threshold = 0.93 #0.9
    if(phoneme=="z"):
        threshold = 0.87 #0.87

    goodies=0
    for i in avg_sim_all:
        if(i>threshold):
            goodies = goodies+1
    
    print("number of occurences: {}".format(np.size(avg_sim_all)))
    print("number of good ones: {}".format(goodies))
    print("percentage: {}".format(round(goodies/np.size(avg_sim_all),2)))


if __name__ == "__main__":

    device = torch.device('cuda')

    if (len(sys.argv) !=4):
        print('Usage: <Phoneme to be compared> <Reference list for phoneme> <model to use>')

    # LOADING PRE-TRAINED MODEL
    model = classes.GRU_NET_DUAL(n_gru_layers=hparams.n_gru_layers,hidden_dim=hparams.hidden_dim,batch_size=1)
    model.cuda()
    #checkpoint = torch.load('./checkpoints/model_R_cons.pth')
    checkpoint = torch.load(sys.argv[3])
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()

    path = os.getcwd()
    test_phones_dir = os.path.join(path,sys.argv[1])

    acc = 0
    nr_files = 0
    avg_sim_all=[]
    for filename in os.listdir(test_phones_dir):
        if filename.endswith(".csv"):
            nr_files = nr_files + 1
            
            test_phone, ref_list = load_files(test_phones_dir+filename,sys.argv[2])
            
            #print(filename,np.shape(test_phone))
            
            for i in range(len(ref_list)): 
                if(np.shape(ref_list[i])==(35,)):
                    ref_list[i]=np.expand_dims(ref_list[i],axis=0)
                ref_list[i] = torch.Tensor(ref_list[i])
            if(np.shape(test_phone)==(35,)):
                test_phone=np.expand_dims(test_phone,axis=0)

            ref_list = torch.nn.utils.rnn.pad_sequence(ref_list, batch_first=True)
            
            test_phone = test_phone.reshape(1,np.shape(test_phone)[0],np.shape(test_phone)[1])     
            test_phone = torch.Tensor(test_phone)
            test_phone = torch.nn.utils.rnn.pad_sequence(test_phone, batch_first=True)
            lens = [np.shape(test_phone)[1]]
            
            

            test_phone = torch.nn.utils.rnn.pack_padded_sequence(test_phone,lens,batch_first=True)  
            res, length, out = compare(test_phone, ref_list, model)
            #print(filename,round(out,2))            
            
            avg_sim_all.append(out)
            acc = acc + res
            continue

    #print(length)
    #print(nr_files)
    
    #print(np.shape(avg_sim_all))
    
    ##print("median sim:  {}".format(np.median(avg_sim_all)))
    ##print("MAD sim: {}".format(median_absolute_deviation(avg_sim_all)))
    
    ##print("average sim: {}".format(np.mean(avg_sim_all)))
    ##print("std dev sim: {}".format(np.std(avg_sim_all)))
    
    phonetic_median_threshold(avg_sim_all,sys.argv[2][12])

    #print((acc/nr_files)*100)


