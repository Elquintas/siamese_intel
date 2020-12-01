import torch
import sys
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import Embedding
import torch.nn.init
import torchvision
import classes
import hparams
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
torch.manual_seed(42)

#   Pathological Speech labels = 1
#   Healthy Speech labels = 0
#
#   Final Labels:
#       1 - different pairs
#       0 - similar pairs

    
def acc_calc(predicted,target):
    size = len(predicted)
    tot = np.abs(predicted-target)
    tot = np.sum(tot)

    acc = ((size-tot)/size)*100
    print('Accuracy: {}'.format(acc))

def precision_recall(predicted,target):
    
    tn=0
    tp=0
    fn=0
    fp=0

    for i in range(len(predicted)):
        if (predicted[i]==target[i]==1):
            tp = tp+1
        if (predicted[i]==target[i]==0):
            tn = tn+1
        if (target[i]==1 and predicted[i]==0):
            fn = fn+1
        if (target[i]==0 and predicted[i]==1):
            fp = fp+1
    print('Precision: {}'.format(tp/(tp+fp)))
    print('Recall: {}'.format(tp/(tp+fn)))

    tpr = tp/(tp+fn)
    fpr = fp/(fp+tn)

    print('TPR: {}'.format(tpr))
    print('FPR: {}'.format(fpr))

    auc = roc_auc_score(target,predicted)
    print('AUC: {}'.format(auc))

def plot_losses(train,validation):
    plt.title('Train and Validation Losses')
    line1, = plt.plot(train,'b',label="Train")
    line2, = plt.plot(validation,'g',label="Validation")
    plt.legend(handles=[line1,line2])
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()

def return_batch_read(batches):
    final = []
    for i in range(len(batches)):
        data = np.loadtxt('./data_mfccs/'+batches[i],delimiter=",",dtype=np.float32)
        #data = data.T
        final.append(data)
    return final



if __name__ == "__main__":
    
    device = torch.device('cuda')
    
    sequential_train = classes.sequential_train
    sequential_validation = classes.sequential_validation
    sequential_test = classes.sequential_test

    #Loading Datasets
    dataset = sequential_train()
    trainloader = DataLoader(dataset=dataset, batch_size=hparams.batch_size,
            shuffle=True,drop_last=True) 
    dataset = sequential_validation()
    validationloader = DataLoader(dataset=dataset, batch_size=hparams.batch_size, 
            shuffle=True,drop_last=True)
    dataset = sequential_test()
    testloader = DataLoader(dataset=dataset, batch_size=hparams.batch_size, 
            shuffle=True,drop_last=True)
    
    model = classes.GRU_NET_DUAL(n_gru_layers=hparams.n_gru_layers,
            hidden_dim=hparams.hidden_dim,batch_size=hparams.batch_size)
    
    #Orthogonal initialization of weightsi in the encoder:
    for m in model.modules():
        if isinstance(m, nn.Linear):
            torch.nn.init.orthogonal_(m.weight)
        if isinstance(m, nn.GRU):
            for name, param in m.named_parameters():
                if 'weight_hh' in name:
                    torch.nn.init.orthogonal_(param.data)
                if 'weight_ih' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                if 'bias' in name:
                    param.data.fill_(0)
                
    
    model.cuda()

    # Different optimizers used. Adam promotes a faster loss 
    # convergence during training, and a smaller loss value
    # at the end. SGD promotes slightly better results at 
    # test times, however with a larger training loss

    #optimizer = torch.optim.Adam(model.parameters(),lr=hparams.lr,weight_decay=0.00001) 
    #optimizer = torch.optim.Adadelta(model.parameters(),lr=hparams.lr)
    #optimizer = torch.optim.Adamax(model.parameters(),lr=hparams.lr)
    optimizer = torch.optim.SGD(model.parameters(), lr=hparams.lr,momentum=0.9)   #momentum 0.9?
    

    # Uncomment this part in order to load a pre-trained model

    #checkpoint = torch.load('./checkpoints/model.pth')
    #model.load_state_dict(checkpoint['model_state_dict'])
    #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #ep = checkpoint['epoch']
    #print('LOST EPOCH: {}'.format(ep)) 
    


    # Loss criterion used. In this case, since we are dealing
    # with a classification problem, we used binary cross entropy.
    # However, since the final goal of this system is to obtain
    # a similarity measure, a mean squared error loss can also
    # be implemented, with the proper modifications
    
    criterion = nn.BCELoss()
    #criterion = nn.MSELoss()
    
    loss_log=[]
    val_loss_track=[]
    train_loss_track=[]
    train_loss=[]
    
    iteration = 1
    
    for ep in range(hparams.epochs):
                     
        model.train()
        train_loss=[]
        
        for batch_idx, (x1,y1,x2,y2) in enumerate(trainloader):
            if(batch_idx%1000 == 0):
                print(ep,batch_idx)

            final_label = np.abs(y2-y1)
            
            # INVERT LABELS:
            # NEGATIVE PAIRS: -1
            # POSITIVE PAIRS: +1
            # This can be done directly on the .csv files that contain the pairs
            # This only reverts the similarity scale used
            final_label = np.where(final_label==0,-1,final_label)
            final_label = np.where(final_label==1,0,final_label)
            final_label = -1*final_label    
            final_label = torch.Tensor(final_label)
            
                        
            x1 = return_batch_read(x1)
            x2 = return_batch_read(x2)
            final_label = final_label.cuda()   
            final_label = final_label.reshape(hparams.batch_size,1)
            
            h = model.init_hidden(hparams.batch_size)
            
            h1 = h.data
            h2 = h.data
            h1 = h1.cuda()
            h2 = h2.cuda()

            model.zero_grad()


            # Network with fully connected layer at the end
            #output, h1, h2, euc_dist = model(x1.to(device).float(),x2.to(device).float(),h1,h2) 

            for i in range(len(x1)):                
                
                
                #Expands dims for features with only ONE frame
                if(np.shape(x1[i])==(35,)):
                    x1[i]=np.expand_dims(x1[i],axis=0)
                     
                if(np.shape(x2[i])==(35,)):
                    x2[i]=np.expand_dims(x2[i],axis=0)
                

                x1[i] = torch.Tensor(x1[i])
                x2[i] = torch.Tensor(x2[i])
            
            len_x1=[]
            len_x2=[]
            for i in range(len(x1)):
                len_x1.append(len(x1[i]))
                len_x2.append(len(x2[i]))
            

            # Pads and packs the batches on the longest file used.
            # Since we are addressing files of variable length, we 
            # need to perform this step so that the whole batch has 
            # the same dimensions
            x1 = torch.nn.utils.rnn.pad_sequence(x1, batch_first=True)
            x2 = torch.nn.utils.rnn.pad_sequence(x2, batch_first=True)
            x1 = x1.reshape(hparams.batch_size,max(len_x1),hparams.feats_dim)   
            x2 = x2.reshape(hparams.batch_size,max(len_x2),hparams.feats_dim)            
            x1 = torch.nn.utils.rnn.pack_padded_sequence(x1,len_x1,
                    batch_first=True, enforce_sorted=False)
            x2 = torch.nn.utils.rnn.pack_padded_sequence(x2,len_x2,
                    batch_first=True, enforce_sorted=False)
            x1 = x1.cuda()
            x2 = x2.cuda()
            

            output, h1, h2, cos_dist = model(x1,x2,h1,h2)            
            loss = criterion(output.squeeze(),final_label.float().squeeze())
            train_loss.append(loss.cpu().detach().numpy())
            
            if batch_idx % 10 ==0:
                loss_log.append(loss)
            
            loss.backward()
            optimizer.step()
            

            # Checkpoint Saving
        #if (ep % hparams.checkpoint_interval == 0 and ep != 0):
        print('SAVING CHECKPOINT AT EPOCH NR: {}'.format(ep))
        torch.save({'epoch': ep,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                    }, "./checkpoints/model.pth")

        
        # Validation at end of each epoch
        model.eval()
        h = model.init_hidden(hparams.batch_size)
        h1 = h.data
        h2 = h.data
        validation_loss = []
        for batch_idx, (x1,y1,x2,y2) in enumerate(validationloader):
            
            x1 = return_batch_read(x1)
            x2 = return_batch_read(x2)
            
            h = model.init_hidden(hparams.batch_size)
            h1 = h.data
            h2 = h.data
            h1 = h1.cuda()
            h2 = h2.cuda()

            final_label_val = np.abs(y1-y2)
             
            final_label_val = np.where(final_label_val==0,-1,final_label_val)
            final_label_val = np.where(final_label_val==1,0,final_label_val)
            final_label_val = -1*final_label_val
            final_label_val = torch.Tensor(final_label_val)

            
            final_label_val = final_label_val.cuda()

            for i in range(len(x1)):
                if(np.shape(x1[i])==(35,)):
                    x1[i]=np.expand_dims(x1[i],axis=0)
                if(np.shape(x2[i])==(35,)):
                    x2[i]=np.expand_dims(x2[i],axis=0)                
                x1[i] = torch.Tensor(x1[i])
                x2[i] = torch.Tensor(x2[i])
            
            len_x1=[]
            len_x2=[]

            for i in range(len(x1)):
                len_x1.append(len(x1[i]))
                len_x2.append(len(x2[i]))

            x1 = torch.nn.utils.rnn.pad_sequence(x1, batch_first=True)
            x2 = torch.nn.utils.rnn.pad_sequence(x2, batch_first=True)
            x1 = x1.reshape(hparams.batch_size,max(len_x1),hparams.feats_dim)
            x2 = x2.reshape(hparams.batch_size,max(len_x2),hparams.feats_dim)
            x1 = torch.nn.utils.rnn.pack_padded_sequence(x1,len_x1,
                    batch_first=True, enforce_sorted=False)
            x2 = torch.nn.utils.rnn.pack_padded_sequence(x2,len_x2,
                    batch_first=True, enforce_sorted=False)
            x1 = x1.cuda()
            x2 = x2.cuda()

            # invert labels (1- positive pair, 0 - negative pair)
            #final_label_val = final_label_val-1
            #final_label_val = np.abs(final_label_val)
            #final_label_val = final_label_val.reshape(hparams.batch_size,1)
                
            val_out, h1, h2, euc_dist = model(x1,x2,h1,h2)    
            val_loss = criterion(val_out.squeeze(),final_label_val.float().squeeze())
            
            #val_loss = contrastive_loss(final_label_val,euc_dist.cuda())
            
            validation_loss.append(val_loss.cpu().detach().numpy())
        
        
        val_loss_track.append(np.mean(validation_loss))
        train_loss_track.append(np.mean(train_loss))
        print('EPOCH: {} TRAINING LOSS: {} -- VALIDATION LOSS: {}'.format(ep,
            np.mean(train_loss),np.mean(validation_loss)))
        train_loss=[]



    # plot train and validation losses
    plot_losses(train_loss_track,val_loss_track)
