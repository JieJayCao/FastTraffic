# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchstat import stat

class Config(object):

    def __init__(self, dataset, embedding):
        self.model_name = 'FastTraffic'
        self.train_path = dataset + '/data/train.txt'                                
        self.dev_path = dataset + '/data/dev.txt'                                    
        self.test_path = dataset + '/data/test.txt'                                 
        #self.test_path = dataset + '/data/tsne.txt'
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt', encoding='utf-8').readlines()]              
        self.vocab_path = dataset + '/data/vocab.pkl'                                
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        
        self.log_path = dataset + '/log/' + self.model_name
        self.embedding_pretrained = None                                       
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
        self.dropout = 0.4                                            
        self.require_improvement = 200000                               
        self.num_classes = len(self.class_list)                         
        self.n_vocab = 0                                               
        self.num_epochs = 10                                            
        self.batch_size = 204
        self.pad_size = 50                                           
        self.learning_rate = 0.0029                                     
        self.embed = 40       
        self.hidden_size = 150                                       
        self.n_gram_vocab = 5000                                      
        self.save_res = dataset +"/"+ self.model_name +"_res.txt"



class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        
        self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=None)
        self.embedding_ngram2 = nn.Embedding(config.n_gram_vocab, config.embed)
        self.embedding_ngram3 = nn.Embedding(config.n_gram_vocab, config.embed)
        self.dropout = nn.Dropout(config.dropout)
       
        self.fc1 = nn.Linear(config.embed*3, config.hidden_size)
        self.bn = nn.BatchNorm1d(config.hidden_size,affine = False)
        self.fc2 = nn.Linear(config.hidden_size, config.num_classes)
       


    def forward(self, x):

        out_word = self.embedding(x[0])
        out_bigram = self.embedding_ngram2(x[2])
        out_trigram = self.embedding_ngram3(x[3])
       
        out = torch.cat((out_word, out_bigram, out_trigram), -1)
        
        #out = torch.cat((out_word), -1)
        #out = out_bigram.mean(dim=1)
        #out = out_word + out_bigram + out_trigram
        out = out.mean(dim=1)

       

        out = self.fc1(out)
        out = self.bn(out)
        out = F.gelu(out)
        out = self.dropout(out)
        
        # TSNE get Embedding
        """f = open("tsne_ustc.txt",'a')
        E_out = out[0].cpu()
        tsne = str(E_out.numpy().tolist()).strip("[").strip("]")
        f.write(tsne+'\n')"""

        out = self.fc2(out)
        return out
    

"""if __name__ == '__main__':
    

    dataset = "/home/dl/Desktop/program/Traffic_class/TrafficisText/kfolddatast/service"
    
    config = Config(dataset, embedding="random")
    net = Model(config)
    print(net)
    x = np.random.randint(0, 255, (1, 50))
    y = np.array(50)
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)
    
    #(x, seq_len, bigram, trigram)
    inputs = (x,y,x,x)
    
    flops, params = profile(net, inputs=(inputs,),verbose=False)
    flops, params = clever_format([flops, params], "%.3f")
    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        output = net(inputs)
        print(prof.key_averages().table(sort_by="self_cpu_time_total"))"""
