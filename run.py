# coding: UTF-8
from statistics import mode
import time
import torch
import numpy as np
from train_eval import train, init_network, test
from importlib import import_module
import argparse
from thop import profile, clever_format
import os
from utils_fasttraffic import build_dataset, build_iterator, get_time_dif

parser = argparse.ArgumentParser(description='Encrypted Traffic Classification')
args = parser.parse_args()



"""def get_flops(net,config):
    #x = np.random.randint((1, 1480))
    #x = torch.from_numpy(x).to(config.device)
    x1 = torch.LongTensor(torch.randint(0,255,(50,))).to(config.device)
    x3 =  torch.LongTensor(torch.randint(0,255,(50,))).to(config.device)
    x4 =  torch.LongTensor(torch.randint(0,255,(50,))).to(config.device)
    
    x2 =   torch.LongTensor(50).to(config.device)
    
    flops, params = profile(net, inputs=((x1,x2,x3,x4),),verbose=False)
    flops, params = clever_format([flops, params], "%.3f")

    print("flops:",flops,"params:",params)

"""
    
def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def gen_train(dataset,ls,train_path):
    train_content = []
    for i in ls:
        f = open(dataset+"/data/"+str(i)+".txt",'r')
        train_content.extend(f.readlines())
    f_train = open(train_path,'w')
    #print(len(train_content))
    for i in train_content:
        f_train.write(i)    
    

 
def main():

    # dataset path 
    dataset = "../dataset/vpn"
    embedding = 'random'
    model_name = 'FastTraffic' 
  

    x = import_module('models.' + model_name)
    config = x.Config(dataset, embedding)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True 
    start_time = time.time()
    print("Loading data...")
    vocab, train_data, dev_data, test_data = build_dataset(config, True)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    config.n_vocab = len(vocab)
    model = x.Model(config).to(config.device)
        
    init_network(model)
    print(model.parameters)
    print(get_parameter_number(model))
    train(config, model, train_iter, dev_iter, test_iter)
    test(config,model,test_iter)
 
    
  
    
if __name__ == '__main__':
    main()


