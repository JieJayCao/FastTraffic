

import numpy as np
from scapy.compat import raw
from scapy.layers.inet import IP, UDP, TCP
from scapy.layers.l2 import Ether
from scapy.packet import Padding
from scipy import sparse
import os
from utlis import should_omit_packet, read_pcap, PREFIX_TO_APP_ID, PREFIX_TO_TRAFFIC_ID
from utlis import PREFIX_TO_TorApp_ID, ID_TO_APP, ID_TO_TRAFFIC
from tqdm import tqdm
import time 
import random
from scapy.all import *

def transform2hex(num):
    if num is not None:
        string = str(hex(num)).replace("0x",'')
        if len(string) == 1:
            return "0"+string
        else:
            return string

def one_gram(packet_string):
    """    
    data = binascii.hexlify(bytes(payload)) 
    packet_string = data.decode()"""
    i = 0
    n = 2
    bigram_string = []
    for i, _ in enumerate(packet_string):
        if i % n == 0:
            bigram_string.append(packet_string[i:i + n])
        i += n  
    if len(bigram_string[-1]) < n:
        bigram_string[-1] += (n- len(bigram_string[-1]))*"0"

    return " ".join(bigram_string)

def bi_gram(packet_string):
    """    
    data = binascii.hexlify(bytes(payload)) 
    packet_string = data.decode()"""
    i = 0
    n = 4
    bigram_string = []
    for i, _ in enumerate(packet_string):
        if i % n == 0:
            bigram_string.append(packet_string[i:i + n])
        i += n  
    if len(bigram_string[-1]) < n:
        bigram_string[-1] += (n- len(bigram_string[-1]))*"0"

    return " ".join(bigram_string)



def remove_ether_header(packet):
    if Ether in packet:
        return packet[Ether].payload

    return packet


def mask_ip(packet):
    if IP in packet:
        packet[IP].src = '0.0.0.0'
        packet[IP].dst = '0.0.0.0'

    return packet

def sport_mask(packet):
    if TCP in packet:
        # get layers after udp
        packet[TCP].sport = 0
        #packet[TCP].dport = 0
    if UDP in packet:
        packet[UDP].sport = 0
        
    return packet

def pad_udp(packet):
    if UDP in packet:
        # get layers after udp
        layer_after = packet[UDP].payload.copy()

        # build a padding layer
        pad = Padding()
        pad.load = '\x00' * 12

        layer_before = packet.copy()
        layer_before[UDP].remove_payload()
        packet = layer_before / pad / layer_after

        return packet

    return packet

def packet_to_sparse_array(packet, max_length=512):
    arr = np.frombuffer(raw(packet), dtype=np.uint8)[0: max_length] #/ 255
    if len(arr) < max_length:
        pad_width = max_length - len(arr)
        arr = np.pad(arr, pad_width=(0, pad_width), constant_values=0)
    return arr

def get_ip_and_transport_header_lengths(packet):
    ip_header_length = 0
    transport_header_length = 0

    # 检查数据包是否包含 IP 层
    if IP in packet:
        ip_header_length = packet[IP].ihl * 4  # IP 首部长度（以字节为单位）

        # 检查数据包是否包含传输层（TCP 或 UDP）
        if TCP in packet:
            transport_header_length = packet[TCP].dataofs * 4  # 传输层首部长度（以字节为单位）
        elif UDP in packet:
            transport_header_length = 20  # UDP 首部固定长度为 8 字节
        
            
    return ip_header_length + transport_header_length



def transform_packet(packet):
    if should_omit_packet(packet):
        return None

    packet = remove_ether_header(packet)
    packet = sport_mask(packet)
    packet = mask_ip(packet)
    packet = pad_udp(packet)
    
    arr = packet_to_sparse_array(packet)
    if arr is not None:
        token = ""
        for i in arr:
            token += transform2hex(i)
        return one_gram(token)




def transform_payload(path):
    f_service = open("service.txt",'a')
    f_app = open("app.txt",'a')

    prefix = path.split('/')[-1].split('.')[0].lower()
    app_label = PREFIX_TO_APP_ID.get(prefix)
    service_label = PREFIX_TO_TRAFFIC_ID.get(prefix)
  
    
    for i, packet in enumerate(read_pcap(path)):
        token = transform_packet(packet)
        
        if token is not None: 
            if app_label is not None:
                f_app.write(token+"\t"+str(app_label)+"\n")
            
            
            if service_label is not None:
                f_service.write(token+"\t"+str(service_label)+"\n")
    
            else:
                return
        if i > 10000 :
            return

               

    
if __name__ == '__main__':
    source =  "VPN-Pcaps/" 
    root = os.listdir(source)
    random.shuffle(root)
  
    for i in tqdm(root):
        path = source + i
        transform_payload(path)

    
