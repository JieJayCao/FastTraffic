# FastTraffic
The repository of FastTraffic, code for our Computer Networks Journal paper: FastTraffic: A lightweight method for encrypted traffic fast classification


## Dependencies
- Python 3.7
- PyTorch 1.4.0

## Dataset
- [ISCX-VPN](https://www.unb.ca/cic/datasets/vpn.html)
- [ISCX-Tor](https://www.unb.ca/cic/datasets/tor.html)
- [USTC-TFC](https://github.com/yungshenglu/USTC-TFC2016)
  
## Traffic preprocessing
- Preprocessing method proposed in [Deep Packet](https://github.com/munhouiani/Deep-Packet)
- python preprocess.py

## Usage 
> Adjust **./dataset** to your data

> python run.py
## Please quote if it helps you
> @article{xu2023fasttraffic,
  title={FastTraffic: A lightweight method for encrypted traffic fast classification},
  author={Xu, Yuwei and Cao, Jie and Song, Kehui and Xiang, Qiao and Cheng, Guang},
  journal={Computer Networks},
  volume={235},
  pages={109965},
  year={2023},
  publisher={Elsevier}
}




