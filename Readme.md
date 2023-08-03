# FastTraffic
The repository of FastTraffic, a lightweight encrypted network traffic classification DL model.

To speed up processing, we set an IP packet as the granularity of FastTraffic, truncate the informative parts in packets as inputs, and utilize a **text-like packet tokenization** method. For a lightweight and effective model, we propose an **N-gram feature embedding method** to represent structured and sequential features of packets and design a three-layer MLP to complete fast classification. 
