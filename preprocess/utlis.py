from pathlib import Path

from scapy.layers.dns import DNS
from scapy.layers.inet import TCP
from scapy.packet import Padding
from scapy.utils import PcapReader

# for app identification
PREFIX_TO_APP_ID = {
    # AIM chat
    'aim_chat_3a': 0,
    'aim_chat_3b': 0,
    'aimchat1': 0,
    'aimchat2': 0,
    # Email
    'email1a': 1,
    'email1b': 1,
    'email2a': 1,
    'email2b': 1,
    # Facebook
    'facebook_audio1a': 2,
    'facebook_audio1b': 2,
    'facebook_audio2a': 2,
    'facebook_audio2b': 2,
    'facebook_audio3': 2,
    'facebook_audio4': 2,
    'facebook_chat_4a': 2,
    'facebook_chat_4b': 2,
    'facebook_video1a': 2,
    'facebook_video1b': 2,
    'facebook_video2a': 2,
    'facebook_video2b': 2,
    'facebookchat1': 2,
    'facebookchat2': 2,
    'facebookchat3': 2,
    # FTPS
    'ftps_down_1a': 3,
    'ftps_down_1b': 3,
    'ftps_up_2a': 3,
    'ftps_up_2b': 3,
    # Gmail
    'gmailchat1': 4,
    'gmailchat2': 4,
    'gmailchat3': 4,
    # Hangouts
    'hangout_chat_4b': 5,
    'hangouts_audio1a': 5,
    'hangouts_audio1b': 5,
    'hangouts_audio2a': 5,
    'hangouts_audio2b': 5,
    'hangouts_audio3': 5,
    'hangouts_audio4': 5,
    'hangouts_chat_4a': 5,
    'hangouts_video1b': 5,
    'hangouts_video2a': 5,
    'hangouts_video2b': 5,
    # ICQ
    'icq_chat_3a': 6,
    'icq_chat_3b': 6,
    'icqchat1': 6,
    'icqchat2': 6,
    # Netflix
    'netflix1': 7,
    'netflix2': 7,
    'netflix3': 7,
    'netflix4': 7,
    # SCP
    'scp1': 8,
    'scpdown1': 8,
    'scpdown2': 8,
    'scpdown3': 8,
    'scpdown4': 8,
    'scpdown5': 8,
    'scpdown6': 8,
    'scpup1': 8,
    'scpup2': 8,
    'scpup3': 8,
    'scpup5': 8,
    'scpup6': 8,
    # SFTP
    'sftp1': 9,
    'sftp_down_3a': 9,
    'sftp_down_3b': 9,
    'sftp_up_2a': 9,
    'sftp_up_2b': 9,
    'sftpdown1': 9,
    'sftpdown2': 9,
    'sftpup1': 9,
    # Skype
    'skype_audio1a': 10,
    'skype_audio1b': 10,
    'skype_audio2a': 10,
    'skype_audio2b': 10,
    'skype_audio3': 10,
    'skype_audio4': 10,
    'skype_chat1a': 10,
    'skype_chat1b': 10,
    'skype_file1': 10,
    'skype_file2': 10,
    'skype_file3': 10,
    'skype_file4': 10,
    'skype_file5': 10,
    'skype_file6': 10,
    'skype_file7': 10,
    'skype_file8': 10,
    'skype_video1a': 10,
    'skype_video1b': 10,
    'skype_video2a': 10,
    'skype_video2b': 10,
    # Spotify
    'spotify1': 11,
    'spotify2': 11,
    'spotify3': 11,
    'spotify4': 11,
    # Vimeo
    'vimeo1': 12,
    'vimeo2': 12,
    'vimeo3': 12,
    'vimeo4': 12,
    # Voipbuster
    'voipbuster1b': 13,
    'voipbuster2b': 13,
    'voipbuster3b': 13,
    'voipbuster_4a': 13,
    'voipbuster_4b': 13,
    # Youtube
    'youtube1': 14,
    'youtube2': 14,
    'youtube3': 14,
    'youtube4': 14,
    'youtube5': 14,
    'youtube6': 14,
    'youtubehtml5_1': 14
    # Torrent 从源文件插入·
    #'torrent01': 15,
}


ID_TO_APP = {
    0: 'AIM Chat',
    1: 'Email',
    2: 'Facebook',
    3: 'FTPS',
    4: 'Gmail',
    5: 'Hangouts',
    6: 'ICQ',
    7: 'Netflix',
    8: 'SCP',
    9: 'SFTP',
    10: 'Skype',
    11: 'Spotify',
    12: 'Vimeo',
    13: 'Voipbuster',
    14: 'Youtube',
    
}

# for traffic identification
PREFIX_TO_TRAFFIC_ID = {
    # Chat
    'aim_chat_3a': 0,
    'aim_chat_3b': 0,
    'aimchat1': 0,
    'aimchat2': 0,
    'facebook_chat_4a': 0,
    'facebook_chat_4b': 0,
    'facebookchat1': 0,
    'facebookchat2': 0,
    'facebookchat3': 0,
    'hangout_chat_4b': 0,
    'hangouts_chat_4a': 0,
    'icq_chat_3a': 0,
    'icq_chat_3b': 0,
    'icqchat1': 0,
    'icqchat2': 0,
    'skype_chat1a': 0,
    'skype_chat1b': 0,

    # Email
    'email1a': 1,
    'email1b': 1,
    'email2a': 1,
    'email2b': 1,
    'gmailchat1': 1,
    'gmailchat2': 1,
    'gmailchat3': 2,

    # File Transfer
    'ftps_down_1a': 2,
    'ftps_down_1b': 2,
    'ftps_up_2a': 2,
    'ftps_up_2b': 2,
    'sftp1': 2,
    'sftp_down_3a': 2,
    'sftp_down_3b': 2,
    'sftp_up_2a': 2,
    'sftp_up_2b': 2,
    'sftpdown1': 2,
    'sftpdown2': 2,
    'sftpup1': 2,
    'skype_file1': 2,
    'skype_file2': 2,
    'skype_file3': 2,
    'skype_file4': 2,
    'skype_file5': 2,
    'skype_file6': 2,
    'skype_file7': 2,
    'skype_file8': 2,
    'scp1': 2,
    'scpdown1': 2,
    'scpdown2': 2,
    'scpdown3': 2,
    'scpdown4': 2,
    'scpdown5': 2,
    'scpdown6': 2,
    'scpup1': 2,
    'scpup2': 2,
    'scpup3': 2,
    'scpup5': 2,
    'scpup6': 2,

    # Streaming
    'vimeo1': 3,
    'vimeo2': 3,
    'vimeo3': 3,
    'vimeo4': 3,
    
    'youtube1': 3,
    'youtube2': 3,
    'youtube3': 3,
    'youtube4': 3,
    'youtube5': 3,
    'youtube6': 3,
    'youtubehtml5_1': 3,
    
    'netflix1': 3,
    'netflix2': 3,
    'netflix3': 3,
    'netflix4': 3,
    
    'spotify1': 3,
    'spotify2': 3,
    'spotify3': 3,
    'spotify4': 3,


    # Torrent 从源文件提取
    #'torrent01': 4,

    # VoIP
    'facebook_audio1a': 4,
    'facebook_audio1b': 4,
    'facebook_audio2a': 4,
    'facebook_audio2b': 4,
    'facebook_audio3': 4,
    'facebook_audio4': 4,
    
    'hangouts_audio1a': 4,
    'hangouts_audio1b': 4,
    'hangouts_audio2a': 4,
    'hangouts_audio2b': 4,
    'hangouts_audio3': 4,
    'hangouts_audio4': 4,
    
    'skype_audio1a': 4,
    'skype_audio1b': 4,
    'skype_audio2a': 4,
    'skype_audio2b': 4,
    'skype_audio3': 4,
    'skype_audio4': 4,



    # VPN: Chat
    'vpn_aim_chat1a': 5,
    'vpn_aim_chat1b': 5,
    'vpn_facebook_chat1a': 5,
    'vpn_facebook_chat1b': 5,
    'vpn_hangouts_chat1a': 5,
    'vpn_hangouts_chat1b': 5,
    'vpn_icq_chat1a': 5,
    'vpn_icq_chat1b': 5,
    'vpn_skype_chat1a': 5,
    'vpn_skype_chat1b': 5,

    # VPN: File Transfer
    'vpn_ftps_a': 6,
    'vpn_ftps_b': 6,
    'vpn_sftp_a': 6,
    'vpn_sftp_b': 6,
    'vpn_skype_files1a': 6,
    'vpn_skype_files1b': 6,

    # VPN: Email
    'vpn_email2a': 7,
    'vpn_email2b': 7,

    # VPN: Streaming
    'vpn_vimeo_a': 8,
    'vpn_vimeo_b': 8,
    'vpn_youtube_a': 8,
    'vpn_netflix_a': 8,
    'vpn_spotify_a': 8,

    # VPN: Torrent
    'vpn_bittorrent': 9,

    # VPN VoIP
    'vpn_facebook_audio2': 10,
    'vpn_hangouts_audio1': 10,
    'vpn_hangouts_audio2': 10,
    'vpn_skype_audio1': 10,
    'vpn_skype_audio2': 10,
    'vpn_voipbuster1a': 10,
    'vpn_voipbuster1b': 10,
}

ID_TO_TRAFFIC = {
    0: 'Chat',
    1: 'Email',
    2: 'File Transfer',
    3: 'Streaming',
   # 4: 'Torrent',
    4: 'Voip',
    5: 'VPN: Chat',
    6: 'VPN: File Transfer',
    7: 'VPN: Email',
    8: 'VPN: Streaming',
    9: 'VPN: Torrent',
    10: 'VPN: Voip',
}

PREFIX_TO_TorApp_ID = {
    # Facebook
    'voip_facebook_voice_gateway':0,
    'chat_facebookchatgateway':0,
    'chat_gate_facebook_chat':0,
    'voip_gate_facebook_audio':0,

    # imap
    'mail_gateway_thunderbird_imap':1,
    'mail_gate_email_imap_filetransfer':1,

    # vuze
    'p2p_tor_p2p_vuze':2,
    'tor_p2p_vuze-2-1':2,

    # skype
    'voip_gate_skype_audio':3,
    'file-transfer_tor_skype_transfer':3,
    'chat_gate_skype_chat':3,
    'voip_skype_voice_gateway':3,
    'chat_skypechatgateway':3,
    
    # sftp
    'file-transfer_gate_sftp_filetransfer':4,

    # icq
    'chat_gate_icq_chat':5,
    'chat_icqchatgateway':5,

    # spotify
    'tor_spotify2-1':6,
    'audio_spotifygateway':6,
    'tor_spotify2-2':6,
    'audio_tor_spotify2':6,
    'audio_tor_spotify':6,
    
    # hangout
    "voip_gate_hangout_audio":7,
    "chat_gate_hangout_chat":7,
    "voip_hangouts_voice_gateway":7,
    "chat_hangoutschatgateway":7,
    
    # aim 
    "chat_aimchatgateway":8,
    "chat_gate_aim_chat":8,

    # pop 
    "mail_gate_pop_filetransfer":9,
    "mail_gateway_thunderbird_pop":9,

    # multiplespeed
    "p2p_tor_p2p_multiplespeed":10,
    "tor_p2p_multiplespeed2-1":10,

    # ftp
    'file-transfer_gate_ftp_transfer':11,

    # SSL
    "browsing_ssl_browsing_gateway":12,
    "browsing_gate_ssl_browsing":12,

    # youtube
    "video_youtube_html5_gateway":13,
    "video_youtube_flash_gateway":13,

    # vimeo
    'video_vimeo_gateway':14,

    # mam
    "browsing_tor_browsing_mam":15,
    "browsing_tor_browsing_mam2":15
}

ID_TO_TorAPP = {
    0:"Facebook",
    1:"Imap",
    2:"Vuze",
    3:"Skype",
    4:"SFTP",
    5:"ICQ",
    6:"Spotify",
    7:"Hangout",
    8:"Aim",
    9:"POP",
    10:"Multiplespeed",
    11:"FTP",
    12:"SSL",
    13:"Youtube",
    14:"Vimeo",
    15:"Mam"
}

def read_pcap(path: Path):
    packets = PcapReader(str(path))

    return packets


def should_omit_packet(packet):
    # SYN, ACK or FIN flags set to 1 and no payload
    if TCP in packet and (packet.flags & 0x13):
        # not payload or contains only padding
        layers = packet[TCP].payload.layers()
        if not layers or (Padding in layers and len(layers) == 1):
            return True
    # DNS segment
    if DNS in packet:
        return True

    return False
