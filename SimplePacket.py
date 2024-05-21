# author: EManuele/Immanuel, JLee/Allgot

from scapy.all import *
import re

class SimplePacket:

    def __init__(self, pkt):
        FIN = 0x01
        SYN = 0x02
        RST = 0x04
        PSH = 0x08
        ACK = 0x10
        URG = 0x20
        ECE = 0x40
        CRW = 0x80
        if(pkt.haslayer(IP) and pkt.haslayer(TCP) and pkt.haslayer(Ether)):
            self.ipSrc = pkt[IP].src
            self.ipDst = pkt[IP].dst
            self.macSrc = pkt[Ether].src
            self.macDst = pkt[Ether].dst
            self.srcPort = pkt[TCP].sport
            self.dstPort = pkt[TCP].dport
            self.len = pkt[IP].len
            self.timestamp = pkt.time
            self.SYN = True if pkt[TCP].flags & SYN else False
            self.FIN = True if pkt[TCP].flags & FIN else False
            self.RST = True if pkt[TCP].flags & RST else False
            self.PSH = True if pkt[TCP].flags & PSH else False
            self.URG = True if pkt[TCP].flags & URG else False
            self.ECE = True if pkt[TCP].flags & ECE else False
            self.ACK = True if pkt[TCP].flags & ACK else False
            self.CRW = True if pkt[TCP].flags & CRW else False
            self.seqn = pkt[TCP].seq
            self.ackn = pkt[TCP].ack
        else:
            # print("Packet has NO IP and/or TCP Layer.")
            raise ValueError("Packet has NO IP and/or TCP Layer.")

    def print(self):
        print("Packet Stats - SRC IP: %s, DST IP: %s, SPORT: %d, DPORT: %d, "
                 "LEN: %d, TS: %f, SYN: %d, FIN: %d, RST: %d, PSH: %d, URG: %d, "
                 "ECE: %d, ACK: %d, CRW: %d, SEQN: %d, ACKN: %d" % (self.ipSrc, self.ipDst,
                 self.srcPort, self.dstPort, self.len, self.timestamp,
                 self.SYN, self.FIN, self.RST, self.PSH, self.URG, self.ECE,
                 self.ACK, self.CRW, self.seqn, self.ackn))
