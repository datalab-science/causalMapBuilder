
from __future__ import print_function
import smtplib #to send emails

import email_config
from google.cloud import language
import os
import boto3
import logging
import sys
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from email.mime.base import MIMEBase
from email import encoders
import networkx as nx
import matplotlib.pyplot as plt
import remove_words
import csv
import nltk
from nltk.corpus import wordnet

def create_a_map():
    plt.clf()
    fig_size = plt.rcParams["figure.figsize"]
    f = csv.reader(open("node_relations.csv","r"))

  


    G = nx.DiGraph()
    for row in f:
        print(row[0],row[1])
        G.add_edge(row[0],row[1])

    nx.draw(G, with_labels=True, font_size=10, node_color='yellowgreen', pos=nx.spring_layout(G),  node_size=1000)
    plt.savefig("networkx.png")
    plt.show()
    plt.close("networkx.png")
    G.clear()

create_a_map()