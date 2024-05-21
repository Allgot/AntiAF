# author: EManuele/Immanuel, JLee/Allgot

import requests
import requests.exceptions
import logging
from lxml import etree, html
import re
import pickle

logging.getLogger("requests").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO)
TOR_NODES_LIST_URL = "https://web.archive.org/web/20180917035310/https://www.dan.me.uk/tornodes"


class NodeGuard:

    def __init__(self):
        self.guards = {}

    def loadNodes(self):
        page = requests.get(TOR_NODES_LIST_URL)
        if page.status_code == requests.codes.ok:
            logging.info("Grabbed List from %s", TOR_NODES_LIST_URL)
            tree = html.fromstring(page.content)
            # Extract the part of the page that contains the TOR node list
            content = page.text.split('__BEGIN_TOR_NODE_LIST__ //-->')[-1].split('<!-- __END_TOR_NODE_LIST__ //-->')[0]
            lines = content.split('<br/>')
            pattern = re.compile(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}")
            tmp_guards = {}
            
            for line in lines:
                if pattern.match(line.strip().split('|')[0]):
                    server = line.strip().split('|')
                    if "G" in server[4]:  # Check if 'G' for Guard is in the flags section
                        tmp_guards[server[0]] = server[4]
                        logging.info("Guard node IP: %s, attrs: %s", server[0], server[4])
            logging.info("Downloaded %d Nodes", len(tmp_guards))
            
            self.guards = tmp_guards if tmp_guards else logging.info("No nodes loaded, list empty.")
        else:
            logging.error("Failed to retrieve the page.")

    def loadNodesFromPickle(self, pickle_file):
        with open(pickle_file + '.pkl', 'rb') as f:
            self.guards = pickle.load(f)
        logging.info("%d Nodes loaded from %s." % (len(self.guards), pickle_file))

    def saveNodesToFile(self, pickle_file):
        with open(pickle_file + '.pkl', 'wb') as f:
            pickle.dump(self.guards, f, 0)
        logging.info("Nodes saved to %s" % (pickle_file))

    # add an IP to NodeGuard obj
    def manually_add(self, ip):
        self.guards[ip] = "G"

    def isGuardNode(self, nodeip):
        return nodeip in self.guards