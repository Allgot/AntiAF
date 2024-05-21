# author: EManuele/Immanuel, JLee/Allgot

import util
import os
import multiprocessing

if __name__ == '__main__':
    # How to use
    # basepath: path to folder that contains other folders with .pcap
    # Ex: captures --|
    #                | youtube -- |
    #                             | youtube_1.pcap
    #                             | youtube_2.pcap
    #
    #                | spotify -- |
    #                             |  spotify_1.pcap
    
    basepath =  "connection_padding_pcaps/" 
    #basepath =  "reduced_connection_padding_pcaps/"
    
    # output_folder: Self-explicative..
    output_folder = "output/"
    # Each entry is a list [ app, category] where app is also the name of the folder that contains capture
    folders_category = [["torbrowser_alpha", "web_and_social"], ["facebook", "web_and_social"], \
                        ["skype", "voip"], ["youtube", "streaming"], ["spotify", "streaming"], \
                        ["dailymotion", "streaming"], ["twitch", "streaming"], \
                        ["instagram", "web_and_social"], \
                        ["replaio_radio", "streaming"], ["utorrent", "filesharing"]]

    # List of Timeouts to use to compute flows
    timeouts = [10] # [10, 15]
    # List of activity timeout to use to ocompute flows (how many sec. before flow is considered idle)
    activity_timeouts = [2] # [2, 5]

    tasks = []

    for timeout in timeouts:
        for activitytimeout in activity_timeouts:
            for folder in folders_category:
                for file in os.listdir(basepath+folder[0]):
                    if file.endswith(".pcap"):
                        tasks.append((basepath+folder[0]+"/"+file, output_folder, folder[0], folder[1], timeout, activitytimeout, "./entry_nodes"))
                        # print("Current Folder: %s Current File: %s Current Timeout: %d Activity Timeout: %d" % (basepath+folder[0], file, timeout, activitytimeout))
                        # util.ProcessPcapTor(basepath+folder[0]+"/"+file,  output_folder=output_folder, \
                        #                     label=folder[0], category=folder[1], \
                        #                     Timeout=timeout, ActivityTimeout=activitytimeout, \
                        #                     TorPickle="./entry_nodes")
    
    with multiprocessing.Pool(5) as p:
        p.starmap(util.ProcessPcapTor, tasks)
        p.close()
        p.join()