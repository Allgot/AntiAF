# author: JLee/Allgot

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
    
    # im_folder: Intermediary directory
    im_folder = "connection_padding_pcaps_merge/"
    
    # output_folder: Final directory
    output_folder = "connection_padding_pcaps_def/"

    # Each entry is a list [ app, category] where app is also the name of the folder that contains capture
    folders = ["dailymotion","torbrowser_alpha", "facebook", \
                "skype", "youtube", "spotify", \
                "twitch", "instagram", \
                "replaio_radio", "utorrent"]

    tasks = []
    # sys.setrecursionlimit(10**9)

    for folder in folders:
        for file in os.listdir(basepath+folder):
            if not os.path.exists(im_folder+folder):
                os.mkdir(im_folder+folder)
            if file.endswith(".pcap"):
                tasks.append((basepath+folder+"/"+file, im_folder+folder+"/"+file, "./entry_nodes"))
                # print("Current Folder: %s Current File: %s Current Timeout: %d Activity Timeout: %d" % (basepath+folder[0], file, timeout, activitytimeout))
        #         break
        # break
    
    with multiprocessing.Pool(5) as p:
        p.starmap(util.ModifyPcap, tasks)
        p.close()
        p.join()

    for folder in folders:
        for file in os.listdir(basepath+folder):
            if not os.path.exists(output_folder+folder):
                os.mkdir(output_folder+folder)
            if file.endswith(".pcap"):
                os.system(f"mergecap -w {output_folder}{folder}/{file} {basepath}{folder}/{file} {im_folder}{folder}/{file}")