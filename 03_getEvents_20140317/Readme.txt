Readme file for getEvents using obspy
#####################################################
The files in this folder contains the tutorial of 
using Obspy to download data based on the arrival
time for different events. 

Author: Qingkai Kong, qingkai.kong@gmail.com

Modify history:
(1) Created by 2014-03-14

#####################################################
Goal this time:

(1) Find the catalog of the earthquakes
(2) Find the arrival time of different phases of the 
    events in the catalog with respect to the stations
(3) Download data based on the time window calcuated
    above



#####################################################
Structure of the folder

Obspy_getEvents.ipynb - this is the ipython notebook
    contains the main script to download the data 
    described above

Obspy_getCMT.ipynb - this is the ipython notebook 
    contains the script to get the moment tensors
    and plot on the map from global CMT, not totally
    finish this time, we may come back to this topic
    next time. Note: if you want to run the script in 
    this notebook, then you need install BeautifulSoup,
    run the following command:
    'sudo pip install BeautifulSoup'

#####################################################


