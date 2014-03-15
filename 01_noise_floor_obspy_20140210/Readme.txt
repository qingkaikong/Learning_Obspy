Readme file for the noise floor using obspy
#####################################################
The files in this folder contains all the files needed
to plot the noise floor of stations from NCEDC

Author: Qingkai Kong, qingkai.kong@gmail.com

Modify history:
(1) Created by 2014-02-09

#####################################################
Structure of the folder

Obspy_noise_floor.ipynb - this is the ipython notebook
    contains the main script to plot the figure, run 
    thecells in it step by step

build_url.py - this file contains all the functions 
    to create the url point to the dataless seed file
    at ncedc web. 
    
parse_ncedc.py - this file contains all the functions
    needed to parse the pole and zero information of 
    the station based on the url created by the above
    build_url.py. Actually, this file is almost the 
    same as the parser.py file from obspy.xseed folder.
    The only difference is that, I changed line 474 in
    this file:
    changed from 
    elif blkt.id == 53 or blkt.id == 60:
    to:
    elif blkt.id == 53 or blkt.id == 61:
    
    The reason for the change is that, the dataless 
    seed file from ncedc contains both blkt.id 53 and 
    60, and cause an error in the following code. It
    seems the error comes from the slightly different
    formate in 60. This is why I changed to 61 to 
    avoid the error (a quick dirty way, may cause 
    trouble when move to other seed files). But for 
    now let's just use this way to have a quick plot. 

#####################################################
Method used to generate the noise floor:

Check this paper:
McNamara, D. E. and Buland, R. P. (2004),
Ambient Noise Levels in the Continental United States,
Bulletin of the Seismological Society of America, 
94 (4), 1517-1527.
http://www.bssaonline.org/content/94/4/1517.abstract



