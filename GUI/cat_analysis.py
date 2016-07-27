import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as cc
import matplotlib as mpl
from mpl_toolkits.basemap import Basemap
from obspy.imaging.beachball import Beach
import matplotlib.cm as cm
import datetime
from matplotlib.lines import Line2D
from obspy.core.util import locations2degrees
import math
from matplotlib.cm import ScalarMappable
from datetime import timedelta
from obspy.fdsn import Client
from obspy import UTCDateTime
from StringIO import StringIO
import re
from obspy import readEvents
import pandas as pd
import urllib
import urllib2
import sys

from collections import OrderedDict

def get_hist_mt(t1, t2, llat = '-90', ulat = '90', llon = '-180', ulon = '180', 
    lmw = 0, umw = 10,
    evla = None, evlo = None, step = 2.0, list = '6'):
    yr = t1.year
    mo = t1.month
    day = t1.day
    oyr = t2.year
    omo = t2.month
    oday = t2.day
    mat = {}
    locs = locals()  
    
    base_url = 'http://www.globalcmt.org/cgi-bin/globalcmt-cgi-bin/CMT4/form'
    
    #note here, we must use the ordered Dictionary, so that the order in the 
    #url is exactly the same order
    param = OrderedDict()
    param['itype'] = 'ymd'
    param['yr'] = yr
    param['mo'] = mo
    param['day'] = day
    param['otype'] = 'ymd'
    param['oyr'] = oyr
    param['omo'] = omo
    param['oday'] = oday
    param['jyr'] = '1976'
    param['jday'] = '1'
    param['ojyr'] = '1976'
    param['ojday'] = '1'
    
    param['nday'] = '1'
    param['lmw'] = str(lmw)
    param['umw'] = str(umw)
    param['lms'] = '0'
    param['ums'] = '10'
    param['lmb'] = '0'
    param['umb'] = '10'
    
    ind = 1
    if evla and evlo is not None:
        llat = evla - step
        ulat = evla + step
        llon = evlo - step
        ulon = evlo + step
        ind = 0

    
    param['llat'] = llat
    param['ulat'] = ulat
    param['llon'] = llon
    param['ulon'] = ulon
    
    param['lhd'] = '0'
    param['uhd'] = '1000'
    param['lts'] = '-9999'
    param['uts'] = '9999'
    param['lpe1'] = '0'
    param['upe1'] = '90'
    param['lpe2'] = '0'
    param['upe2'] = '90'
    param['list'] = list
    
    url = "?".join((base_url, urllib.urlencode(param)))
    print url
    
    page = urllib2.urlopen(url)
    from BeautifulSoup import BeautifulSoup

#
    parsed_html = BeautifulSoup(page)

    mecs_str = parsed_html.findAll('pre')[1].text.split('\n')

    def convertString(mecs_str):
        return map(float, mecs_str.split()[:9])
        
    psmeca = np.array(map(convertString, mecs_str))
    
    mat['psmeca'] = psmeca
    mat['ind'] = ind
    mat['url'] = url
    mat['range'] = (llat, ulat, llon, ulon)
    mat['evloc'] = (evla, evlo)
    return mat
    
def plot_hist_mt(psmeca_dict, figsize = (16,24), mt_size = 10, pretty = False, resolution='l'):
    if psmeca_dict['psmeca'] != []:
        psmeca = psmeca_dict['psmeca']
        #get the latitudes, longitudes, and the 6 independent component
        lats = psmeca[:,1]
        lons = psmeca[:,0]
        focmecs = psmeca[:,3:9]
        depths =  psmeca[:,2]    
        (llat, ulat, llon, ulon) = psmeca_dict['range'] 
        evla = psmeca_dict['evloc'][0]
        evlo = psmeca_dict['evloc'][1]

        plt.figure(figsize=figsize)
        m = Basemap(projection='cyl', lon_0=142.36929, lat_0=38.3215, 
                    llcrnrlon=llon,llcrnrlat=llat,urcrnrlon=ulon,urcrnrlat=ulat,resolution=resolution)
    
        m.drawcoastlines()
        m.drawmapboundary()
    
        if pretty:    
            m.etopo()
        else:
            m.fillcontinents()
    
        llat = float(llat)
        ulat = float(ulat)
        llon = float(llon)
        ulon = float(ulon)
    
        m.drawparallels(np.arange(llat, ulat, (ulat - llat) / 4.0), labels=[1,0,0,0])
        m.drawmeridians(np.arange(llon, ulon, (ulon - llon) / 4.0), labels=[0,0,0,1])   
    
        ax = plt.gca()
    
        x, y = m(lons, lats)
    
        for i in range(len(focmecs)):
            '''
            if x[i] < 0:
                x[i] = 360 + x[i]
            '''
        
            if depths[i] <= 50:
                color = '#FFA500'
                #label_
            elif depths[i] > 50 and depths [i] <= 100:
                color = 'g'
            elif depths[i] > 100 and depths [i] <= 200:
                color = 'b'
            else:
                color = 'r'
        
            index = np.where(focmecs[i] == 0)[0]
        
            #note here, if the mrr is zero, then you will have an error
            #so, change this to a very small number 
            if focmecs[i][0] == 0:
                focmecs[i][0] = 0.001
        
            try:
                b = Beach(focmecs[i], xy=(x[i], y[i]),width=mt_size, linewidth=1, facecolor=color)
            except:
                pass
            
            b.set_zorder(10)
            ax.add_collection(b)
        
        x_0, y_0 = m(evlo, evla)
        m.plot(x_0, y_0, 'r*', markersize=25) 
    
        circ1 = Line2D([0], [0], linestyle="none", marker="o", alpha=0.6, markersize=10, markerfacecolor="#FFA500")
        circ2 = Line2D([0], [0], linestyle="none", marker="o", alpha=0.6, markersize=10, markerfacecolor="g")
        circ3 = Line2D([0], [0], linestyle="none", marker="o", alpha=0.6, markersize=10, markerfacecolor="b")
        circ4 = Line2D([0], [0], linestyle="none", marker="o", alpha=0.6, markersize=10, markerfacecolor="r")
        plt.legend((circ1, circ2, circ3, circ4), ("depth $\leq$ 50 km", "50 km $<$ depth $\leq$ 100 km", 
                        "100 km $<$ depth $\leq$ 200 km", "200 km $<$ depth"), numpoints=1, loc=3)
        plt.show()
    else:
        print 'No historical MT found!'

def eq2df(earthquakes):

    df = pd.DataFrame(earthquakes, columns= [ 'evla', 'evlo', 'evdp', 'mag', 'mag_type', 'event_type', 'date'] )
    #df = df.set_index('date', drop=True)  
    df.index = pd.to_datetime(df.pop('date'))
    df.index = [tmp.datetime for tmp in df.index] 
    return df 

def plot_cum(earthquakes, freq = '1D', figsize = (12,8)):
    '''
    B = business day frequency
    C = custom business day frequency (experimental)
    D = calendar day frequency
    W = weekly frequency
    M = month end frequency
    BM = business month end frequency
    MS = month start frequency
    BMS = business month start frequency
    Q = quarter end frequency
    BQ = business quarter endfrequency
    QS = quarter start frequency
    BQS = business quarter start frequency
    A = year end frequency
    BA = business year end frequency
    AS = year start frequency
    BAS = business year start frequency
    H = hourly frequency
    T = minutely frequency
    S = secondly frequency
    L = milliseonds
    U = microseconds
    '''

    fig = plt.figure(facecolor='white')
    ax1 = fig.add_subplot(111)
    df = eq2df(earthquakes)
    rs = df['mag'].resample(freq,how='count')
    rs.cumsum().plot(figsize=figsize)

    ax1.set_xlabel('Time')
    ax1.set_ylabel('Number of Earthquakes')
    fig.show()

def plot_seimicity_rate(earthquakes, time = 'hour', figsize = (8,8)):
    '''
    Function get from Thomas Lecocq
    http://www.geophysique.be/2013/09/25/seismicity-rate-using-obspy-and-pandas/
    '''
    
    m_range = (-1,11)
    time = time.lower()
    
    if time == 'second':
        time_format = '%y-%m-%d, %H:%M:%S %p'
    elif time == 'minute':
        time_format = '%y-%m-%d, %H:%M %p'
    elif time == 'hour':
        time_format = '%y-%m-%d, %H %p'
    elif time == 'day':
        time_format = '%y-%m-%d'
    elif time == 'month':
        time_format = '%y-%m'
    elif time == 'year':
        time_format = '%y'
    else:
        time_format = '%y-%m-%d, %H %p'
    
    df = eq2df(earthquakes)

    #Arrange quakes by their magnitude and color appropiately
    
    bins = np.arange(m_range[0], m_range[1])
    labels = np.array(["[%i:%i)"%(i,i+1) for i in bins])
    colors = [cm.hsv(float(i+1)/(len(bins)-1)) for i in bins]
    df['Magnitude_Range'] = pd.cut(df['mag'], bins=bins, labels=False)
    
    df['Magnitude_Range'] = labels[df['Magnitude_Range'].values]
    
    df['Year_Month_day'] = [di.strftime(time_format) for di in df.index]
    
    rate = pd.crosstab(df.Year_Month_day, df.Magnitude_Range)
    
    rate.plot(kind='bar',stacked=True,color=colors,figsize=figsize)
    plt.legend(loc='best')
    plt.ylabel('Number of earthquakes')
    plt.xlabel('Date and Time')
    plt.tight_layout()
    plt.grid()
    plt.show()

def get_event_info(catalog, M_above, llat, ulat, llon, ulon, color, label):
    lats = []
    lons = []
    labels = []
    mags = []
    colors = []
    times = []
    
    for i, event in enumerate(catalog):
        mag = np.float(event[3])
        lat = event[0]
        lon = event[1]
        
        if mag >= M_above and np.float(lat) >= np.float(llat) \
        and np.float(lat) <= np.float(ulat) and np.float(lon) >= np.float(llon) \
        and np.float(lon) <= np.float(ulon):
            lats.append(event[0])
            lons.append(event[1])
            mags.append(mag)
            times.append(event[6])
            labels.append(('    %.1f' % mag) if mag and label == 'magnitude'
                          else '')
            if color == 'date':
                colors.append(event[6])
            elif color == 'depth':
                colors.append(event[2])
    return lats, lons, mags, times, labels, colors
            
    

def get_cat(data_center = None, **kwargs):
    '''
    Function to get catalog data from different data center
    data_center - specify the data center i.e. 'IRIS'
    Other arguments you can use:
    :type starttime: :class:`~obspy.core.utcdatetime.UTCDateTime`, optional
        :param starttime: Limit to events on or after the specified start time.
        :type endtime: :class:`~obspy.core.utcdatetime.UTCDateTime`, optional
        :param endtime: Limit to events on or before the specified end time.
        :type minlatitude: float, optional
        :param minlatitude: Limit to events with a latitude larger than the
            specified minimum.
        :type maxlatitude: float, optional
        :param maxlatitude: Limit to events with a latitude smaller than the
            specified maximum.
        :type minlongitude: float, optional
        :param minlongitude: Limit to events with a longitude larger than the
            specified minimum.
        :type maxlongitude: float, optional
        :param maxlongitude: Limit to events with a longitude smaller than the
            specified maximum.
        :type latitude: float, optional
        :param latitude: Specify the latitude to be used for a radius search.
        :type longitude: float, optional
        :param longitude: Specify the longitude to the used for a radius
            search.
        :type minradius: float, optional
        :param minradius: Limit to events within the specified minimum number
            of degrees from the geographic point defined by the latitude and
            longitude parameters.
        :type maxradius: float, optional
        :param maxradius: Limit to events within the specified maximum number
            of degrees from the geographic point defined by the latitude and
            longitude parameters.
        :type mindepth: float, optional
        :param mindepth: Limit to events with depth more than the specified
            minimum.
        :type maxdepth: float, optional
        :param maxdepth: Limit to events with depth less than the specified
            maximum.
        :type minmagnitude: float, optional
        :param minmagnitude: Limit to events with a magnitude larger than the
            specified minimum.
        :type maxmagnitude: float, optional
        :param maxmagnitude: Limit to events with a magnitude smaller than the
            specified maximum.
        :type magnitudetype: str, optional
        :param magnitudetype: Specify a magnitude type to use for testing the
            minimum and maximum limits.
        :type includeallorigins: bool, optional
        :param includeallorigins: Specify if all origins for the event should
            be included, default is data center dependent but is suggested to
            be the preferred origin only.
        :type includeallmagnitudes: bool, optional
        :param includeallmagnitudes: Specify if all magnitudes for the event
            should be included, default is data center dependent but is
            suggested to be the preferred magnitude only.
        :type includearrivals: bool, optional
        :param includearrivals: Specify if phase arrivals should be included.
        :type eventid: str or int (dependent on data center), optional
        :param eventid: Select a specific event by ID; event identifiers are
            data center specific.
        :type limit: int, optional
        :param limit: Limit the results to the specified number of events.
        :type offset: int, optional
        :param offset: Return results starting at the event count specified,
            starting at 1.
        :type orderby: str, optional
        :param orderby: Order the result by time or magnitude with the
            following possibilities:
                * time: order by origin descending time
                * time-asc: order by origin ascending time
                * magnitude: order by descending magnitude
                * magnitude-asc: order by ascending magnitude
        :type catalog: str, optional
        :param catalog: Limit to events from a specified catalog
        :type contributor: str, optional
        :param contributor: Limit to events contributed by a specified
            contributor.
        :type updatedafter: :class:`~obspy.core.utcdatetime.UTCDateTime`,
            optional
        :param updatedafter: Limit to events updated after the specified time.
        :type filename: str or open file-like object
        :param filename: If given, the downloaded data will be saved there
            instead of being parse to an ObsPy object. Thus it will contain the
            raw data from the webservices.
    
    '''
    #get the catalog
    if data_center is None:
        data_center = 'USGS'
        
    client = Client(data_center)
    sio = StringIO()
    #save the catalog into a StringIO object
    cat = client.get_events(filename=sio, **kwargs)
    
    #specify the entries you want to replace with (the inconsistent ones) in the following dictionary
    rep = {"quarry_blast": "quarry blast", "quarry": "quarry blast", "quarry blast_blast":"quarry blast" }
    
    #replace the multiple entries, and save the modified entries into StringIO object
    rep = dict((re.escape(k), v) for k, v in rep.iteritems())
    pattern = re.compile("|".join(rep.keys()))
    
    sio.seek(0)
    sio2 = StringIO()
    sio2.write(pattern.sub(lambda m: rep[re.escape(m.group(0))], sio.buf))
    
    #read the catalog from this StringIO object
    sio2.seek(0)
    cat = readEvents(sio2)
    return cat

def cat2list(cat, mt_type = 'Moment_tensor'):
    '''
    Function to convert catalog object to list (easy work with)
    Input:
        cat - catalog object
        mt_type - flag of getting 'Focal' or 'Moment_tensor'
    Return:
        earthquakes - list of earthquake information
        mt - list of focal/moment_tensor information
        event_id - event ID corresponding to the earthquakes
        quarry_blast - list of quarry blast list
    '''
    
    eq_matrix = []
    evtime_mat = []
    mt = []
    event_id =[]
    
    #mt_type can be:
    #'Focal'
    #'Moment_tensor'
    #'Both'
    
    for index in range(cat.count()):
        myevent = cat[index]
        
        try:
        
            origins = myevent.origins[0]
            evtime_mat.append(origins.time)
            evla = origins.latitude
            evlo = origins.longitude
            evdp = origins.depth / 1000.
            #quality = origins.quality
            mag_type = myevent.magnitudes[0].magnitude_type
            mag = myevent.magnitudes[0].mag
            event_type = myevent.event_type

            if myevent.focal_mechanisms != []:  
                
                if mt_type == 'Moment_tensor':
                    moment_tensor = myevent.focal_mechanisms[0].moment_tensor.tensor
                    if moment_tensor is not None:
                        eventid = myevent['resource_id'].id.split('&')[0].split('=')[1]
                        m_rr = moment_tensor.m_rr
                        m_tt = moment_tensor.m_tt
                        m_pp = moment_tensor.m_pp
                        m_rt = moment_tensor.m_rt
                        m_rp = moment_tensor.m_rp
                        m_tp = moment_tensor.m_tp
                        
                        #m_rr=3.315e+16, m_tt=-6.189e+16, m_pp=2.874e+16, m_rt=-5311000000000000.0, m_rp=-1.653e+16, m_tp=5044000000000000.0
                        mt.append([evla, evlo, evdp, mag, m_rr, m_tt,m_pp,m_rt,m_rp,m_tp])
                        event_id.append(eventid)
                elif mt_type == 'Focal':
                    nodal_plane = myevent.focal_mechanisms[0].nodal_planes.nodal_plane_1
                    if nodal_plane is not None:
                        eventid = myevent['resource_id'].id.split('&')[0].split('=')[1]
                        mt.append([evla, evlo, evdp, mag, nodal_plane.strike, nodal_plane.dip, nodal_plane.rake])
                        event_id.append(eventid)
                elif mt_type == 'Both':
                    
                    nodal_plane = myevent.focal_mechanisms[0].nodal_planes.nodal_plane_1
                    if nodal_plane is not None:
                        eventid = myevent['resource_id'].id.split('&')[0].split('=')[1]
                        mt.append([evla, evlo, evdp, mag, nodal_plane.strike, nodal_plane.dip, nodal_plane.rake])
                        event_id.append(eventid)
                        break
                    else:
                        pass
                    
                    moment_tensor = myevent.focal_mechanisms[0].moment_tensor.tensor
                    if moment_tensor is not None:
                        eventid = myevent['resource_id'].id.split('&')[0].split('=')[1]
                        m_rr = moment_tensor.m_rr
                        m_tt = moment_tensor.m_tt
                        m_pp = moment_tensor.m_pp
                        m_rt = moment_tensor.m_rt
                        m_rp = moment_tensor.m_rp
                        m_tp = moment_tensor.m_tp
                        
                        #m_rr=3.315e+16, m_tt=-6.189e+16, m_pp=2.874e+16, m_rt=-5311000000000000.0, m_rp=-1.653e+16, m_tp=5044000000000000.0
                        mt.append([evla, evlo, evdp, mag, m_rr, m_tt,m_pp,m_rt,m_rp,m_tp])
                        event_id.append(eventid)
                    
        
            eq_matrix.append((evla, evlo, evdp, mag, mag_type, event_type, origins.time))

        except:
            continue
        #print mt
        #evla = origins[1]
    
    quarry_blast = [tmp for tmp in eq_matrix if tmp[5] == 'quarry blast']
    earthquakes =  [tmp for tmp in eq_matrix if tmp[5] == 'earthquake']
    
    evtime = [tmp.datetime for tmp in evtime_mat]
    return earthquakes, mt, event_id, quarry_blast

def check_collision(lats, lons, radius, dist_bt, angle_step):
    '''
    This function will check if the moment tensors are collide on 
    the plot, if there are collisions, then we will put all the
    MT on a circle lined back to the location of the event
    
    input: 
    lats - list of latitudes of the events
    lons - list of longitudes of the events
    radius - radius of the circle to put the MT
    dist_bt - the distance difference between MTs to be checked, if
        the distance between the two MTs is smaller than dist_bt, 
        then we will put them on a circle
    angle_step - the increase of angle on the circle to put the
        MTs. 
        
    returns:
    lats_m - list of latitudes of the modified location
    lons_m - list of longitudes of the modified location 
    indicator - a list of flag showing which events we modified
    '''
    
    lats_m = np.zeros(len(lats))
    lons_m = np.zeros(len(lats))
    indicator = np.zeros(len(lats))
    
    angles = range(0,360,angle_step)
    j = 0
    for i in range(len(lats)-1):
        for k in range(i+1, len(lats)):
            dist = locations2degrees(lats[i], lons[i], lats[k], lons[k]) * 111
            if dist < dist_bt:
                indicator[i] =1
                indicator[k] =1
            else:
                pass
    
    for i in range(len(lats)):
        if indicator[i] == 1:
    
            ix = j%len(angles)
            a = radius*np.cos(angles[ix]*math.pi/180)
            b = radius*np.sin(angles[ix]*math.pi/180)
            lats_m[i] = lats[i] + a
            lons_m[i] = lons[i] + b
            j +=1
        else:
            lats_m[i] = lats[i]
            lons_m[i] = lons[i] 
    
    return lats_m, lons_m, indicator

def plot_mt(mapobj,axisobj,figobj,earthquakes, mt, event_id, location = None, M_above = 3.0, show_above_M = True, 
            llat = '-90', ulat = '90', llon = '-170', ulon = '190', figsize = (12,8),
            radius = 25, dist_bt = 600, mt_width = 2, angle_step = 20, show_eq = True, 
            par_range = (-90., 120., 30.), mer_range = (0, 360, 60),
            pretty = False,  legend_loc = 4, title = '', resolution = 'l'):
    '''
    Function to plot moment tensors on the map
    Input:
        mapobj - already existing Basemap object onwhich to plot the quakes
        earthquakes - list of earthquake information
        mt - list of focal/moment_tensor information
        event_id - event ID corresponding to the earthquakes 
        location - predefined region, choose from 'US' or 'CA', 
            default is 'None' which will plot the whole world
        M_above - Only show the events with magnitude larger than this number
            default is 5.0, use with show_above_M 
        show_above_M - Flag to turn on the M_above option, default is True, 
        llat - bottom left corner latitude, default is -90
        ulat - upper right corner latitude, default is 90
        llon - bottom left corner longitude, default is -170 
        ulon - upper right corner longitude, default is 190
        figsize - figure size, default is (12,8)
        radius - used in checking collisions (MT), put the MT on a circle with this 
            radius, default is 25
        dist_bt - used in checking collisions (MT), if two events within dist_bt km, 
            then we say it is a collision, default is 600
        angle_step - used in checking collisions (MT), this is to decide the angle 
            step on the circle, default is 20 degree 
        mt_width - size of the MT on the map. Different scale of the map may need 
            different size, play with it.
        show_eq -  flag to show the seismicity as well, default is True
        par_range - range of latitudes you want to label on the map, start lat, end lat
            and step size, default is (-90., 120., 30.),
        mer_range - range of longitudes you want to label on the map, start lon, end lon
            and step size, default is (0, 360, 60),
        pretty - draw a pretty map, default is False to make faster plot
        legend_loc - location of the legend, default is 4
        title - title of the plot
        resolution - resolution of the map,  Possible values are
            * ``"c"`` (crude)
            * ``"l"`` (low)
            * ``"i"`` (intermediate)
            * ``"h"`` (high)
            * ``"f"`` (full)
            Defaults to ``"l"``
    '''
    
    if location == 'US':
        llon=-125
        llat=20
        ulon=-70
        ulat=60
        M_above = 4.0
        radius = 5
        dist_bt = 200 
        par_range = (20, 60, 15)
        mer_range = (-120, -60, 15)
        mt_width = 0.8        
        drawCountries = True
        
    elif location == 'CAL':
        llat = '30'
        ulat = '45'
        llon = '-130'
        ulon = '-110'
        M_above = 3.0
        radius = 1.5
        dist_bt = 50
        mt_width = 0.3
        drawStates = True
    else:
        location = None
        
        
    times = [event[6] for event in earthquakes]  

    
    if show_above_M:
        mags = [row[3] for row in mt]
        index = np.array(mags) >= M_above
        mt_select = np.array(mt)[index]
        evid = np.array(event_id)[index]
        times_select = np.array(times)[index]
    else:
        evid = [row for row in event_id]
        times_select = times
        mt_select = mt

    
    lats = [row[0] for row in mt_select]
    lons = [row[1] for row in mt_select]
    depths = [row[2] for row in mt_select]
    mags =  [row[3] for row in mt_select]
    focmecs = [row[4:] for row in mt_select]
    
    lats_m, lons_m, indicator = check_collision(lats, lons, radius, dist_bt, angle_step) 

    #sys.exit(1) 
    
    count = 0
    
    colors=[]
    
    min_color = min(times_select)
    max_color = max(times_select)
    colormap = plt.get_cmap()
    
    for i in times_select:
        colors.append(i)
    
    scal_map = ScalarMappable(norm=cc.Normalize(min_color, max_color),cmap=colormap)
    scal_map.set_array(np.linspace(0, 1, 1))
    
    colors_plot = [scal_map.to_rgba(c) for c in colors]
     
    ys = np.array(lats_m)
    xs = np.array(lons_m)
    url = ['http://earthquake.usgs.gov/earthquakes/eventpage/' + tmp + '#summary' for tmp in evid]
    
    stnm = np.array(evid)    
    
    #fig, ax1 = plt.subplots(1,1, figsize = figsize)
    #map_ax = fig.add_axes([0.03, 0.13, 0.94, 0.82])
    
    # if show_eq:
    #     cm_ax = fig.add_axes([0.98, 0.39, 0.04, 0.3])   
    #     plt.sca(ax1)
    #     cb = mpl.colorbar.ColorbarBase(ax=cm_ax, cmap=colormap, orientation='vertical')
    #     cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    #     color_range = max_color - min_color
    #     cb.set_ticklabels([_i.strftime('%Y-%b-%d, %H:%M:%S %p')
    #     for _i in [min_color, min_color + color_range * 0.25,
    #        min_color + color_range * 0.50,
    #        min_color + color_range * 0.75, max_color]])
           
    #m = Basemap(projection='cyl', lon_0=142.36929, lat_0=38.3215, llcrnrlon=llon,llcrnrlat=llat,urcrnrlon=ulon,urcrnrlat=ulat,resolution=resolution)
    
    #mapobj.drawmapboundary()
    #mapobj.drawcountries()
    #mapobj.drawparallels(np.arange(par_range[0], par_range[1], par_range[2]), labels=[1,0,0,0], linewidth=0)
    #mapobj.drawmeridians(np.arange(mer_range[0],mer_range[1], mer_range[2]), labels=[0,0,0,1], linewidth=0)
    
        
    x, y = mapobj(lons_m, lats_m)

    mtlineobjs = []
    mtobjs = []
    
    for i in range(len(focmecs)):
            
        index = np.where(focmecs[i] == 0)[0]
        
        #note here, if the mrr is zero, then you will have an error
        #so, change this to a very small number 
        if focmecs[i][0] == 0:
            focmecs[i][0] = 0.001
            
        
        width = mags[i] * mt_width
        
        if depths[i] <= 50:
            color = '#FFA500'
            #label_
        elif depths[i] > 50 and depths [i] <= 100:
            color = '#FFFF00'
        elif depths[i] > 100 and depths [i] <= 150:
            color = '#00FF00'
        elif depths[i] > 150 and depths [i] <= 200:
            color = 'b'
        else:
            color = 'r'
                  
        if indicator[i] == 1:
            lineobj = mapobj.plot([lons[i],lons_m[i]],[lats[i], lats_m[i]], 'k') 
            #mtlineobjs.append(lineobj)
            #m.plot([10,20],[0,0])  
        else:

            lineobj = None

        try:
            
            b = Beach(focmecs[i], xy=(x[i], y[i]),width=width, linewidth=1, facecolor= color, alpha=1)
            count += 1
            #line = axisobj.plot(x[i],y[i], 'o', picker=5, markersize=30, alpha =0)
            #mtlineobjs.append(line) 
    
        except:
            pass
        b.set_zorder(3)
        axisobj.add_collection(b)
        mtobjs.append(b)
        mtlineobjs.append(lineobj)
        
        
    d=2
    circ1 = Line2D([0], [0], linestyle="none", marker="o", alpha=0.6, markersize=10, markerfacecolor="#FFA500")
    circ2 = Line2D([0], [0], linestyle="none", marker="o", alpha=0.6, markersize=10, markerfacecolor="#FFFF00")
    circ3 = Line2D([0], [0], linestyle="none", marker="o", alpha=0.6, markersize=10, markerfacecolor="#00FF00")
    circ4 = Line2D([0], [0], linestyle="none", marker="o", alpha=0.6, markersize=10, markerfacecolor="b")
    circ5 = Line2D([0], [0], linestyle="none", marker="o", alpha=0.6, markersize=10, markerfacecolor="r")
    
    M4 = Line2D([0], [0], linestyle="none", marker="o", alpha=0.4, markersize= 4*d, markerfacecolor="k")
    M5 = Line2D([0], [0], linestyle="none", marker="o", alpha=0.4, markersize= 5*d, markerfacecolor="k")
    M6 = Line2D([0], [0], linestyle="none", marker="o", alpha=0.4, markersize= 6*d, markerfacecolor="k")
    M7 = Line2D([0], [0], linestyle="none", marker="o", alpha=0.4, markersize= 7*d, markerfacecolor="k")
    
    if location == 'World':
    
        title = str(count) + ' events with focal mechanism - color codes depth, size the magnitude'
        
    elif location == 'US':
    
        title = 'US events with focal mechanism - color codes depth, size the magnitude'
        
    elif location == 'CAL':
        title = 'California events with focal mechanism - color codes depth, size the magnitude'
    elif location is None:
        pass

    #handles, labels = axisobj.get_legend_handles_labels()
        
    #axisobj.legend(handles,labels,(circ1, circ2, circ3, circ4, circ5), ("depth $\leq$ 50 km", "50 km $<$ depth $\leq$ 100 km", 
    #                                   "100 km $<$ depth $\leq$ 150 km", "150 km $<$ depth $\leq$ 200 km","200 km $<$ depth"), numpoints=1, loc=legend_loc)
    
    #plt.title(title)
    #plt.gca().add_artist(legend1)
    
    #if location == 'World':
    #    axisobj.legend(han,dles,labels(M4,M5,M6,M7), ("M 4.0", "M 5.0", "M 6.0", "M 7.0"), numpoints=1, loc=legend_loc)
        
    x, y = mapobj(lons, lats)
    min_size = 1
    max_size = 8
    min_mag = min(mags)
    max_mag = max(mags)
    
    if show_eq:
        if len(lats) > 1:
            frac = [(_i - min_mag) / (max_mag - min_mag) for _i in mags]
            magnitude_size = [(_i * (max_size - min_size)) ** 2 for _i in frac]
            magnitude_size = [(_i * min_size/2)**2 for _i in mags]
        else:
            magnitude_size = 15.0 ** 2
            colors_plot = "red"
        
        quakedots = mapobj.scatter(x, y, marker='o', s=magnitude_size, c=colors_plot,
                zorder=10)
    
    print 'Max magnitude ' + str(np.max(mags)), 'Min magnitude ' + str(np.min(mags))


    return mtlineobjs,mtobjs,quakedots,xs,ys,url


def plot_events(mapobj,axisobj,catalog,label= None, color='depth', pretty = False, colormap=None, 
             llat = -90, ulat = 90, llon = -180, ulon = 180, figsize=(16,24), 
             par_range = (-90., 120., 30.), mer_range = (0., 360., 60.),
             showHour = False, M_above = 0.0, location = 'World', min_size=1, max_size=8,**kwargs):

    '''Simplified version of plot_event'''

    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    lats, lons, mags, times, labels, colors = get_event_info(catalog, M_above, llat, ulat, llon, ulon, color, label)
                    
    min_color = min(colors)
    max_color = max(colors)

    if colormap is None:
        if color == "date":
             colormap = plt.get_cmap()
        else:
            # Choose green->yellow->red for the depth encoding.
             colormap = plt.get_cmap("RdYlGn_r")
                
    scal_map = ScalarMappable(norm=Normalize(min_color, max_color),
                                  cmap=colormap)
    scal_map.set_array(np.linspace(0, 1, 1))

    x, y = mapobj(lons, lats)

    min_mag = 0
    max_mag = 10
    if len(mags) > 1:
        frac = [(_i - min_mag) / (max_mag - min_mag) for _i in mags]
        magnitude_size = [(_i * (max_size - min_size)) ** 2 for _i in frac]
        #magnitude_size = [(_i * min_size) for _i in mags]
        #print magnitude_size
        colors_plot = [scal_map.to_rgba(c) for c in colors]
    else:
        magnitude_size = 15.0 ** 2
        colors_plot = "red"

    quakes = mapobj.scatter(x, y, marker='o', s=magnitude_size, c=colors_plot, zorder=10)
    #mapobj.drawmapboundary(fill_color='aqua')
    #mapobj.drawparallels(np.arange(-90,90,30),labels=[1,0,0,0])
    #mapobj.drawmeridians(np.arange(mapobj.lonmin,mapobj.lonmax+30,60),labels=[0,0,0,1])

    # if len(mags) > 1:
    #     cb = mpl.colorbar.ColorbarBase(ax=axisobj, cmap=colormap, orientation='vertical')
    #     cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    #     color_range = max_color - min_color
    #     cb.set_ticklabels([_i.strftime('%Y-%b-%d') if color == "date" else '%.1fkm' % (_i)
    #             for _i in [min_color, min_color + color_range * 0.25,
    #                    min_color + color_range * 0.50,
    #                    min_color + color_range * 0.75, max_color]])

    return quakes

def determinemtsize(minlon,maxlon,minlat,maxlat):

    '''Determine the size of the moment tensor beachball and event circle to plot based on the map boundary'''


    mt_width = round(((maxlat-minlat)/55.0),3)
    mt_rad = round(((maxlat-minlat)/15.0),1)

    if mt_width < 0.01:
        print mt_width
        mt_width = 0.01

    min_dot = round(((maxlat-minlat)/200.0),1)
    max_dot = round(((maxlat-minlat)/150.0),1)

    min_quake = round(((maxlat-minlat)/15.0),2)
    max_quake = round(((maxlat-minlat)/5.0),2)

    return mt_width,mt_rad,min_dot,max_dot,min_quake,max_quake



def read_faults(input_file):
    '''
    Read in the faults file to plot the faults - used for adding tectonic plate boundaries
    '''
    count = 0
    faults = {}
    lats = []
    lons = []
    for line in open(input_file):
        # Check for empty/commented lines
        if not line or line.startswith('>'):
            # If 1st one: new block
            if count == 0:
                pass
            count += 1
        # Non empty line: add line in current(last) block
            faults[count] = [lons, lats]
            lats = []
            lons = []
            
        else:
            try:
                lon = float(line.split()[1])
                lat = float(line.split()[0])
                lats.append(lat)
                lons.append(lon)
            except:
                print 'error in reading coordinate: skipped'            
            
    return faults



def plot_event(catalog, projection='cyl', resolution='l',
             continent_fill_color='0.9', water_fill_color='white',
             label= None, color='depth', pretty = False, colormap=None, 
             llat = -90, ulat = 90, llon = -180, ulon = 180, figsize=(16,24), 
             par_range = (-90., 120., 30.), mer_range = (0., 360., 60.),
             showHour = False, M_above = 0.0, location = 'World', **kwargs):  # @UnusedVariable
        """
        Creates preview map of all events in current Catalog object.

        :type projection: str, optional
        :param projection: The map projection. Currently supported are
            * ``"cyl"`` (Will plot the whole world.)
            * ``"ortho"`` (Will center around the mean lat/long.)
            * ``"local"`` (Will plot around local events)
            Defaults to "cyl"
        :type resolution: str, optional
        :param resolution: Resolution of the boundary database to use. Will be
            based directly to the basemap module. Possible values are
            * ``"c"`` (crude)
            * ``"l"`` (low)
            * ``"i"`` (intermediate)
            * ``"h"`` (high)
            * ``"f"`` (full)
            Defaults to ``"l"``
        :type continent_fill_color: Valid matplotlib color, optional
        :param continent_fill_color:  Color of the continents. Defaults to
            ``"0.9"`` which is a light gray.
        :type water_fill_color: Valid matplotlib color, optional
        :param water_fill_color: Color of all water bodies.
            Defaults to ``"white"``.
        :type label: str, optional
        :param label:Events will be labeld based on the chosen property.
            Possible values are
            * ``"magnitude"``
            * ``None``
            Defaults to ``"magnitude"``
        :type color: str, optional
        :param color:The events will be color-coded based on the chosen
            proberty. Possible values are
            * ``"date"``
            * ``"depth"``
            Defaults to ``"depth"``
        :type colormap: str, optional, any matplotlib colormap
        :param colormap: The colormap for color-coding the events.
            The event with the smallest property will have the
            color of one end of the colormap and the event with the biggest
            property the color of the other end with all other events in
            between.
            Defaults to None which will use the default colormap for the date
            encoding and a colormap going from green over yellow to red for the
            depth encoding.

        .. rubric:: Example

        >>> cat = readEvents( \
            "http://www.seismicportal.eu/services/event/search?magMin=8.0") \
            # doctest:+SKIP
        >>> cat.plot()  # doctest:+SKIP
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable
        import matplotlib as mpl

        if color not in ('date', 'depth'):
            raise ValueError('Events can be color coded by date or depth. '
                             "'%s' is not supported." % (color,))
        if label not in (None, 'magnitude', 'depth'):
            raise ValueError('Events can be labeled by magnitude or events can'
                             ' not be labeled. '
                             "'%s' is not supported." % (label,))
        
        if location == 'US':
            llon=-125
            llat=20
            ulon=-60
            ulat=60
            lat_0=38
            lon_0=-122.0
            par_range = (20, 61, 10)
            mer_range = (-120, -59, 20)
            
        elif location == 'CA':
            llat = '30'
            ulat = '45'
            llon = '-130'
            ulon = '-110'
            lat_0=38
            lon_0=-122.0
            par_range = (30, 46, 5)
            mer_range = (-130, -109, 10)
        else:
            lat_0=0
            lon_0=0
            
        lats, lons, mags, times, labels, colors = get_event_info(catalog, M_above, llat, ulat, llon, ulon, color, label)
                    
        min_color = min(colors)
        max_color = max(colors)

        # Create the colormap for date based plotting.
        if colormap is None:
            if color == "date":
                colormap = plt.get_cmap()
            else:
                # Choose green->yellow->red for the depth encoding.
                colormap = plt.get_cmap("RdYlGn_r")
                
        scal_map = ScalarMappable(norm=Normalize(min_color, max_color),
                                  cmap=colormap)
        scal_map.set_array(np.linspace(0, 1, 1))

        #fig = plt.figure(figsize = figsize)
        # The colorbar should only be plotted if more then one event is
        # present.
        #if len(catalog) > 1:
        #    map_ax = fig.add_axes([0.03, 0.13, 0.94, 0.82])
            #cm_ax = fig.add_axes([0.03, 0.05, 0.94, 0.05])
            #rect = [left, bottom, width, height]
        #    cm_ax = fig.add_axes([0.98, 0.39, 0.04, 0.3])
        #    plt.sca(map_ax)
        #else:
        #    map_ax = fig.add_axes([0.05, 0.05, 0.90, 0.90])

        if projection == 'cyl':
            map = Basemap(resolution=resolution, lat_0 = lat_0, lon_0 = lon_0,
                        llcrnrlon=llon,llcrnrlat=llat,urcrnrlon=ulon,urcrnrlat=ulat)
        elif projection == 'ortho':
            map = Basemap(projection='ortho', resolution=resolution,
                          area_thresh=1000.0, lat_0=sum(lats) / len(lats),
                          lon_0=sum(lons) / len(lons))
        elif projection == 'local':
            if min(lons) < -150 and max(lons) > 150:
                max_lons = max(np.array(lons) % 360)
                min_lons = min(np.array(lons) % 360)
            else:
                max_lons = max(lons)
                min_lons = min(lons)
            lat_0 = (max(lats) + min(lats)) / 2.
            lon_0 = (max_lons + min_lons) / 2.
            if lon_0 > 180:
                lon_0 -= 360
            deg2m_lat = 2 * np.pi * 6371 * 1000 / 360
            deg2m_lon = deg2m_lat * np.cos(lat_0 / 180 * np.pi)
            if len(lats) > 1:
                height = (max(lats) - min(lats)) * deg2m_lat
                width = (max_lons - min_lons) * deg2m_lon
                margin = 0.2 * (width + height)
                height += margin
                width += margin
            else:
                height = 2.0 * deg2m_lat
                width = 5.0 * deg2m_lon
            
            map = Basemap(projection='aeqd', resolution=resolution,
                          area_thresh=1000.0, lat_0=lat_0, lon_0=lon_0,
                          width=width, height=height)
            # not most elegant way to calculate some round lats/lons

            def linspace2(val1, val2, N):
                """
                returns around N 'nice' values between val1 and val2
                """
                dval = val2 - val1
                round_pos = int(round(-np.log10(1. * dval / N)))
                delta = round(2. * dval / N, round_pos) / 2
                new_val1 = np.ceil(val1 / delta) * delta
                new_val2 = np.floor(val2 / delta) * delta
                N = (new_val2 - new_val1) / delta + 1
                return np.linspace(new_val1, new_val2, N)
            N1 = int(np.ceil(height / max(width, height) * 8))
            N2 = int(np.ceil(width / max(width, height) * 8))
            map.drawparallels(linspace2(lat_0 - height / 2 / deg2m_lat,
                                        lat_0 + height / 2 / deg2m_lat, N1),
                              labels=[0, 1, 1, 0])
            if min(lons) < -150 and max(lons) > 150:
                lon_0 %= 360
            meridians = linspace2(lon_0 - width / 2 / deg2m_lon,
                                  lon_0 + width / 2 / deg2m_lon, N2)
            meridians[meridians > 180] -= 360
            map.drawmeridians(meridians, labels=[1, 0, 0, 1])
        else:
            msg = "Projection %s not supported." % projection
            raise ValueError(msg)

        # draw coast lines, country boundaries, fill continents.
        map.drawcoastlines(color="0.4")
        map.drawcountries(color="0.75")
        if location == 'CA' or location == 'US':
            map.drawstates(color="0.75")
        
    
        # draw lat/lon grid lines
        map.drawparallels(np.arange(par_range[0], par_range[1], par_range[2]), labels=[1,0,0,0], linewidth=0)
        map.drawmeridians(np.arange(mer_range[0],mer_range[1], mer_range[2]), labels=[0,0,0,1], linewidth=0)

        if pretty:
            map.etopo()
        else:
            map.drawmapboundary(fill_color=water_fill_color)
            map.fillcontinents(color=continent_fill_color,
                           lake_color=water_fill_color)
        
        # compute the native map projection coordinates for events.
        x, y = map(lons, lats)
        # plot labels
        if 100 > len(mags) > 1:
            for name, xpt, ypt, colorpt in zip(labels, x, y, colors):
                # Check if the point can actually be seen with the current map
                # projection. The map object will set the coordinates to very
                # large values if it cannot project a point.
                if xpt > 1e25:
                    continue
                plt.text(xpt, ypt, name, weight="heavy",
                         color=scal_map.to_rgba(colorpt))
        elif len(mags) == 1:
            plt.text(x[0], y[0], labels[0], weight="heavy", color="red")
        min_size = 6
        max_size = 30
        min_mag = min(mags)
        max_mag = max(mags)
        if len(mags) > 1:
            frac = [(_i - min_mag) / (max_mag - min_mag) for _i in mags]
            magnitude_size = [(_i * (max_size - min_size)) ** 2 for _i in frac]
            #magnitude_size = [(_i * min_size) for _i in mags]
            #print magnitude_size
            colors_plot = [scal_map.to_rgba(c) for c in colors]
        else:
            magnitude_size = 15.0 ** 2
            colors_plot = "red"
        map.scatter(x, y, marker='o', s=magnitude_size, c=colors_plot,
                    zorder=10)
                    
        if len(mags) > 1:
            plt.title(
                "{event_count} events ({start} to {end}) "
                "- Color codes {colorcode}, size the magnitude".format(
                    event_count=len(lats),
                    start=min(times).strftime("%Y-%m-%d"),
                    end=max(times).strftime("%Y-%m-%d"),
                    colorcode="origin time" if color == "date" else "depth"))
        else:
            plt.title("Event at %s" % times[0].strftime("%Y-%m-%d"))

        # Only show the colorbar for more than one event.
        if len(mags) > 1:
            cb = mpl.colorbar.ColorbarBase(ax=cm_ax, cmap=colormap, orientation='vertical')
            cb.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
            color_range = max_color - min_color
            if showHour:
                cb.set_ticklabels([
                _i.strftime('%Y-%b-%d, %H:%M:%S %p') if color == "date" else '%.1fkm' %
                (_i)
                    for _i in [min_color, min_color + color_range * 0.25,
                           min_color + color_range * 0.50,
                           min_color + color_range * 0.75, max_color]])
            else:
                cb.set_ticklabels([_i.strftime('%Y-%b-%d') if color == "date" else '%.1fkm' % (_i)
                    for _i in [min_color, min_color + color_range * 0.25,
                           min_color + color_range * 0.50,
                           min_color + color_range * 0.75, max_color]])

        plt.show()