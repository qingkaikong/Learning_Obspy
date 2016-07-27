#!/usr/bin/env python 

#RMS 2016 
#Functions to do some basic earthquake statisitcs using a catalog downloaded by the QuakeWatch program

import cat_analysis as quaketools
from obspy import UTCDateTime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def write2file(quakes,mts):

	'''Write earthquake and mt information to file, ready to be manipulated wuth GMT (for making profiles, for example)'''

	outfile1 = open('quakestmp.dat','w')
	outfile2 = open('mttmp.dat','w')

	#Write event information to file
	for element in quakes:
		lat = element[0]
		lon = element[1]

		if lon < 0:
			lon = 360+lon

		depth = element[2]

		outfile1.write('%g %g %g\n' %(lon,lat,depth))

	outfile1.close()

	#Write mt informaton to file
	for element in mts:
		lat = element[0]
		lon = element[1]

		if lon < 0:
			lon = lon+360

		depth = element[2]
		mag = element[3]
		mrr = element[4]
		mtt = element[5]
		mpp = element[6]
		mtr = element[7]
		mrp = element[8]
		mtp = element[9]

		outfile2.write('%g %g %g %g %g %g %g %g %g %g\n' %(lon,lat,depth,mag,mrr,mtt,mpp,mtr,mrp,mtp))

	outfile2.close()

	print 'Written files!'


def quakeswithtime(quakes,freq='day'):

	'''plots a histogram of quake activity'''

	#plot a cumulative earthquake number graph with frequency of 1 day
	quaketools.plot_seimicity_rate(quakes,time=freq)

def cumulativemoment(quakes):

	'''Plot cumulative moment over time from start to end of the requerst period'''

	df = pd.DataFrame(quakes, columns= [ 'evla', 'evlo', 'evdp', 'mag', 'mag_type', 'event_type', 'date'] )

	starttime = df.ix[:,6].min()

	times = np.array(df.ix[:,6].values)-starttime
	mags = np.array(df.ix[:,3].values)

	maxmag = max(mags)

	timesplot = []
	momentsplot = []
	quakenumberplot = []
	totalquakes = 0
	totalmoment = 0

	for element in sorted(zip(times,mags)):
		timesplot.append(element[0]/3600.0) #time measured in hrs
		mag = float(element[1])

		mo = 10**((3/2.0)*(mag+10.7))
		mo = mo*10**(-7) #convert to Nm 
		totalmoment = mo + totalmoment
		totalquakes += 1

		if mag == maxmag:
			mt = totalquakes
			mm = element[0]/3600.0

		momentsplot.append(totalmoment)
		quakenumberplot.append(totalquakes)

	fig = plt.figure(facecolor='white')
	ax1 = fig.add_subplot(211)
	ax1.plot(timesplot,momentsplot)
	ax1.set_xlabel('Time in hours from first event',fontsize=12)
	ax1.set_ylabel('Cumulative moment release (Nm)',fontsize=12)

	ax2 = fig.add_subplot(212)
	ax2.plot(timesplot,quakenumberplot)
	ax2.plot(mm,mt,'ro')
	ax2.set_xlabel('Time in hours from first event',fontsize=12)
	ax2.set_ylabel('Cumulative number of events',fontsize=12)
	fig.show()

def depthslicequakes(quakes,mts,startlon,startlat,endlon,endlat):

	'''Use gmt commands to make a slice though earthquakes in a particular cross section, and plot'''

	write2file(quakes, mts)

	minlon = min(startlon,endlon)
	maxlon = max(startlon,endlon)

	minlat = min(startlat,endlat)
	maxlat = max(startlat,endlat)

	#Work out the distance from each quake to the profile line, and write to a file 
	#Create the section coordinates. Should make some sort of auto-decision about the spacing
	sectionname = 'tmp_toposection.dat'

	#Path to gsbco_08 global bathy/topo grid
	topopath = '/Users/rmartinshort/Documents/Workshops/CIDER_2016/plumes/Bathy/gebco_08.nc'

	if not os.path.isfile(topopath):
		print 'Error! GEBCO 08 dataset does not exist in expected location %s' %topopath
		sys.exit(1)

	print '---------------------\nMaking section through topography\n---------------------'

	os.system('gmt project -C%g/%g -E%g/%g -G10 -Q > %s' %(minlon,minlat,maxlon,maxlat,sectionname))
	os.system('gmt grdtrack %s -G%s > gridsectiontopo.dat' %(sectionname,topopath))

	#Open the topo file and extract the longest distance. This will be used to scale the quake locations

	infile = open('gridsectiontopo.dat','r')
	lines = infile.readlines()
	infile.close()

	topoX = []
	topoY = []

	for line in lines:
	    vals = line.split()
	    topoX.append(float(vals[2]))
	    topoY.append(float(vals[3]))

	maxdist = topoX[-1]
	topoX = np.array(topoX)
	topoY = np.array(topoY)

	print '---------------------\nGetting quake distances\n---------------------'

	#Make a file containing quakelon, quakelat, dist, and dist along profile
	os.system('gmt mapproject quakestmp.dat -Lgridsectiontopo.dat/k > quake_dist.dat')

	#Reorder the columns and do another grdtrack to get distance along the profile
	os.system("awk '{print $5,$6,$1,$2,$3,$4}' quake_dist.dat > quaketmp.dat")
	os.system("rm quake_dist.dat")

	#Now, calculate distance along the profile from the start point
	os.system('gmt mapproject quaketmp.dat -G%g/%g/k > quake_points.dat' %(minlon,minlat))
	os.system('rm quaketmp.dat')

	#Now, open the newly created file and grid section file, and pull the distance data
	infile1 = open('quake_points.dat','r')
	lines1 = infile1.readlines()
	infile1.close()

	Xdistances_quakes = []
	Ydepths_quakes = []

	for line in lines1:
	    vals = line.split(' ')
	    try:

	        evlon = float(vals[0].split('\t')[-1])
	        evlat = float(vals[1])
	        evdep = float(vals[2])
	        evdist = float(vals[3].split('\t')[-2])
	        evdistalong = float(vals[3].split('\t')[-1])

	        #Only keep if the distance between the event and the profile line is less then 50km
	        if evdist <= 50:
	            Xdistances_quakes.append(evdistalong)
	            Ydepths_quakes.append(-evdep)

	        #for some reason, some depths don't exist, so use this try; except statement
	    except:
	        continue

	os.system('rm quake_points.dat')

	quakefig = plt.figure(facecolor='white')
	ax1 = quakefig.add_subplot(211)
	ax1.plot(topoX,topoY,'k')
	ax1.set_xlabel('Distance along profile [km]')
	ax1.set_ylabel('Surface height [m]')
	ax1.set_xlim([min(topoX),max(topoX)])
	ax1.set_ylim([min(topoY),max(topoY)])
	ax1.set_title('Topography')

	ax2 = quakefig.add_subplot(212)

	ax2.plot(Xdistances_quakes,Ydepths_quakes,'bo',label='Quake hypocenters: Within 50km of slice')
	ax2.plot([0,maxdist],[0,0],'r--',linewidth=3)
	ax2.set_xlim([min(topoX),max(topoX)])
	ax2.set_ylim([min(Ydepths_quakes)-50,50])
	ax2.set_xlabel('Distance along profile [km]')
	ax2.set_ylabel('Depth [km]')

	quakefig.show()


	#plt.tight_layout()
	#plt.grid()
	#plt.legend(loc='best')
	#ax2.set_title('Test earthquake profile plot')
	#plt.savefig('Earthquakes_%s.jpg' %name,dpi=200)
	#plt.show()







def main():

	t1 = UTCDateTime('2016-07-25')
	t2 = UTCDateTime('2016-07-01')
	mag1 = 5.01
	mag2 = 10.0

	cat = quaketools.get_cat(data_center='USGS',includeallorigins=True,starttime=t2,endtime=t1,minmagnitude=mag1,maxmagnitude=mag2,maxlongitude=180,minlongitude=-180,maxlatitude=90,minlatitude=-90)
	quakes, mts, events, qblasts = quaketools.cat2list(cat)

	#print quakes 
	#print mts
	#print events
	#print qblasts

	
	#write2file(quakes,mts)
	#quakeswithtime(quakes)
	cumulativemoment(quakes)




if __name__ == '__main__':

	main()