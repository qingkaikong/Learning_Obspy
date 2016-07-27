#!/usr/bin/env python 

#RMS 2016
#Main GUI class for QuakeWatch program

print 'Importing modules....'

from Tkinter import *
from point_browser import *
import matplotlib 
import numpy as np
import time
import datetime
from datetime import datetime as dt
import os
import cat_analysis as quaketools
import quake_stats as quakestats
from obspy import UTCDateTime
import tkFileDialog

#Import quit dialog box options
from quitter import Quitter

#Allow matplotlib to be used within a tkinter canvas
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, cm 

#Control matplotlib font size so that the map labels look OK
matplotlib.rcParams.update({'font.size': 8})

#This class handles the mouse-click options
Browse = PointBrowser()

print 'Done imports'

###########################################
#GUI base class for QuakeWatch 
###########################################

class QWGUI(Frame):
	'''Base class controlling QuakeWatch GUI aspects'''

	def __init__(self, parent, width=1100, height=400, **options):

		Frame.__init__(self,parent, **options)

		Grid.rowconfigure(self, 0, weight=1)
		Grid.columnconfigure(self, 0, weight=1)

		self.grid(sticky=E+W+N+S)
		self.userentries = {}

		top=self.winfo_toplevel()

		#Configure the width of rows and columns when tthe gui gets stretched. The second number provides the relative
		#weight of the column or row given by the first number

		norows = 14
		nocols = 13

		for i in range(norows):
			Grid.rowconfigure(self,i,weight=1)
			Grid.rowconfigure(parent,i,weight=1)
		for i in range(nocols):
			Grid.columnconfigure(self,i,weight=1)
			Grid.columnconfigure(parent,i,weight=1)


		#Various default settings
		self.momenttensors = True #display moment tensors where possible
		self.quakesplotted = None
		self.MTs = None

		t2 = str(datetime.datetime.today()).split(' ') #Current time
		t2 = t2[0]+'T'+t2[1][:-3]
		self.now = UTCDateTime(t2)

		self.starttime = 604800 #1 week quakes

		#--------------------------
		#Default map boundaries
		#--------------------------
		self.minlon = -179.9
		self.maxlon = 179.9
		self.minlat = -89
		self.maxlat = 89
		#--------------------------

		self.minmag = 4.5
		self.maxmag = 10.0
		self.mtradius = 10
		self.datacenter = 'USGS' #default datacenter to retrieve quake data from

        #create subplot where the map will go
		self.f = plt.figure(dpi=250,facecolor='white')

		#set the size of the figure for use with global map. Will need to choose this on the fly when
		#resizing the figure
		
		self.f.set_size_inches(5.0,2.2)
		self.a = self.f.add_subplot(111)

		#Call function to download some earhquakes and ... 

		#Initial meap setup and quakes
		print 'Drawing map....'
		self.map = Basemap(ax=self.a,lat_0=38,lon_0=-122.0,resolution ='l',llcrnrlon=-179.9,llcrnrlat=-89.9,urcrnrlon=179.9,urcrnrlat=89.9)
		self.map.fillcontinents()
		self.map.drawcoastlines(linewidth=0.2)

		#plot the plate boundaries
		print 'Drawing plate boundaries.....'
		self.faults = quaketools.read_faults('usgs_plates.txt.gmtdat')
		for i in self.faults:
		    faults_lons = self.faults[i][0]
		    faults_lats = self.faults[i][1]
		    x,y = self.map(faults_lons, faults_lats)
		    self.map.plot(x,y,'b-',linewidth=0.5)

		self.map.drawparallels(np.arange(-90,90,30),labels=[1,1,0,0],linewidth=0.5,fontsize=4)
		self.map.drawmeridians(np.arange(-180,180,30),labels=[0,0,0,1],linewidth=0.5,fontsize=4)

		self.canvas = FigureCanvasTkAgg(self.f, self)

		#plot quakes for the last week on the globe map with default parameters
		self.worldquakes()

		Browse.addobjs(self.canvas,self.map)

		self.f.canvas.mpl_connect('button_press_event',Browse.onpick)
		self.f.canvas.mpl_connect('motion_notify_event', Browse.motion)
		self.f.canvas.mpl_connect('button_release_event',Browse.releasepick)

		self.canvas.get_tk_widget().grid(row=1,sticky=W+S+N+E,columnspan=14,rowspan=10)
		self.canvas.show()

		self.SetElements()

		parent.title("QuakeWatch Mapper")

		self.Createmenubar(parent)


	def SetElements(self):

		'''Sets up the the GUI elements'''

		Label(self,text='Earthquakes',bg='azure',height=2,pady=2,font='Helvetica 22 bold').grid(row=0,column=0,columnspan=14,sticky=W+E+S+N)

		#Zoom box bounds

		Label(self,text='Bounds of zoom box [lon/lat]',bg='azure',height=4,pady=2,padx=1,font='Helvetica 14 bold').grid(row=11,column=1,columnspan=4,sticky=W+E+S+N)
		Ncord = Entry(self)
		Ncord.grid(row=12,column=2,columnspan=1,sticky=E)
		Label(self,text='Northeastern corner').grid(row=12,column=1,columnspan=1,sticky=W)
		self.userentries['Northeast_box'] = Ncord

		Scord = Entry(self)
		Scord.grid(row=13,column=2,columnspan=1,sticky=E)
		Label(self,text='Southwestern corner').grid(row=13,column=1,columnspan=1,sticky=W)
		self.userentries['Southwest_box'] = Scord

		Button(self, text='Zoom',pady=1,padx=1,command=self.zoomin).grid(row=14,column=1,sticky=W+S+E+N,columnspan=1)
		Button(self, text='Reset',pady=1,padx=1,command=self.resetzoom).grid(row=14,column=2,sticky=W+S+E+N,columnspan=1)
		Button(self, text='Draw',pady=1,padx=1,command=self.drawbox).grid(row=14,column=3,sticky=W+S+E+N,columnspan=1)

		#Drawing profiles

		Label(self,text='Bounds of profile [lon/lat]',bg='azure',height=4,pady=2,padx=1,font='Helvetica 14 bold').grid(row=11,column=6,columnspan=4,sticky=W+E+S+N)

		Stcord = Entry(self)
		Stcord.grid(row=12,column=7,columnspan=1,sticky=E)
		Label(self,text='Start:').grid(row=12,column=6,columnspan=3,sticky=W)
		self.userentries['profile_start'] = Stcord

		Edcord = Entry(self)
		Edcord.grid(row=13,column=7,columnspan=1,sticky=E)
		Label(self,text='End:').grid(row=13,column=6,columnspan=3,sticky=W)
		self.userentries['profile_end'] = Edcord

		Button(self, text='Plot profile',pady=1,padx=1,command=self.plotprofile).grid(row=14,column=6,sticky=W+S+E+N,columnspan=4)

		#Event labels

		Label(self,text='Event options',bd=5,bg='azure',height=4,pady=2,padx=1,font='Helvetica 14 bold').grid(row=11,column=11,columnspan=4,sticky=W+E+S+N)

		Evtmags = Entry(self)
		Evtmags.grid(row=12,column=12,columnspan=2,sticky=E)
		Label(self,text='Event magnitude range [min-max]').grid(row=12,column=11,columnspan=1,sticky=W)
		self.userentries['magrange'] = Evtmags

		Evttime = Entry(self)
		Evttime.grid(row=13,column=12,columnspan=2,sticky=E)
		Label(self,text='Event start date [yyyy-mm-dd]').grid(row=13,column=11,columnspan=1,sticky=W)
		self.userentries['evttime'] = Evttime

		Button(self, text='Set',pady=1,padx=1,command=self.setquakesmanual).grid(row=14,column=11,sticky=W+S+E+N,columnspan=1)
		Button(self, text='Reset',pady=1,padx=1,command=self.removemapobjs).grid(row=14,column=12,sticky=W+S+E+N,columnspan=1)

        #Set up the label showing when the datasets are refreshed: the current time goes into that label
		self.timer = StringVar()
		Label(self,textvariable=self.timer,bg='azure',height=1,pady=1,padx=1,font='Helvetica 10 bold').grid(row=0,column=0,columnspan=4,sticky=W+E+S+N)
		self.timer.set('Updated %s' %str(time.asctime()))

		self.quakeinfo = StringVar()
		Label(self,textvariable=self.quakeinfo,bg='azure',height=1,padx=1,pady=1,font='Helvetica 10 bold').grid(row=0,column=8,columnspan=4,sticky=W+E+S+N)
		self.quakeinfo.set('Displaying: M%s to M%s' %(self.minmag,self.maxmag))

		Quitter(self).grid(row=0,column=13,sticky=E)

	def SetStartMap(self):

		'''Make global pretty map. Takes a long time so only make on startup'''

		self.map = Basemap(ax=self.a,lat_0=38,lon_0=-122.0,resolution ='l',llcrnrlon=-179.9,llcrnrlat=-89,urcrnrlon=179.9,urcrnrlat=89)
		#self.map.shadedrelief()
		#self.map.arcgisimage(service='NatGeo_World_Map',verbose=False,xpixels=10000)

		#placeholder - makes things run much faster for debugging
		self.map.fillcontinents()

		#plot the plate boundaries
		for i in self.faults:
		    faults_lons = self.faults[i][0]
		    faults_lats = self.faults[i][1]
		    x,y = self.map(faults_lons, faults_lats)
		    self.map.plot(x,y,'b-',linewidth=0.5)
		
		self.map.drawparallels(np.arange(-90,90,30),labels=[1,1,0,0],linewidth=0.5,fontsize=4)
		self.map.drawmeridians(np.arange(-180,180,30),labels=[0,0,0,1],linewidth=0.5,fontsize=4)
		self.canvas.draw()

		#--------------------------
		#Default map boundaries
		#--------------------------
		self.minlon = -179.9
		self.maxlon = 179.9
		self.minlat = -89
		self.maxlat = 89
		#--------------------------


	def SetZoomMap(self,lon1,lon2,lat1,lat2):

		'''Function that handles the zoom in map creation'''


		self.a.clear()

		#Ensure that no extra elememts are plotted
		self.MTs = None
		self.quakesplotted = None

		#scale the longitude and latiude grid increments
		latinc = (max(lat1,lat2)-min(lat1,lat2))/5
		loninc = abs((lon2-lon1)/4)
		lon0 = int(abs(lon2)-abs(lon1))/2
		lat0 = int((lat2-lat1)/2)

		#choose the resolution: high resolition maps take AGES to load, however

		if 2.0 < abs(lat2-lat1) < 6.0:
			res = 'i'
		elif 0.1 < abs(lat2-lat1) < 1.0:
			res = 'h'
		elif 0 < abs(lat2-lat1) < 0.1:
			res = 'f'
		else:
			res = 'l'

		#choose a resolution based on the size of the provided box
		print 'Lower left lon: %g' %lon1
		print 'Upper right lon: %g' %lon2
		print 'lower left lat: %g' %lat1
		print 'upper right lat: %g' %lat2
		print 'resolution = %s' %res

		#--------------------------
		#Set map boundaries
		#--------------------------
		self.minlat = lat1
		self.maxlat = lat2
		self.minlon = lon1
		self.maxlon = lon2


		self.map = Basemap(ax=self.a,lat_0=lat0,lon_0=lon0,resolution=res,llcrnrlon=lon1,llcrnrlat=lat1,urcrnrlon=lon2,urcrnrlat=lat2)
		self.map.drawparallels(np.arange(lat1,lat2,latinc),labels=[1,1,0,0],linewidth=0.5,fontsize=4)
		self.map.drawmeridians(np.arange(lon1,lon2,loninc),labels=[0,0,0,1],linewidth=0.5,fontsize=4)

		self.map.fillcontinents()
		self.map.drawcountries(linewidth=0.2)
		self.map.drawcoastlines(linewidth=0.2)
		self.map.drawstates(linewidth=0.1)

		#plot the plate boundaries
		for i in self.faults:
		    faults_lons = self.faults[i][0]
		    faults_lats = self.faults[i][1]
		    x,y = self.map(faults_lons, faults_lats)
		    self.map.plot(x,y,'b-',linewidth=0.5)

		#Determine the new sizes of the map objects, so that they don't crowd the map
		mt_width,mt_rad,min_dot,max_dot,min_quake,max_quake = quaketools.determinemtsize(self.minlon,self.maxlon,self.minlat,self.maxlat)
		print min_quake,max_quake

		print 'Mt width and radius are as follows:'
		print mt_width
		print mt_rad

		#redraw the earthquakes, if they exist (they should always exist)
		#keeps the catalog object thats already been saved in memory and just replots the events (unless we want moment tensors)

		if self.momenttensors == True:

			self.mtlines, self.MTs, self.quakedots, xs, ys, urls = quaketools.plot_mt(self.map,self.a,self.f,self.quakes,self.mts,self.events,llat=lat1,ulat=lat2,llon=lon1,ulon=lon2,dist_bt=200,radius=mt_rad,mt_width=mt_width)
			Browse.updatedata(xs,ys,urls,mt_rad)

		else: 

			quaketools.plot_events(self.map,self.a,self.quakes,llat=lat1,ulat=lat2,llon=lon1,ulon=lon2,dist_bt=100,min_size=min_quake,max_size=max_quake)


		#self.map = Basemap(ax=self.a,projection='merc',llcrnrlat=lat1,urcrnrlat=lat2,llcrnrlon=lon1,urcrnrlon=lon2,lat_ts=true_scale_lat,resolution='l')
		#self.map.arcgisimage(service='NatGeo_World_Map',verbose=False,xpixels=10000)

		#self.map.drawlsmask(land_color="#ddaa66", ocean_color="#7777ff",resolution='i')
		self.canvas.draw()


	def zoomin(self):

		'''Choose how to zoom - using a draw-on box (default) or a user-defined box. The GetBoxCoors() function
		deals with whether to choose from a user-drawn box or a coordinate-entered box'''

		try:
			NElat,NElon,SWlat,SWlon = self.GetBoxCoors()
			self.SetZoomMap(SWlon,NElon,SWlat,NElat)
			Browse.updateboxcoords()
		except:
			print 'User coordinates not entered correctly'


	def drawbox(self):

		'''Draw a box on the map according to the user's choice of coordinates'''

		boxcoordsNE = self.userentries['Northeast_box'].get()
		boxcoordsSW = self.userentries['Southwest_box'].get()

		try:
			NElon = float(boxcoordsNE.split('/')[0])
			NElat = float(boxcoordsNE.split('/')[1])
			SWlon = float(boxcoordsSW.split('/')[0])
			SWlat = float(boxcoordsSW.split('/')[1])
		except:
			print 'User coordinates not entered correctly'

		boxlats = [SWlat,SWlat,NElat,NElat,SWlat]
		boxlons = [SWlon,NElon,NElon,SWlon,SWlon]

		xevent,yevent = self.map(boxlons,boxlats)
		self.map.plot(xevent,yevent,'r-',linewidth=1,alpha=0.9)
		self.map.plot(xevent,yevent,'k.')
		self.canvas.draw()


	def resetzoom(self,resettensors=False):

		'''Reset the zoom to the global map'''

		if resettensors == True:
			self.momenttensors = True

		self.a.clear()
		self.MTs = None
		self.quakesplotted = None

		#reset defaults
		self.minmag = 4.5
		self.maxmag = 10.0

		#reset the display defaults
		self.quakeinfo.set('Displaying: M%s to M%s' %(self.minmag,self.maxmag))	
		Browse.updateboxcoords()

		#Return to the default quake map
		self.SetStartMap()
		self.WDquakes()

	def plotprofile(self):

		'''Draw a profile on the map according to the user's choice of coordinates. Will be extended to draw a depth
		cross section along the profile line, showing the depth distribition of the seismicity on the map'''

		linecoordsstart = self.userentries['profile_start'].get()
		linecoordsend = self.userentries['profile_end'].get()

		try:
			Slon = linecoordsstart.split('/')[0]
			Slat = linecoordsstart.split('/')[1]
			Elon = linecoordsend.split('/')[0]
			Elat = linecoordsend.split('/')[1]
		except:
			print 'User coordinates not entered correctly'

		lats = [Slat,Elat]
		lons = [Slon,Elon]

		xevent,yevent = self.map(lons,lats)
		self.map.plot(xevent,yevent,'r-',linewidth=1,alpha=0.9)
		self.map.plot(xevent,yevent,'k.')
		self.canvas.draw()

		print '-----------------------------------------------'
		print 'Building depth slice'
		print '-----------------------------------------------'

		quakestats.depthslicequakes(self.quakes,self.mts,float(lons[0]),float(lats[0]),float(lons[1]),float(lats[1]))

	def setquakesmanual(self,t1=None):

		'''Fetch an earthquake catalog corresponding to the user's choice'''

		#Reset the map
		self.removemapobjs()

		if not t1:

			starttime = self.userentries['evttime'].get()

			t1 = str(starttime)+'T00:00:00.000'


		t2 = self.now

		try:
			t1 = UTCDateTime(t1)
		except:
			print 'Alert: Times not entered correctly!'
			print 'Default time range : 1970-01-01 to today'
			t1 = UTCDateTime("1970-01-01T00:00:00.000")


		mags = self.userentries['magrange'].get()

		try:
			mag1 = mags.split('-')[0].strip()
			mag2 = mags.split('-')[1].strip()
		except:
			print 'Alert: Magnitudes not entered correctly!'

			if self.minmag:

				mag1 = self.minmag
				mag2 = self.maxmag
			else:

				print 'Default magnitude range: 6-10'
				mag1 = 6
				mag2 = 10

		#get the box coordinates, if they exist
		try:

			NElat,NElon,SWlat,SWlon = self.GetBoxCoors()

			#set the dafault map coordinates to be those corresponding to the inputs
			self.minlon = SWlon
			self.maxlon = NElon
			self.minlat = SWlat
			self.maxlat = NElat

		except:

			print 'No user defined coordinates found - using coordinates stored in memory!'

		#get the earthquake catalog just for the region of interest (may be a zoom box)
		#Note that the earthquake statistics package uses the catalog output from this call to get_cat, so ensure to
		#update the quake catalog before looking at the stats

		mt_width,mt_rad,min_dot,max_dot,min_quake,max_quake = quaketools.determinemtsize(self.minlon,self.maxlon,self.minlat,self.maxlat)

		print 'Mt width and radius are as follows:'
		print mt_width
		print mt_rad

		self.catalog = quaketools.get_cat(data_center=self.datacenter,includeallorigins=True,starttime=t1,endtime=t2,minmagnitude=mag1,maxmagnitude=mag2,maxlongitude=self.maxlon,minlongitude=self.minlon,maxlatitude=self.maxlat,minlatitude=self.minlat)
		self.quakes, self.mts, self.events, self.qblasts = quaketools.cat2list(self.catalog)

		if self.momenttensors == True:

			#plot the moment tensors and redraw
			self.mtlines, self.MTs, self.quakedots, xs, ys, urls = quaketools.plot_mt(self.map,self.a,self.f,self.quakes,self.mts,self.events,llat=self.minlat,ulat=self.maxlat,llon=self.minlon,ulon=self.maxlon,dist_bt=200,radius=mt_rad,mt_width=mt_width)
			Browse.updatedata(xs,ys,urls,mt_rad)

		else:
			#only plotting events, so continue
			self.quakesplotted = quaketools.plot_events(self.map,self.a,self.quakes,llat=self.minlat,ulat=self.maxlat,llon=self.minlon,ulon=self.maxlon,dist_bt=100,min_size=min_quake,max_size=max_quake)
		

		#update display information
		self.minmag = mag1
		self.maxmag = mag2
		self.quakeinfo.set('Displaying: M%s to M%s' %(self.minmag,self.maxmag))	

		self.canvas.draw()

	def removemapobjs(self):

		'''Remove all objects painted on top of the map, but does not redraw the map itself'''

		Browse.updatedata()
		Browse.updateboxcoords()

		if self.quakesplotted != None:

			self.quakesplotted.remove()
			self.canvas.draw()
			self.quakesplotted = None

		if self.MTs != None:

			for e1 in self.MTs:
				e1.remove()

			for e2 in self.mtlines:
				if e2 is not None:
					e2[0].remove()

			self.quakedots.remove()

			#self.mtlines.remove()
			#self.mtdots.remove()
			#self.MTs.remove()
			self.canvas.draw()
			self.MTs = None

		else:

			print 'Default reset function'


	def refreshloop(self):

		'''Refresh the map data accordingly'''

		print 'Refreshing quake dataset'
		self.timer.set(str(time.asctime()))

		#GetLatestseismicity(14) #Get seismicity from the last 14 days, down to magnitude 0.1
		#self.Autoupdateplot()
		#self.Autoupdate()
		tk.after(210000,self.refreshloop) #Refresh the dataset every 10 mins


	def Createmenubar(self,parent): 

		'''Create the drop down menu: allows user to add data layers to the Alaska'''

		menubar = Menu(self)
		parent.config(menu=menubar)
		filemenu = Menu(menubar,tearoff=0,font="Helvetica 16 bold") #insert a drop-down menu

		submenu1 = Menu(filemenu)
		submenu1.add_command(label='M2.5+ 1 Week',command=self.M25_1wk)
		submenu1.add_command(label='M4.5+ 1 Week',command=self.M45_1wk)
		submenu1.add_command(label='M2.5+ 30 days',command=self.M25_30d)
		submenu1.add_command(label='M4.5+ 30 days',command=self.M45_30d)
		submenu1.add_command(label='M6.0+ 365 days',command=self.M60_365d)
		filemenu.add_cascade(label='Event options',menu=submenu1,underline=0)

		filemenu.add_separator()

		submenu2 = Menu(filemenu)
		submenu2.add_command(label='World',command=self.WDquakes)
		submenu2.add_command(label='USA',command=self.USAquakes)
		submenu2.add_command(label='California',command=self.CAquakes)
		submenu2.add_command(label='Oklahoma',command=self.OKquakes)
		submenu2.add_command(label='Alaska',command=self.AKquakes)
		filemenu.add_cascade(label='Region options',menu=submenu2,underline=0) #add the drop down menu to the menu bar

		filemenu.add_separator() 

		submenu3 = Menu(filemenu)
		submenu3.add_command(label='Save current frame',command=self.SaveasPDF)
		submenu3.add_command(label='Pretty map (may take a long time to load!)',command=lambda: self.PrettyMap(self.minlat,self.maxlat,self.minlon,self.maxlon))
		submenu3.add_command(label='Display moment tensors',command=self.setMTs)
		submenu3.add_command(label='Display events only',command=self.setevents)
		filemenu.add_cascade(label='Other options',menu=submenu3) #add the drop down menu to the menu bar 

		menubar.add_cascade(label="Options",menu=filemenu)

		subm1 = Menu(menubar,tearoff=0)
		subm1.add_command(label='Gutenburg-Richter plot',command=self.placeholder)
		subm1.add_command(label='Cumulative moment release',command=self.cumulate_moment)
		subm1.add_command(label='Binned quake activity',command=self.quaketimeplot)
		menubar.add_cascade(label="Statistics",menu=subm1)


#############################################################################
# Some standard options for plotting quakes for various lengths of time
#############################################################################


	def M25_1wk(self):

		self.minmag = 2.5
		self.maxmag = 10

		#Set the map to display all events in the last week

		t1=self.now-604800

		self.setquakesmanual(t1)

		self.starttime = t1

	def M45_1wk(self):

		self.minmag = 4.5
		self.maxmag = 10

		#Set the map to display all events in the last week

		t1=self.now-604800

		self.setquakesmanual(t1)

		self.starttime = t1


	def M25_30d(self):

		self.minmag = 2.5
		self.maxmag = 10

		#Set the map to display all events of M2.5+ in the last 30 days

		t1=self.now-2592000

		self.setquakesmanual(t1)

		self.starttime = t1

	def M45_30d(self):

		self.minmag = 4.5
		self.maxmag = 10

		#Set the map to display all events of M4.5+ in the last 30 days

		t1=self.now-2592000

		self.setquakesmanual(t1)

		self.starttime = t1 

	def M60_365d(self):

		self.minmag = 6.0
		self.maxmag = 10

		#Set the map to display all events in the last year 

		t1=self.now-31536000

		self.setquakesmanual(t1)

		self.starttime = t1

#############################################################################
# Some standard options for plotting quakes for various lengths of time
#############################################################################

	def quaketimeplot(self):

		'''Link to quake_atats historgram of quake activity binned by time'''

		if ((self.now-self.starttime)) < 3e6:
			fq = 'day'
		else:
			fq = 'week'

		quakestats.quakeswithtime(self.quakes,freq=fq)

	def cumulate_moment(self):

		'''Link to quake_stats plot of cumulative moment release'''

		quakestats.cumulativemoment(self.quakes)


	def placeholder(self):

		'''Will link to some other command'''

		print 'not yet coded!'

	def worldquakes(self):

		'''Get world catalog of quakes and plot. If we already on a global map, then we just plot the elemments'''

		print 'Getting world quakes!!'

		t2 = self.now
		t1 = t2-self.starttime

		#Determine the new sizes of the map objects, so that they don't crowd the map
		mt_width,mt_rad,min_dot,max_dot,min_quake,max_quake = quaketools.determinemtsize(self.minlon,self.maxlon,self.minlat,self.maxlat)
		print min_quake,max_quake

		self.catalog = quaketools.get_cat(data_center=self.datacenter,includeallorigins=True,starttime=t1,endtime=t2,minmagnitude=self.minmag,maxmagnitude=self.maxmag)
		self.quakes, self.mts, self.events, self.qblasts = quaketools.cat2list(self.catalog)

		if self.momenttensors == True:

			self.quakesplotted = quaketools.plot_events(self.map,self.a,self.quakes)
			self.mtlines, self.MTs, self.quakedots, xs, ys, urls = quaketools.plot_mt(self.map,self.a,self.f,self.quakes,self.mts,self.events,mt_width=3,radius=self.mtradius,angle_step=40)
			Browse.updatedata(xs,ys,urls,self.mtradius)
			#print xs,ys,urls

		else:
			#only plotting events, so continue
			self.quakesplotted = quaketools.plot_events(self.map,self.a,self.quakes,min_size=min_quake,max_side=max_quake)
		
		self.canvas.draw()

	def Autoupdate(self):

		'''For use with the auto-updater - get only the quakes within the map region and update'''

		print 'Autoupdating catalog!'

		self.setquakesmanual(t1=self.starttime)


	def USAquakes(self):

		'''Zoom in on USA region'''

		self.SetZoomMap(-132,-70,27,52)

	def CAquakes(self):

		'''Zoom in on CA region'''

		self.SetZoomMap(-132,-115,32,42)

	def AKquakes(self):

		print 'display'

	def OKquakes(self):

		'''Zoom in on OK/Texas region'''

		self.SetZoomMap(-105,-90.5,29,38)

	def WDquakes(self):

		self.SetZoomMap(-179.9,179.9,-89.9,89.9)

	def PrettyMap(self,lat1,lat2,lon1,lon2):

		'''Display topography and street names on the current map - could be modified for various map types'''

		#scale the longitude and latiude grid increments
		latinc = (max(lat1,lat2)-min(lat1,lat2))/5
		loninc = abs((lon2-lon1)/4)
		lon0 = int(abs(lon2)-abs(lon1))/2
		lat0 = int((lat2-lat1)/2)

		if 2.0 < abs(lat2-lat1) < 6.0:
			res = 'i'
		elif 0.1 < abs(lat2-lat1) < 1.0:
			res = 'h'
		elif 0 < abs(lat2-lat1) < 0.1:
			res = 'f'
		else:
			res = 'l'

		self.a.clear()

		#Determine the new sizes of the map objects, so that they don't crowd the map
		mt_width,mt_rad,min_dot,max_dot,min_quake,max_quake = quaketools.determinemtsize(self.minlon,self.maxlon,self.minlat,self.maxlat)

		#Map setup and quakes
		self.map = Basemap(ax=self.a,lat_0=lat0,lon_0=lon0,resolution ='l',llcrnrlon=lon1,llcrnrlat=lat1,urcrnrlon=lon2,urcrnrlat=lat2)
		self.map.arcgisimage(service='NatGeo_World_Map',verbose=False,xpixels=600)

		#plot some extra information

		self.map.drawparallels(np.arange(lat1,lat2,latinc),labels=[1,1,0,0],linewidth=0.5,fontsize=4)
		self.map.drawmeridians(np.arange(lon1,lon2,loninc),labels=[0,0,0,1],linewidth=0.5,fontsize=4)
		self.map.drawcountries()
		self.map.drawcoastlines(linewidth=0.1)

		#replot the quakes

		if self.momenttensors == True:

			#plot the moment tensors and redraw
			#self.quakesplotted = quaketools.plot_events(self.map,self.a,self.quakes)
			self.mtlines, self.MTs, self.quakedots, xs, ys, urls = quaketools.plot_mt(self.map,self.a,self.f,self.quakes,self.mts,self.events,mt_width=mt_width,radius=mt_rad,angle_step=40,llat=lat1,ulat=lat2,llon=lon1,ulon=lon2)
			Browse.updatedata(xs,ys,urls,mt_rad)

		else:
			#only plotting events, so continue
			self.quakesplotted = quaketools.plot_events(self.map,self.a,self.quakes,llat=self.minlat,ulat=self.maxlat,llon=self.minlon,ulon=self.maxlon,dist_bt=100,min_size=min_quake,max_size=max_quake)

		self.canvas.draw()


	def setMTs(self):

		self.momenttensors = True
		self.removemapobjs()
		self.setquakesmanual(t1=self.starttime)

	def setevents(self):

		self.momenttensors = None
		self.removemapobjs()
		self.setquakesmanual(t1=self.starttime)

	def SaveasPDF(self):
		'''Saves the current frame as a .pdf file'''

		#self.f.savefig('test.pdf',format='pdf')

		filelocation = tkFileDialog.asksaveasfilename(defaultextension='.pdf')
		self.f.savefig(filelocation,format='pdf')
		print 'Saved current figure'

	def GetBoxCoors(self):

		'''Returns the coordinates of user defined zoom box'''

		#first try to get the coordinates of a zoom region:
		blats,blons = Browse.returnboxcoords()

		if blats:
			SWlon = blons[0]
			NElon = blons[2]
			SWlat = blats[0]
			NElat = blats[1]

			if NElat < SWlat:
				tmp = NElat.copy()
				NElat = SWlat
				SWlat = tmp
			if NElon < SWlon:
				tmp = NElon.copy()
				NElon = SWlon
				SWlon = tmp

			return NElat,NElon,SWlat,SWlon
		else:
			boxcoordsNE = self.userentries['Northeast_box'].get()
			boxcoordsSW = self.userentries['Southwest_box'].get()

			try:
				NElon = float(boxcoordsNE.split('/')[0])
				NElat = float(boxcoordsNE.split('/')[1])
				SWlon = float(boxcoordsSW.split('/')[0])
				SWlat = float(boxcoordsSW.split('/')[1])
				return NElat,NElon,SWlat,SWlon
			except:
				return None



if __name__ == '__main__':

	'''The GUI loop'''
	tk = Tk()
	viewer = QWGUI(tk)
	viewer.refreshloop()
	tk.mainloop()




