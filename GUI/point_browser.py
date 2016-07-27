#!/usr/bin/env python
import webbrowser
import numpy as np


class PointBrowser:
    """
    Click on a point to select and highlight it -- the data that
    generated the point will be shown in the lower axes.  Use the 'n'
    and 'p' keys to browse through the next and previous points
    """
    def __init__(self,xs=None,ys=None):
        self.lastind = 0
        self.xs = None
        self.ys = None
        self.urls = None
        self.dragging = None
        self.box = None
        self.boxlats = None
        self.boxlons = None

    def addobjs(self,canvasobj,mapobj):

      self.mapobj = mapobj
      self.canvasobj = canvasobj

    def updateboxcoords(self):

      '''reset the box coordinates to none'''

      self.boxlats = None
      self.boxlons = None
      self.box = None

    def updatedata(self,xs=None,ys=None,urls=None,rad=None):
 
        self.xs = xs
        self.ys = ys
        self.url = urls
        if rad:
          self.mtsize = rad/2.0
        else:
          self.mtsize = None

    def motion(self,event):

      '''define what happens when the user moves the mouse over the canvas'''

      lon = event.xdata
      lat = event.ydata

      if self.dragging:

        print 'Dragging!'

        if self.box:
          self.box[0].remove()

        boxlats = [self.startlat,lat,lat,self.startlat,self.startlat]
        boxlons = [self.startlon,self.startlon,lon,lon,self.startlon]
        xevent,yevent = self.mapobj(boxlons,boxlats)

        self.box = self.mapobj.plot(xevent,yevent,'r-',linewidth=1,alpha=0.9)
        self.canvasobj.draw()

    def releasepick(self,event):

      '''define what happens when the user releases the cursor'''

      lon = event.xdata
      lat = event.ydata

      if self.dragging:

        self.boxlats = [self.startlat,lat,lat,self.startlat,self.startlat]
        self.boxlons = [self.startlon,self.startlon,lon,lon,self.startlon]

        self.dragging = None


    def returnboxcoords(self):

      '''return box coordinates to user'''

      if self.boxlats:

        return self.boxlats,self.boxlons


    def onpick(self, event):

      '''define what happens when the user presses the cursor'''

      # the click locations

      lon = event.xdata
      lat = event.ydata

      self.startlon = lon
      self.startlat = lat
      self.dragging = True

      try:

        if self.xs.any():

          #determine distances from the click point to each of the moment tensors
          distances = np.hypot(lon-self.xs, lat-self.ys)

          #determine index of the nearest moment tensor
          indmin = distances.argmin()

          if distances[indmin] < self.mtsize:

            self.lastind = indmin
            self.update()

      except:

        print 'Currently no xs vector'







    def update(self):
        if self.lastind is None: return

        dataind = self.lastind

        open_url = self.url[dataind]
        webbrowser.open(open_url, new=0, autoraise=True)