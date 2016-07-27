#!/usr/bin/env python 

#Modfied from version in Programming Python 3rd Edition textbook 

#############################################
# a Quit button that verifies exit requests;
# to reuse, attach an instance to other GUIs
#############################################

from Tkinter import *                          # get widget classes
from tkMessageBox import askokcancel           # get canned std dialog

class Quitter(Frame):                          # subclass our GUI
    def __init__(self, parent=None):             # constructor method
        Frame.__init__(self, parent)
        #parent.title('Quit SeismoFetch program')
        self.pack( )
        widget = Button(self, text='Quit', command=self.quit)
        widget.pack()
    def quit(self):
        ans = askokcancel('Verify exit', "Really quit?")
        if ans: Frame.quit(self)

if __name__ == '__main__':  Quitter().mainloop()