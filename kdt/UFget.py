'''
UFget.py - A hook into the University of Florida's Sparse Matrix Library
for use in Python.  For more detailed documentation, feel free to visit
the website at www.cs.hmc.edu/~koelze/

Written by Kevin Oelze, July 2008

'''


import urllib
import tarfile
import os
import pickle
import DiGraph as dg

#The folder you want to download files into
UFfolder = 'UFget/'

#Download and extract the ith matrix from the sparse matrix library
def UFdownload(filename):
    global UFfolder
    baseURL = 'http://www.cise.ufl.edu/research/sparse/MM/'
    postfix = '.tar.gz'
    mtxfix = '.mtx'
    tarDir = baseURL + filename
    dlLoc = UFfolder + filename[:filename.find('/')]
    mtxFile = UFfolder + filename + filename[filename.find('/'):] + mtxfix
    tarFile = UFfolder + filename + postfix
    #Download and untar the file and make the directories (if necessary)
    if(not os.path.exists(mtxFile)):
        try:
            os.makedirs(os.path.dirname(tarFile))
        except:
            pass
        urllib.urlretrieve(tarDir +  postfix, tarFile)
        extract = tarfile.open(tarFile)
        extract.extractall(dlLoc)
        #Now get rid of that pesky tar file..
        #os.remove(tarFile)
    else:
        print 'Matrix', filename, 'already exists, doing nothing.'
    #Return the file path of the .mtx file...
    return mtxFile

#Download the matrix at index i from the
#sparse matrix database and read it into a Matrix object
def UFget(filename):
    return dg.spm.load(UFdownload(filename))
