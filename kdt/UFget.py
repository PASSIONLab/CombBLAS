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
from Mat import Mat
from Util import verbosity, INFO, p

#The folder you want to download files into
UFfolder = 'UFget/'

#Download and extract the named matrix from the sparse matrix library
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
		if verbosity >= INFO:
			p("Matrix %s already local, loading local copy."%(filename))
	#Return the file path of the .mtx file...
	return mtxFile

#Download the matrix at index i from the
#sparse matrix database and read it into a Matrix object
#	@staticmethod
def UFget(filename):
	"""
	downloads a file of the given base name from the University of Florida 
	Sparse Matrix Library (http://www.cise.ufl.edu/research/sparse/MM/), 
	extracts the file as needed, and loads the results into a DiGraph 
	instance.
	
	Note:  Matrix Market format numbers vertices from 1 to N.  Python and
	KDT number vertices from 0 to N-1.  UFget makes this conversion as it
	loads the file.

	Original Python version written by Kevin Oelze, July 2008
	"""
	G = dg.DiGraph(edges=Mat.load(UFdownload(filename)))
	return G
