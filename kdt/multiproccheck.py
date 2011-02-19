"""
Functions to ease checking correctness of multi-proc code as compared to single proc code.

For example:
v = pcb.pyDenseParVec.range(10, 0)
checkvect(v, 'v')

will either save v or load+compare, depending on the number of processors we're running on.
So first run the code on one processor, then again right away on multiple processors. Any differences will be flagged.
Use a checkvect or checkmat to test every suspicious vector or matrix, or just sprinkle them in the whole method to identify the weak spot.

Note: checkmat hasn't been thoroughly tested.
"""

def checkmat(m, name):
	"""If run on one processor it will save m. If run on multiple processors it will load the one-proc m and compare it to the argument and complain if they don't match. """
	import pyCombBLAS as pcb
	
	if (pcb._nprocs() == 1):
		m.save("checkfile_%s"%(name))
	else:
		one = pcb.pySpParMat()
		one.load("checkfile_%s"%(name))
		test = pcb.EWiseApply(m, one, pcb.equal_to())
		if (test.Count(pcb.bind2nd(pcb.equal_to(), 1)) != test.getnee()):
			if (pcb.root()):
				print "%s failed."%(name)

def saveVect(v, name):
	import sys
	import pyCombBLAS as pcb
	
	if (pcb.root()):
		file = open(name, 'w')
	l = len(v)
	if (pcb.root()):
		file.write(str(l)+"\n")
	for i in range(l):
		val = v[i]
		if (pcb.root()):
			file.write(str(i) + "\t" + str(val) + "\n")
		
def loadDenseVect(name, length):
	import sys
	import csv
	import pyCombBLAS as pcb

	#needs to happen on all processors	
	inputFile = open(name, "rb")
	parser = csv.reader(inputFile, dialect="excel-tab")
	firstRec = True
	
	for fields in parser:
		if firstRec:
			firstRec = False
			n = int(fields[0])
			if (n != length):
				return pcb.pyDenseParVec(length, 0)
				
			ret = pcb.pyDenseParVec(n, 0)
		else:
			if len(fields) == 2:
				ret[int(fields[0])] = float(fields[1])
	return ret
	
			
def checkvect(v, name):
	"""If run on one processor it will save v. If run on multiple processors it will load the one-proc v and compare it to the argument and complain if they don't match. """
	import pyCombBLAS as pcb
	
	if (pcb._nprocs() == 1):
		saveVect(v, "checkfile_%s"%(name))
	else:
		one = loadDenseVect("checkfile_%s"%(name), len(v))
		if (len(one) != len(v)):
			print "%s failed. length_1 = %d, lengh_p = %d"%(name, len(one), len(v))
			return
		one.EWiseApply(v, pcb.equal_to())
		if (one.Count(pcb.bind2nd(pcb.equal_to(), 1)) != v.getnee()):
			if (pcb.root()):
				print "%s failed."%(name)
