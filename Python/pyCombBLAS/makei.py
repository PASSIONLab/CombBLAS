#!/usr/bin/python
import re
import sys

templFilename = "pyCombBLAS.i.templ"
outFilename = "pyCombBLAS.i"


def writeFile(filename, outfile):
	print("Including from file " + filename)
	input = open(filename, 'r')
	shouldPrint = False;
	
	line = input.readline()
	while (len(line) > 0):
		m = re.match(".*INTERFACE_INCLUDE_([A-Z]*).*", line);
		if (m == None):
			if (shouldPrint):
				outfile.write(line)
		else:
			s = m.group(1)
			if (s == "BEGIN"):
				shouldPrint = True
			if (s == "END"):
				shouldPrint = False
		
		line = input.readline()

if len(sys.argv) == 3:
	templFilename = sys.argv[1]
	outFilename = sys.argv[2]
else:
	print("SWIG interface file maker.\nTakes an interface file template that is complete except for missing class bodies and fills them in from the C++ header files. This means any changes to the header files don't have to be manually copied into the interface file.\n")
	print("Usage:")
	print("python makei.py templatefile interfacefile\n")
	sys.exit(1)

templ = open(templFilename, 'r')
out = open(outFilename, 'w')

line = templ.readline()
while (len(line) > 0):
	m = re.match(".*INCLUDE \"([^\"]*)\".*", line);
	if (m == None):
		out.write(line)
	else:
		writeFile(m.group(1), out)
	
	line = templ.readline()


