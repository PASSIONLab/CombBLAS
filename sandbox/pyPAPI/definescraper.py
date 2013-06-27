#!/usr/bin/python
import re
import sys

inputHeader = sys.stdin
outputHeader = sys.stdout

const_names = []

multiline = False
line = inputHeader.readline()
while (len(line) > 0):
	m = re.match("#define ([^ ]*) .*", line);
	
	if (m is not None):
		const_name = m.group(1)
		const_names.append(const_name)
		
		multiline = line.rstrip().endswith("\\")
		outputHeader.write(line)
	elif multiline:
		multiline = line.rstrip().endswith("\\")
		outputHeader.write(line)
	
	line = inputHeader.readline()

outputHeader.write("\n\n")
outputHeader.write("#include <iostream>\n")
outputHeader.write("")
outputHeader.write("using namespace std;\n")
outputHeader.write("void main(void) {\n")
for const_name in const_names:
	outputHeader.write("cout << \"" + const_name + " = \" << " + const_name + " << endl;\n")

outputHeader.write("}\n")