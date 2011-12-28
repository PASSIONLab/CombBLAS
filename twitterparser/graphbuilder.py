import re
import string
import os
import sys
import itertools
import time
from datetime import date
import dateutil.parser

if len(sys.argv) < 2:
    sys.exit('Usage: %s retweetdata' % sys.argv[0])

if not os.path.exists(sys.argv[1]):
    sys.exit('ERROR: File %s was not found!' % sys.argv[1])

# Test python
s = 'tokenize these words'
words = re.compile(r'\b\w+\b|\$')  # \b: matches the empty string at the beginning and end of a word
tokens = words.findall(s)
print tokens


url = re.compile(r'http://(www.)?twitter.com/(.+)')
rtw = re.compile(r'@(.+)')	# retweet
name = re.compile(r'@([A-Za-z0-9-_]+)') # valid username

infilename = sys.argv[1]
oufilename = infilename + ".triples"
oufile = open(oufilename, "w")
infile = open(infilename, "r")

entries = 0
date = ""
print string.punctuation
while infile:
        line = infile.readline()
	if not line:	# end of file
		break
	if line.startswith('T'):	# new record
		entries = entries +1 ;
		info = line.split()
		date = info[1]	# unparsed string
		time = info[2]
		date = dateutil.parser.parse(date+"T"+time)
		line = infile.readline()	# read the next line
	
		if line.startswith('U'):
			info = line.split()
			user = url.match(info[1])
			if user:
				username = user.group(2)
			else:		
				print line
			line = infile.readline()	# read the next line

			if line.startswith('W'):
				allrt = re.findall(name,line)
				for rt in allrt:
					tofile = username + "\t" + rt + "\t" + date.isoformat() + "\n"
					oufile.write(tofile)
			else:
				sys.exit('U (user) line should be followed by W (words) line')
	
		else:
			sys.exit('T (time) line should be followed by U (user) line')
	
	
	



