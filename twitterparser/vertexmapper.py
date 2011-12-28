import re
import string
import os
import sys
import itertools
import time
from datetime import date
import dateutil.parser

if len(sys.argv) < 4:
    sys.exit('Usage: %s data_tobemapped maximum_entries_toprocess retweetdata' % sys.argv[0])

if not os.path.exists(sys.argv[1]):
    sys.exit('ERROR: File %s was not found!' % sys.argv[1])

if not os.path.exists(sys.argv[3]):
    sys.exit('ERROR: File %s was not found!' % sys.argv[3])


# Test python
s = 'tokenize these words'
words = re.compile(r'\b\w+\b|\$')  # \b: matches the empty string at the beginning and end of a word
tokens = words.findall(s)
print tokens

infilename = sys.argv[1]
rtfilename = sys.argv[3]
rtnumfilename = rtfilename + ".num"
infile = open(infilename, "r")
rtfile = open(rtfilename, "r")
ofile = open('newmap.txt', "w")
rtnumfile = open(rtnumfilename, "w")

limit = long(sys.argv[2])
entries = 0
maxid = 0
twlist = []
twitters = {}	# empty dictionary
while infile:
        line = infile.readline()
	idname = line.split()
	if len(idname) == 0:
		break	# end of file
	id = long(idname[0])
	maxid = max(maxid, id)
	twlist.append((idname[1],id))
	entries = entries + 1
	if entries > limit:
		d = dict(twlist)
		twlist = [] # hope to delete memoty
		twitters.update(d)
		print (str(entries) + " processed, max id so far: " + str(maxid) )
		limit = limit + long(sys.argv[2])
d = dict(twlist)
twitters.update(d)

empty = 0
while rtfile:
	line = rtfile.readline()
	if not line:
		break
	fr_to_date = line.split()
	if len(fr_to_date) < 3:
		empty = empty +1	#empty retweet
		continue
	retweeter = fr_to_date[0]
	retweeted = fr_to_date[1]
	if retweeter in twitters:
		retweeter_id = twitters[retweeter]	
	else:
		maxid = maxid+1
		retweeter_id = maxid	# assign a new unused id
		twitters[retweeter] = retweeter_id # add that to the dictionary
		ofile.write(str(retweeter_id) + " " + retweeter + "\n")
	if retweeted in twitters:
		retweeted_id = twitters[retweeted]	
	else:
		maxid = maxid+1
		retweeted_id = maxid	# assign a new unused id
		twitters[retweeted] = retweeted_id # add that to the dictionary
		ofile.write(str(retweeted_id) + " " + retweeted + "\n")
	tofile = str(retweeter_id) + "\t" + str(retweeted_id) + "\t" + fr_to_date[2] + "\n"
	rtnumfile.write(tofile)

ofile.close()
rtfile.close()
rtnumfile.close()
infile.close()
print ("Empty retweets : %d\n", empty)

