import sys

s = sys.stdin.readline()
total = 0.0
doing = ""
printBest = False

best_ops = 0.0
best_total = 0.0
best_doing = ""
while len(s) != 0:
	if s[0] == 'm' or s[0] == 'a':
		printBest=True
	elif s[0] == 'b':
		doing = s
		total = 0.0
	elif s[0] == 't':
		sp = doing.split()
		rpt = 1#float(sp[10])
		op = rpt*float(sp[1])
		ops = op/total
		if ops > best_ops:
			best_ops = ops
			best_total = total
			best_doing = doing
		total = 0.0
	else:
		time = float(s)
		total += time

	s = sys.stdin.readline()
	if len(s) == 0:
		printBest=True

	if printBest:
		print "%f\t%f\t%s"%(best_ops,best_total,best_doing)
		printBest=False
