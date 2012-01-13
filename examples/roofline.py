import sys
import kdt
import kdt.pyCombBLAS as pcb
import time

veclen = [10, 50, 100, 150, 200, 500, 1000, 1300, 1500, 1700]#, 2000, 2500, 3000, 5000, 10000, 15000]	
repeats = [10, 25, 50, 75, 100, 200, 300, 400]#, 1000, 10000, 100000]

def twitterMulMod(e, f):
	if e.follower and e.count > 0 and e.latest > 946684800:
		return e
	else:
		return e

def twitterAdd(f1, f2):
	if f2 == -1:
		return f1
	return f2

def runExperiment(func, e):
	best_ops_per_s = 0
	best_len = -1
	best_repeats = -1
	
	for vl in veclen:
		vec = kdt.Vec(vl, element=e, sparse=False)
		for r in repeats:
			begin = time.time()
			for i in range(r):
				vec.applyInd(func)
			t = time.time() - begin
			ops = vl*r / t
			kdt.p("len %5d, repeat %3d times: %f (took %f seconds)"%(vl, r, ops, t))
			if best_ops_per_s < ops:
				best_ops_per_s = ops
				best_len = vl
				best_repeats = r
	
	kdt.p("\nbest:")
	kdt.p("len %5d, repeat %3d times: %f"%(best_len, best_repeats, best_ops_per_s))


kdt.p("running on Multiply:")
runExperiment(twitterMulMod, kdt.Obj2())
kdt.p("running on add:")
runExperiment(twitterAdd, 0)