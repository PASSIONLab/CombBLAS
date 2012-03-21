import sys
import kdt
import kdt.pyCombBLAS as pcb
import time

mults_mulonly = range(8, 2000, 8)
mults_muladd = range(1, 40, 1)
adds = range(8, 200, 16) 
repeats = [100]

def twitterFilter(e):
	f = (e.follower == 0 and e.count >= 0 and e.latest < 946684800)
	return True

def makeMat(rows, cols, full):
	if full:
		return kdt.Mat.ones(cols, rows, element=kdt.Obj2())
	else:
		i = kdt.Vec.range(rows, sparse=False)
		j = kdt.Vec(rows, element=0, sparse=False)
		v = kdt.Vec(rows, element=kdt.Obj2(), sparse=False)
		
		return kdt.Mat(i, j, v, cols, rows)

def runExperiment(name, full):
	best_ops_per_s = 0
	best_len = -1
	best_repeats = -1
	
	if full:
		mults = mults_muladd
		cols = adds
	else:
		mults = mults_mulonly
		cols = [1]
	
	for r in mults:
		for c in cols:
			M = makeMat(r, c, full)
			M.addFilter(twitterFilter)
			v = kdt.Vec.ones(c, sparse=True)
			sr = kdt.sr_select2nd
			
			for rpt in repeats:
				begin = time.time()
				for i in range(rpt):
					M.SpMV(v, semiring=sr, inPlace=True)

				t = time.time() - begin
				if full:
					op = r*c + r*(c-1) # multiplies + adds
				else:
					op = r
				
				ops = op*rpt / t
				if best_ops_per_s < ops:
					kdt.p("%5d-by-%5d (row-by-col), repeat %3d times: %f (took %f seconds)"%(r, c, rpt, ops, t))
					best_ops_per_s = ops
					best_r = r
					best_c = c
					best_repeats = rpt
	
	p = kdt._nproc()
	kdt.p("BEST %s on %3d procs: %5d-by-%5d (row-by-col), repeat %3d times: %f Op/s"%(name, p, best_r, best_c, best_repeats, best_ops_per_s))


kdt.p("running on Multiply only:")
runExperiment("mult only", False)
kdt.p("running on Multiply+Add:")
runExperiment("mult+add ", True)
