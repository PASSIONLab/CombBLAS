import sys
import kdt
import kdt.pyCombBLAS as pcb
import time

mults_mulonly = [100]
mults_muladd = [20]
adds = [20] 
repeats = [1]

p = kdt._nproc()
if p == 9:
	mults_mulonly = [297]
	mults_muladd = [18]
	adds=[18]
if p == 16:
	mults_mulonly = [640]
	mults_muladd = [32]
	adds=[32]
if p == 25:
	mults_mulonly = [1000]
	mults_muladd = [50]
	adds = [50]
if p==36:
	mults_mulonly = [720]
	mults_muladd = [72]
	adds = [72]
if p==49:
	mults_mulonly = [980]
	mults_muladd = [98]
	adds = [98]

def twitterFilter(e):
	return (e.follower == 0 and e.count >= 0 and e.latest < 946684800)

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
		#kdt.p("a")
	else:
		mults = mults_mulonly
		cols = [1]
		#kdt.p("m")
	
	#p = kdt._nproc()
	#mults = [2*p]
	#cols = [p]
	for r in mults:
		for c in cols:
			M = makeMat(r, c, full)
			M.addFilter(twitterFilter)
			v = kdt.Vec.ones(c, sparse=True)
			sr = kdt.sr_select2nd
			
			for rpt in repeats:

				if full:
					op = r*c + r*(c-1) # multiplies + adds
				else:
					op = r
				
				op *= 10000 # LOTSOFRUNS

				#kdt.p("b %d\t(Ops) on \t%d procs\t%5d-by-%5d\t(row-by-col), repeat\t%3d\ttimes"%(op, p, r, c, rpt))
				begin = time.time()
				for i in range(rpt):
					M.SpMV(v, semiring=sr, inPlace=True)
				t = time.time() - begin
				#kdt.p("%f"%(t))
				
				#kdt.p("t")
				ops = op*rpt / t
				if best_ops_per_s < ops:
					#kdt.p("%5d-by-%5d (row-by-col), repeat %3d times: %f (took %f seconds)"%(r, c, rpt, ops, t))
					best_op = op
					best_ops_per_s = ops
					best_r = r
					best_c = c
					best_repeats = rpt
					best_time = t
	
	p = kdt._nproc()
	kdt.p("BEST\t%s\ton %3d procs:\t%5d-by-%5d (row-by-col), repeat %3d times on \t%f\t ops (\t%f\tsec):\t%f\tOp/s"%(name, p, best_r, best_c, best_repeats, best_op, best_time, best_ops_per_s))


#kdt.p("running on Multiply only:")
runExperiment("mult only", False)
#kdt.p("running on Multiply+Add:")
runExperiment("mult+add ", True)
