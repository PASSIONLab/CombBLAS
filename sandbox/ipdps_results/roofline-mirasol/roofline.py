import sys
import kdt
import kdt.pyCombBLAS as pcb
import time

kdt.PDO_enable(False)

sweep_dimensions = True
repeats = range(1,20)#[1, 2, 3, 4, 5, 10, 13, 15, 20]
printProgress = False
useSEJITS = True

try:
	from pcb_predicate import *
	useSEJITS = True
except ImportError:
	#useSEJITS = False
	#kdt.p("SEJITS parts not found, doing Python only.")
	kdt.p("problems importing SEJITS, will try anyway.")


if sweep_dimensions:
	mults_mulonly = range(8, 2000, 8)
	mults_muladd = range(1, 40, 1)
	adds = range(8, 200, 16)
	eWiseVecLens = mults_mulonly
else:
	mults_mulonly = [100]
	mults_muladd = [20]
	adds = [20]

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

	eWiseVecLens = mults_mulonly

#################################################

# Python filter
def py_twitterFilter(e):
	#return (e.follower == 0 and e.count >= 0 and e.latest < 946684800)
	return (e.count >= 0 and e.latest < 946684800)

def py_select1st(x, y):
	return x

# SEJITS filter
if useSEJITS: # put here so if the system doesn't have SEJITS it won't crash
	from pcb_predicate import *

	class TwitterFilter(PcbUnaryPredicate):
		def __init__(self, filterUpperValue):
			self.filterUpperValue = filterUpperValue
			super(TwitterFilter, self).__init__()
		def __call__(self, e):
			if (e.count > 0 and e.latest < self.filterUpperValue):
					return True
			else:
					return False
	before = time.time()
	sejits_filter = TwitterFilter(946684800).get_predicate()
	sejits_filter_create_time = time.time()-before
	kdt.p("Created SEJITS filter in\t%f\ts."%(sejits_filter_create_time))

# SEJITS bin funcs
if useSEJITS:
        import pcb_predicate, pcb_function, pcb_function_sm as f_sm

        s1st_DO = pcb_function.PcbBinaryFunction(f_sm.BinaryFunction([f_sm.Identifier("x"), f_sm.Identifier("y")],
                                             f_sm.FunctionReturn(f_sm.Identifier("x"))),
                         types=["double", "double", "Obj2"])
        s2nd_OD = pcb_function.PcbBinaryFunction(f_sm.BinaryFunction([f_sm.Identifier("x"), f_sm.Identifier("y")],
                                             f_sm.FunctionReturn(f_sm.Identifier("y"))),
                         types=["double", "Obj2", "double"])
        s2nd_DD = pcb_function.PcbBinaryFunction(f_sm.BinaryFunction([f_sm.Identifier("x"), f_sm.Identifier("y")],
                                               f_sm.FunctionReturn(f_sm.Identifier("y"))),
                           types=["double", "double", "double"])
        sejits_select1st_DO = s1st_DO.get_function()
        sejits_select2nd_OD = s2nd_OD.get_function()
        sejits_select2nd_DD = s2nd_DD.get_function()


#################################################

def makeMat(rows, cols, full):
	if full:
		return kdt.Mat.ones(cols, rows, element=kdt.Obj2())
	else:
		i = kdt.Vec.range(rows, sparse=False)
		j = kdt.Vec(rows, element=0, sparse=False)
		v = kdt.Vec(rows, element=kdt.Obj2(), sparse=False)

		return kdt.Mat(i, j, v, cols, rows)

def runSpMVExperiment(filter, sr, full, name):
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
			if printProgress:
				kdt.p("trying %5d-by-%5d (row-by-col)"%(r, c))
			M = makeMat(r, c, full)
			M.addFilter(filter)
			v = kdt.Vec.ones(c, sparse=True)

			for rpt in repeats:

				if full:
					op = rpt*(r*c + r*(c-1)) # multiplies + adds
				else:
					op = rpt*r

				#op *= 10000 # LOTSOFRUNS

				#kdt.p("b %d\t(Ops) on \t%d procs\t%5d-by-%5d\t(row-by-col), repeat\t%3d\ttimes"%(op, p, r, c, rpt))
				begin = time.time()
				for i in range(rpt):
					M.SpMV(v, semiring=sr, inPlace=True)
				t = time.time() - begin
				#kdt.p("%f"%(t))

				#kdt.p("t")
				ops = op / t
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

#################################################

def runEWiseMulExperiment(filter, binop, e, name):
	best_op = 0
	best_ops_per_s = 0
	best_len = -1
	best_repeats = -1

	for vl in eWiseVecLens:
		if printProgress:
			kdt.p("trying vec length %d"%(vl))
		vec = kdt.Vec(vl, element=e, sparse=False)
		vec.addFilter(filter)
		resVec = kdt.Vec(vl, element=0, sparse=False)
		for rpt in repeats:
			begin = time.time()
			for i in range(rpt):
				resVec.eWiseApply(vec, binop, inPlace=True)
			t = time.time() - begin

			op = vl*rpt
			ops = op / t
			if best_ops_per_s < ops:
				#kdt.p("len %5d, repeat %3d times: %f (took %f seconds)"%(vl, r, ops, t))
				best_op = op
				best_ops_per_s = ops
				best_len = vl
				best_repeats = rpt
				best_time = t

	p = kdt._nproc()
	#kdt.p("BEST %s on %3d procs: len %5d, repeat %3d times: %f Op/s"%(name, p, best_len, best_repeats, best_ops_per_s))
	kdt.p("BEST\t%s\ton %3d procs:\t%5d (len), repeat %3d times on \t%f\t ops (\t%f\tsec):\t%f\tOp/s"%(name, p, best_len, best_repeats, best_op, best_time, best_ops_per_s))

#################################################

#kdt.p("running SpMV tests")
#runSpMVExperiment(py_twitterFilter, kdt.sr_select2nd, False, "SpMV mult only (C++/Python)")
#runSpMVExperiment(py_twitterFilter, kdt.sr_select2nd, True, "SpMV mult+add (C++/Python)")

kdt.p("running eWiseApply tests")
runEWiseMulExperiment(py_twitterFilter, py_select1st, kdt.Obj2(), "eWiseApply twitter filtered select1st (Python/Python)")

if useSEJITS:
	runEWiseMulExperiment(py_twitterFilter, sejits_select1st_DO, kdt.Obj2(), "eWiseApply twitter filtered select1st (SEJITS/Python)")
	runEWiseMulExperiment(sejits_filter, py_select1st, kdt.Obj2(), "eWiseApply twitter filtered select1st (Python/SEJITS)")
	runEWiseMulExperiment(sejits_filter, sejits_select1st_DO, kdt.Obj2(), "eWiseApply twitter filtered select1st (SEJITS/SEJITS)")
