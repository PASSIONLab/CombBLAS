import time
time_very_beginning = time.time()

import kdt
import math
import sys
import random
from stats import splitthousands, printstats

kdt.PDO_enable(False) # disable Python-Defined Objects, since we're using Obj2 directly.

if (len(sys.argv) < 2):
	kdt.p("Usage: python %s scale [1]"%(sys.argv[0]))
	kdt.p("The 1st argument is the scale for graph generation. There will be 2^scale vertices.")
	kdt.p("The 2nd argument specifies the runs to do. The 3 letters specify the semiring and filtering SEJITS usage and OTF/Mat, the number specifies the matrix.")
	kdt.p("p=Python, s=SEJITS; o=On-The-Fly, m=Materialize; 1=RMAT (NOT the Graph500 RMAT), 2=Erdos-Renyi, 4=Torus")
	kdt.p("")
	kdt.p("Examples:")
	kdt.p("python %s 16 pp1"%(sys.argv[0]))
	kdt.p("python %s 14 ps4"%(sys.argv[0]))
	sys.exit()

scale = int(sys.argv[1])

whatToDoArg = ["ppo1"]
if (len(sys.argv) >= 3):
	whatToDoArg = sys.argv[2:]

def verifyMIS(G, MIS):
	ok = True
	
	#def ret1(x, y):
	#	return 1
	#SR = kdt.sr(ret1, ret1)
	SR = kdt.sr_select2nd
	invMIS = G.e.SpMV(MIS, semiring=SR, inPlace=False)
		
	# sanity check
	if (invMIS.nnn() + MIS.nnn() != G.nvert()):
		kdt.p("size of MIS does not match: MIS size=%d, inverse size=%d, (sum=%d, should be number of verts=%d)"%(MIS.nnn(), invMIS.nnn(), (invMIS.nnn() + MIS.nnn()), G.nvert()))
		ok = False
	
	# make sure there is no overlap
	overlap_set = invMIS.eWiseApply(MIS, op=(lambda x,y: 1), allowANulls=False, allowBNulls=False, allowIntersect=True, inPlace=False)
	if (overlap_set.nnn() != 0):
		kdt.p("MIS and invMIS overlap in %d vertices!"%(overlap_set.nnn()))
		ok=False
	
	if ok:
		return "verification succeeded"
	else:
		return "VERIFICATION FAILED"




# this is the MIS algorithm from Algorithms.py but is currently placed here because of the incomplete front-end
# translation for SEJITS semirings

MISrand = None
MISreturn1 = None
MISis2ndSmaller = None
MISmyMin = None
MISselect2nd = None
MISselect2nd_d = None

def initialize_sejitsMIS():
        global MISrand, MISreturn1, MISis2ndSmaller, MISmyMin, MISselect2nd, MISselect2nd_d
        import kdt.pyCombBLAS
	# callbacks used by MIS
#	def rand( verc ):
#		import random
#		if verc > 0:
#			return random.random()
        import pcb_predicate, pcb_function, pcb_function_sm as f_sm
        import asp.codegen.python_ast as ast
        MISrand = pcb_function.PcbUnaryFunction(f_sm.UnaryFunction(f_sm.Identifier("v"),
                                                                   f_sm.FunctionReturn(f_sm.Identifier("_random()")))).get_function()

	
#	def return1(x, y):
#		return 1

        MISreturn1 = pcb_function.PcbBinaryFunction(f_sm.BinaryFunction([f_sm.Identifier("x"), f_sm.Identifier("y")],
                                                                        f_sm.FunctionReturn(f_sm.Constant(1)))).get_function()
	
#	def is2ndSmaller(m, c):
#		return (c < m)

        class Is2ndSmaller(pcb_predicate.PcbBinaryPredicate):
                def __call__(self, x, y):
                    return y<x

        MISis2ndSmaller = Is2ndSmaller().get_predicate()
	
#	def myMin(x,y):
#		if x<y:
#			return x
#		else:
#			return y

        MISmyMin = pcb_function.PcbBinaryFunction(f_sm.BinaryFunction([f_sm.Identifier("x"), f_sm.Identifier("y")],
                                                                      f_sm.IfExp(f_sm.Compare(f_sm.Identifier("x"),
                                                                                              ast.Lt(),
                                                                                              f_sm.Identifier("y")),
                                                                                 f_sm.FunctionReturn(f_sm.Identifier("x")),
                                                                                 f_sm.FunctionReturn(f_sm.Identifier("y"))))).get_function()
	
#	def select2nd(x, y):
#		return y

        MISselect2nd_d = pcb_function.PcbBinaryFunction(f_sm.BinaryFunction([f_sm.Identifier("x"), f_sm.Identifier("y")],
                                                                          f_sm.FunctionReturn(f_sm.Identifier("y")))).get_function()

        MISselect2nd = pcb_function.PcbBinaryFunction(f_sm.BinaryFunction([f_sm.Identifier("x"), f_sm.Identifier("y")],
                                             f_sm.FunctionReturn(f_sm.Identifier("y"))),
                         types=["double", "Obj2", "double"]).get_function()


def sejitsMIS(G, use_SEJITS_SR=True):
	"""
	find the Maximal Independent Set of an undirected graph.

	Output Arguments:
		ret: a sparse Vec of length equal to the number of vertices where
		     ret[i] exists and is 1 if i is part of the MIS.
	"""
        global MISrand, MISreturn1, MISis2ndSmaller, MISmyMin, MISselect2nd, MISselect2nd_d
	graph = G
	
	Gmatrix = graph.e
	nvert = graph.nvert();
	
        def myMin(x,y):
            if (x<y):
                return x
            else:
                return y

        def select2nd(x,y):
            return y

	def is2ndSmaller(m, c):
		return (c < m)

        def return1(x,y):
            return 1




	# the final result set. S[i] exists and is 1 if vertex i is in the MIS
	S = kdt.Vec(nvert, sparse=True)
	
	# the candidate set. initially all vertices are candidates.
	# this vector doubles as 'r', the random value vector.
	# i.e. if C[i] exists, then i is a candidate. The value C[i] is i's r for this iteration.
	C = kdt.Vec.ones(nvert, sparse=True)
		
	while (C.nnn()>0):
		# label each vertex in C with a random value
		C.apply(MISrand)
		
		# find the smallest random value among a vertex's neighbors
		# In other words:
		# min_neighbor_r[i] = min(C[j] for all neighbors j of vertex i)
		min_neighbor_r = Gmatrix.SpMV(C, kdt.sr(MISmyMin,MISselect2nd)) # could use "min" directly

		# The vertices to be added to S this iteration are those whose random value is
		# smaller than those of all its neighbors:
		# new_S_members[i] exists if C[i] < min_neighbor_r[i]
		new_S_members = min_neighbor_r.eWiseApply(C, MISreturn1, doOp=MISis2ndSmaller, allowANulls=True, allowBNulls=False, inPlace=False, ANull=2)

		# new_S_members are no longer candidates, so remove them from C
		C.eWiseApply(new_S_members, MISreturn1, allowANulls=False, allowIntersect=False, allowBNulls=True, inPlace=True)

		# find neighbors of new_S_members
		new_S_neighbors = Gmatrix.SpMV(new_S_members, kdt.sr(MISselect2nd_d,MISselect2nd))

		# remove neighbors of new_S_members from C, because they cannot be part of the MIS anymore
		C.eWiseApply(new_S_neighbors, MISreturn1, allowANulls=False, allowIntersect=False, allowBNulls=True, inPlace=True)

		# add new_S_members to S
		S.eWiseApply(new_S_members, MISreturn1, allowANulls=True, allowBNulls=True, inPlace=True)
		
	return S





# used for a percentage-based filtering scheme
def Twitter_obj_randomizer(obj, bin):
	obj.count = 1
	obj.latest = random.randrange(0, 10000) # not 10,000 because we'll get two latests added together during symmetrication
	obj.follower = 0
	
	return obj

# random numbers on the filter will mean that the graph will no longer be undirected
# make sure edges and backedges share the same value
def SymmetricizeRands(G, name):
	before = time.time()
	GT = G.copy()
	GT.reverseEdges()
	def callback(g, t):
		# have to deterministically choose between one of the two values.
		# cannot just add them because that will change the distribution (small values are unlikely to survive)
		if (int(g.latest + t.latest) & 1) == 1:
			g.latest = min(g.latest, t.latest)
		else:
			g.latest = max(g.latest, t.latest)
		return g
	G.e.eWiseApply(GT.e, callback, inPlace=True)
	kdt.p("Symmetricized randoms in %fs. %s has %d vertices and %d edges."%(time.time()-before, name, G.nvert(), G.nedge()))
	

GRmat = None
GER = None
Gtorus = None
# generate the graphs we need
for whatToDo in whatToDoArg:
	if whatToDo[3] == '1':
		if GRmat is None:
			before = time.time()
			tempG = kdt.DiGraph.generateRMAT(scale, edgeFactor=5, initiator=[0.3, 0.1, 0.1, 0.5], element=1.0) #, delIsolated=False) # delIsolated commented out to keep random number generator deterministic
			kdt.p("Generated RMAT graph in %fs: \t%d\t vertices and \t%d\t edges."%(time.time()-before, tempG.nvert(), tempG.nedge()))
			GRmat = kdt.DiGraph(nv=tempG.nvert(), element=kdt.Obj2())
			GRmat.e.eWiseApply(tempG.e, op=Twitter_obj_randomizer, allowANulls=True, inPlace=True)
			kdt.p("Converted in %fs. GRmat has %d vertices and %d edges."%(time.time()-before, GRmat.nvert(), GRmat.nedge()))
			SymmetricizeRands(GRmat, "GRmat")
	elif whatToDo[3] == '2':
		if GER is None:
			before = time.time()
			# fake Erdos-Renyi generator with equal weight RMAT
			tempG = kdt.DiGraph.generateRMAT(scale, edgeFactor=5, initiator=[0.25, 0.25, 0.25, 0.25], element=1.0) #, delIsolated=False) # delIsolated commented out to keep random number generator deterministic
			kdt.p("Generated Erdos-Renyi graph in %fs: \t%d\t vertices and \t%d\t edges."%(time.time()-before, tempG.nvert(), tempG.nedge()))
			GER = kdt.DiGraph(nv=tempG.nvert(), element=kdt.Obj2())
			GER.e.eWiseApply(tempG.e, op=Twitter_obj_randomizer, allowANulls=True, inPlace=True)
			kdt.p("Converted in %fs. GER has %d vertices and %d edges."%(time.time()-before, GER.nvert(), GER.nedge()))
			SymmetricizeRands(GER, "GER")
	elif whatToDo[3] == '4':
		if Gtorus is None:
			before = time.time()
			torusDim = int(math.sqrt(2**scale)) # match scale of RMAT
			tempG = kdt.DiGraph.generate2DTorus(torusDim)
			kdt.p("Generated Torus graph in %fs: \t%d\t vertices and \t%d\t edges."%(time.time()-before, tempG.nvert(), tempG.nedge()))
			Gtorus = kdt.DiGraph(nv=tempG.nvert(), element=kdt.Obj2())
			Gtorus.e.eWiseApply(tempG.e, op=Twitter_obj_randomizer, allowANulls=True, inPlace=True)
			kdt.p("Converted in %fs. Gtorus has %d vertices and %d edges."%(time.time()-before, Gtorus.nvert(), Gtorus.nedge()))
			SymmetricizeRands(Gtorus, "Gtorus")
	else:
		raise NotImplementedError,"Unknown graph type %s in %s"%(whatToDo[2], whatToDo)


def run(G, filter, filterPercent, run_ID, use_SEJITS_SR, use_SEJITS_Filter, materialize):
	n_unfiltered_edges = G.nedge()
	G.addEFilter(filter)
	materializeTime = 0
	if materialize:
		kdt.p("--Materializing the filter")
		before = time.time()
		G.e.materializeFilter()
		materializeTime = time.time()-before
		kdt.p("Materialized %f in\t%f\ts."%(filterPercent, materializeTime))
		kdt.p("%f\t: \t%d\t edges survived the filter (%f%%)."%(filterPercent, G.nedge(), 100.0*G.nedge()/n_unfiltered_edges))

	elapsed = [];
	MIS_verts = [];

	for i in range(16):
		start = time.time()
                if use_SEJITS_SR:
                    S = sejitsMIS(G)
                else:
                    S = G.MIS(use_SEJITS_SR=use_SEJITS_SR)
		finish = time.time()
		verifyString = verifyMIS(G, S)
		iter_time = finish-start
		
		kdt.p("%s\t%d\t procs time:\t%f\tMIS size is\t%d\t%s"%(run_ID, kdt._nproc(), iter_time, S.nnn(), verifyString))
		elapsed.append(iter_time)
		MIS_verts.append(S.nnn())
	G.delEFilter(filter)

	# report summary
	if kdt.master():
		print ""
		printstats(elapsed, "%stime\t%f\t"%(run_ID, filterPercent), False, True, True)
		print ""
		printstats(MIS_verts, "%s_MISverts\t%f\t"%(run_ID, filterPercent), False, True, True)
		print ""


for filterPercent in (1, 10, 25, 100):
	filterUpperValue = int(filterPercent*100)
	
	for whatToDo in whatToDoArg:
		# determine the semiring type to use
		if whatToDo[0] == 'p':
			use_SEJITS_SR = False
			SR_str = "Python"
		elif whatToDo[0] == 's':
			use_SEJITS_SR = True
			SR_str = "Sejits"
                        initialize_sejitsMIS()
		else:
			raise ValueError,"Invalid semiring specified in whatToDo %s"%whatToDo
		
		# determine the filter type to use
		if whatToDo[1] == 'p':
			use_SEJITS_Filter = False
			Filter_str = "Python"
		elif whatToDo[1] == 's':
			use_SEJITS_Filter = True
			Filter_str = "Sejits"
		else:
			raise ValueError,"Invalid filter type specified in whatToDo %s"%whatToDo
	
		# determine OTF or Materialize
		if whatToDo[2] == 'o':
			materialize = False
			Mat_str = "OTF"
		elif whatToDo[2] == 'm':
			materialize = True
			Mat_str = "Mat"
		else:
			raise ValueError,"Invalid materialization flag specified in whatToDo %s"%whatToDo
	
		# determine the graph to use
		doRMAT = bool(int(whatToDo[3]) & 1)
		doER = bool(int(whatToDo[3]) & 2)
		doTorus = bool(int(whatToDo[3]) & 4)
	
		G = None
		G_str = None
		## RMAT
		if doRMAT:
			G = GRmat
			G_str = "RMAT"
		
		## Erdos-Renyi
		if doER:
			G = GER
			G_str = "ER"
		
		## torus
		if doTorus:
			G = Gtorus
			G_str = "torus"
		
		run_ID = "%sSR_%sFilter_%s_%s_%d"%(SR_str, Filter_str, G_str, Mat_str, scale)
		
		single_runtime_before = time.time()

		# set up the filter
		def twitterEdgeFilter(e):
			return e.count > 0 and e.latest < filterUpperValue
	
		if use_SEJITS_Filter:
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
			filter = TwitterFilter(filterUpperValue).get_predicate()
			sejits_filter_create_time = time.time()-before
			kdt.p("Created SEJITS filter for \t%d\t%% in\t%f\ts."%(filterPercent, sejits_filter_create_time))
		else:
			filter = twitterEdgeFilter
		run(G, filter, filterPercent, run_ID, use_SEJITS_SR, use_SEJITS_Filter, materialize)

		single_runtime = time.time() - single_runtime_before
		kdt.p("Total runtime for %s on %s %d%% is\t%f\ts."%(whatToDo, run_ID, filterPercent, single_runtime))

kdt.p("Total runtime for everything is %f"%(time.time()-time_very_beginning))
