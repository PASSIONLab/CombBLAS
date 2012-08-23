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




# used for a percentage-based filtering scheme
def Twitter_obj_randomizer(obj, bin):
	obj.count = 1
	obj.latest = random.randrange(0, 5000) # not 10,000 because we'll get two latests added together during symmetrication
	obj.follower = 0
	return obj

# random numbers on the filter will mean that the graph will no longer be undirected
# make sure edges and backedges share the same value
def SymmetricizeRands(G, name):
	before = time.time()
	GT = G.copy()
	GT.reverseEdges()
	def callback(g, t):
		g.latest = (g.latest+t.latest)
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
			tempG = kdt.DiGraph.generateRMAT(scale, edgeFactor=5, initiator=[0.3, 0.1, 0.1, 0.5], delIsolated=False, element=1.0)
			kdt.p("Generated RMAT graph in %fs: \t%d\t vertices and \t%d\t edges."%(time.time()-before, tempG.nvert(), tempG.nedge()))
			GRmat = kdt.DiGraph(nv=tempG.nvert(), element=kdt.Obj2())
			GRmat.e.eWiseApply(tempG.e, op=Twitter_obj_randomizer, allowANulls=True, inPlace=True)
			kdt.p("Converted in %fs. GRmat has %d vertices and %d edges."%(time.time()-before, GRmat.nvert(), GRmat.nedge()))
			SymmetricizeRands(GRmat, "GRmat")
	elif whatToDo[3] == '2':
		if GER is None:
			before = time.time()
			# fake Erdos-Renyi generator with equal weight RMAT
			tempG = kdt.DiGraph.generateRMAT(scale, edgeFactor=5, initiator=[0.25, 0.25, 0.25, 0.25], delIsolated=False, element=1.0)
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
