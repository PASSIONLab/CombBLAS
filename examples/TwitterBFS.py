import sys
import kdt
import pygraphviz as pgv
import kdt.pyCombBLAS as pcb

directed = False

#parse arguments
if (len(sys.argv) < 2):
	print "Usage: python %s twittergraph.mtx"%(sys.argv[0])
	sys.exit()


inmatrixfile = sys.argv[1]

# load
print "Reading network from",inmatrixfile
G = kdt.DiGraph.load(inmatrixfile, eelement=kdt.Obj2())

print G

print "Doing BFS on original"
start = 0
parents = G.bfsTree(start)

print parents
sys.exit()

print "Doing BFS on edge type 1 and 3"
def f1(x):
	return (x != 2)
G.e.addFilter(f1)
G.e.materializeFilter()
start = 0
parents = G.bfsTree(start)
G.e.delFilter(f1)

print "Doing BFS on edge type 2 and 3"
def f2(x):
	return (x != 1)
G.e.addFilter(f2)
G.e.materializeFilter()
start = 0
parents = G.bfsTree(start)
G.e.delFilter(f2)
