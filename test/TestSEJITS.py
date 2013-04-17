import kdt
import time
from kdt.specializer.pcb_function import *
from kdt.specializer.pcb_function_sm import *
import kdt.specializer.pcb_predicate as pcb_predicate
import ast
import unittest

#kdt.set_verbosity(kdt.DEBUG)

class MulFn(kdt.KDTBinaryFunction):
    def __call__(self, x, y):
        return y

class AddFn(kdt.KDTBinaryFunction):
    def __call__(self, x, y):
        if x>y:
            return x
        else:
            return y

class Sel1st(kdt.KDTBinaryFunction):
    def __call__(self, x, y):
        return x

class NotEq(kdt.KDTBinaryPredicate):
	def __call__(self, x, y):
		return x != y

class TestSEJITS(unittest.TestCase):
    def test_conncomp(self):
        mulFn = MulFn()
        addFn = AddFn()
        select1st = Sel1st()
        noteq = NotEq()
        selectMax = kdt.sr(addFn, mulFn)
        
        def connComp(inmat):
            """
            finds the connected components of the graph by BFS.
            
            Output Arguments:
            self:  a DiGraph instance
            ret:  a dense Vec of length equal to the number of vertices
            in the graph. The value of each element of the vector
            denotes a cluster root for that vertex.
            """
            
            # TODO: use a boolean matrix
            # we want a symmetric matrix with self loops
            n = inmat.nvert()
            G = inmat.e.copy(element=1.0)
            G_T = G.copy()
            G_T.transpose()
            G += G_T
            G += kdt.Mat.eye(n, n, element=1.0)
            G.apply(kdt.op_set(1))
            
            # the semiring we want to use
            
            roots = kdt.Vec.range(n, sparse=False)
            frontier = roots.sparse()
            
            while frontier.nnn() > 0:
                frontier = G.SpMV(frontier, semiring=selectMax)
                
                # prune the frontier of vertices that have not changed
                #frontier.eWiseApply(roots, op=(lambda f,r: f), doOp=(lambda f,r: f != r), inPlace=True)
                frontier.eWiseApply(roots, op=select1st, doOp=noteq, inPlace=True)
                
                # update the roots
                roots[frontier] = frontier
            
            return roots
        
        
        scale = 10
        
        kdt.p("--- Generating plain RMAT of size %s" % scale)
        starttime = time.time()
        mat = kdt.DiGraph.generateRMAT(scale, element=1.0, delIsolated=True)
        
        vec = kdt.Vec.range(mat.nvert()).sparse()
        
        kdt.p("Generated in %s sec" % str(time.time()-starttime))
        
        kdt.p("--- Running first & discarding...")
        ret = connComp(mat)
        #mat.e.SpMV(vec, semiring=selectMax)
        
        kdt.p("--- Running...")
        starttime = time.time()
        
        ret = connComp(mat)
        #mat.e.SpMV(vec, semiring=selectMax)
        
        
        endtime = time.time()
        kdt.p("Elapsed time: %s" % str(endtime-starttime))

if __name__ == '__main__':
    unittest.main()
