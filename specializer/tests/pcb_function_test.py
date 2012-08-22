from pcb_operator_convert import *
from pcb_function_sm import *
from pcb_function import *

import unittest


class Select2ndTests(unittest.TestCase):
    def test_get_function(self):
        sm = BinaryFunction([Identifier("x"), Identifier("y")],
                            FunctionReturn(Identifier("y")))
        # this shouldn't raise an exception
        import kdt
        PcbBinaryFunction(sm, types=["double", "double", "double"]).get_function()

        
    def test_full(self):
        sm = BinaryFunction([Identifier("x"), Identifier("y")],
                            FunctionReturn(Identifier("y")))
        # this shouldn't raise an exception
        import kdt
        bf = PcbBinaryFunction(sm, types=["double", "double", "double"])

        # stolen from BFS implementation
        j = kdt.Vec(10, 1, sparse=False)
        k = kdt.Vec(10, -1, sparse=False)
        k.eWiseApply(j, op=(lambda f,p: p), inPlace=True)

        # now let's do it using SEJITS
        j2 = kdt.Vec(10, 1, sparse=False)
        k2 = kdt.Vec(10, -1, sparse=False)
        k2.eWiseApply(j, op=bf.get_function(), inPlace=True)

        for x in range(10):
          self.assertEqual(k2[x], 1.0)

    def test_full_SpMV(self):
        sm = BinaryFunction([Identifier("x"), Identifier("y")],
                            FunctionReturn(Identifier("y")))
        # this shouldn't raise an exception
        import kdt
        bf = PcbBinaryFunction(sm, types=["double", "double", "double"])

        frontier = kdt.Vec(10, sparse=True)
        frontier[2] = 2
        frontier.spRange()
        mat = kdt.DiGraph.fullyConnected(10)
        f = bf.get_function()
        mat.e.SpMV(frontier, semiring=kdt.sr(f,f), inPlace=True)


if __name__ == '__main__':
    unittest.main()