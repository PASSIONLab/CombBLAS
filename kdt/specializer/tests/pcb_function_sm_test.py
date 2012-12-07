from pcb_operator_convert import *
from pcb_function_sm import *

import unittest


class Select2ndTests(unittest.TestCase):
    def test_conversion(self):
        sm = BinaryFunction([Identifier("x"), Identifier("y")],
                            FunctionReturn(Identifier("y")))
        # this shouldn't raise an exception
        cpp_ast = PcbOperatorConvert().convert(sm, types=["Obj2", "Obj1", "Obj2"])

        print cpp_ast

if __name__ == '__main__':
    unittest.main()