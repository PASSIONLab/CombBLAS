from pcb_function import *

import unittest



class FullUnaryFunctionTest(unittest.TestCase):
    def test_basic(self):
        class MyFunc(PcbUnaryFunction):
            def __call__(self, x):
                return x+10.0

        import kdt

        # this should not throw an exception
        f = MyFunc().get_function()



if __name__ == '__main__':
    unittest.main()