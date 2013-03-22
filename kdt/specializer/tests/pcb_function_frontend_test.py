from pcb_operator_convert import *
from pcb_function import *
from pcb_function_frontend import *


import unittest

def get_ast(func):
    import ast, inspect
    return ast.parse(inspect.getsource(func).lstrip())

def assert_contains_node_type(root_node, node_type):
    assert ContainsNodeTypeVisitor().contains(root_node, node_type), "Expected tree to contain node of type %s" % node_type

class ContainsNodeTypeVisitor(ast.NodeVisitor):
    def contains(self, root_node, node_type):
        self.node_type = node_type
        self.result = False
        self.visit(root_node)
        return self.result

    def visit(self, node):
        if isinstance(node, self.node_type):
            self.result = True
        self.generic_visit(node)


class SimpleReturnTest(unittest.TestCase):
    def test_conversion(self):
        class Foo(PcbUnaryFunction):
            def __call__(self, x):
                return x
        
        model = PcbUnaryFunctionFrontEnd().parse(get_ast(Foo.__call__))

        assert_contains_node_type(model, UnaryFunction)
        assert_contains_node_type(model, FunctionReturn)

if __name__ == '__main__':
    unittest.main()