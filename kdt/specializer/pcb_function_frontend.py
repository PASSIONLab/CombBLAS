"""

Takes a Python AST of a Pcb*Function-derived class and converts the __call__ method to
a semantic model corresponding to nodes from function_sm.
"""

import asp.codegen.ast_tools as ast_tools
import asp.codegen.python_ast as ast
from pcb_function import *
from pcb_function_sm import *

class PcbUnaryFunctionFrontEnd(ast_tools.NodeTransformer):
    def parse(self, node, env={}):
        self.env = env
        return self.visit(node)

    def visit_Module(self, node):
        # this assertion should never be false, assuming the right thing is passed to this class
        assert len(node.body) == 1, "Front end conversion failed with too many nodes in top-level module body"

        return self.visit(node.body[0])

    def visit_FunctionDef(self, node):
        # we only translate the __call__ function, with a single statement inside it
        assert node.name == "__call__", "Front end conversion failed with unknown function: " + node.name.id

        #FIXME: do we want to relax this constraint? will require changes to the SM grammar as well.
        assert len(node.body) == 1, "Front end conversion failed: only a single statement is supported for the function body"

        return UnaryFunction(self.visit(node.args), self.visit(node.body[0]))

    def visit_arguments(self, node):
        # we restrict the arguments to be (self, something)
        assert len(node.args) == 2, "Front end conversion failed with too many arguments to function"
        assert node.args[0].id == "self", "Front end converion failed with unknown argument at first position"

        return Identifier(node.args[1].id)

    def visit_Return(self, node):
        return FunctionReturn(self.visit(node.value))

    def visit_Name(self, node):
        # by some quirk of Python's parser, it treats "True" and "False" boolean values as
        # identifiers.  Don't ask me.
        if node.id == "True" or node.id == "False":
            return BoolConstant(True if node.id == "True" else False)

        # otherwise, it must be either self or the input
        #FIXME: implement a check to make sure it is either self or input
        return Identifier(node.id)

    def visit_If(self, node):
        # again, single statements in each body
        assert len(node.body) == 1, "Front end conversion failed: Only a single statement allowed in the body of if statement"
        assert len(node.orelse) == 1, "Front end conversion failed: Single statement required in else clause of if statement"

        return IfExp(self.visit(node.test), self.visit(node.body[0]), self.visit(node.orelse[0]))

    def visit_BoolOp(self, node):
        return BoolOp(node.op, map(self.visit, node.values))
        
    def visit_BinOp(self, node):
        return BinaryOp(self.visit(node.left), node.op, self.visit(node.right))

    def visit_Compare(self, node):
        #FIXME: under what circumstances can node.ops/node.comparators be more than a single item?
        return Compare(self.visit(node.left), node.ops[0], self.visit(node.comparators[0]))

    def visit_Attribute(self, node):
        # a special case is if this is self.foo
        if type(node.value) == ast.Name and node.value.id == "self":
            # lookup the item in our env
            #FIXME: catch a key error here and return a better explanation of why it failed
            val = self.env[node.attr]
            assert isinstance(val, int) or isinstance(val, float), "Values of the form self.foo must be numeric types only"
            return Constant(val)
        else:
            return Attribute(self.visit(node.value), Identifier(node.attr))
        

    def visit_Num(self, node):
        return Constant(node.n)


# this is for binary predicates
class PcbBinaryFunctionFrontEnd(PcbUnaryFunctionFrontEnd):
    def visit_FunctionDef(self, node):
        # we only translate the __call__ function with a single (functional) statement inside
        assert node.name == "__call__", "Front end conversion failed with unknown function: " + node.name.id
        assert len(node.body) == 1, "Front end conversion failed: only a single statement is supported for the function body"

        return BinaryFunction(self.visit(node.args), self.visit(node.body[0]))

    def visit_arguments(self, node):
        # arguments must be self, something, something
        assert len(node.args) == 3, "Front end conversion failed with wrong number of arguments to __call__ function"
        assert node.args[0].id == "self", "Front end conversion failed with unknwown argument at first position"

        return [Identifier(x.id) for x in node.args[1:]]

    
