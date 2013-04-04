from pcb_predicate_sm import *
from pcb_operator_convert import *

from asp.jit import asp_module
from asp.codegen import cpp_ast

class PcbUnaryPredicate(object):
    """
    Top-level for PCB predicate functions.

    """

    def __init__(self):
        #FIXME: consider moving all of this to get_predicate() so that we can ensure it runs.  currently, users have
        # to make sure to call super(..).__init__() in their init function *last* to make self.foo lookups work.
        # problem is, how do we then save the vars of the instance so they can be passed in to translation machinery?

        #FIXME: catch any exceptions and resort to using pure python if this fails
        try:
            # create semantic model
            import ast, inspect
            from pcb_predicate_frontend import *
            sm = PcbUnaryPredicateFrontEnd().parse(ast.parse(inspect.getsource(self.__call__).lstrip()), env=vars(self))

            include_files = ["pyOperationsObj.h"]
            self.mod = asp_module.ASPModule(specializer="kdt")

            # add some include directories
            for x in include_files:
                self.mod.add_header(x)

            self.mod.backends["c++"].toolchain.cflags = ["-g", "-fPIC", "-shared"]
            self.mod.backends["c++"].toolchain.cflags.append("-O3")
            self.mod.backends["c++"].toolchain.defines.append("USESEJITS")
            self.mod.backends["c++"].toolchain.defines.append("FAST_64BIT_ARITHMETIC")
            self.mod.backends["c++"].toolchain.defines.append("PYCOMBBLAS_MPIOK=0")
            self.mod.backends["c++"].toolchain.defines.append("SWIG_TYPE_TABLE=pyCombBLAS")

            if "-bundle" in self.mod.backends["c++"].toolchain.ldflags:
                self.mod.backends["c++"].toolchain.ldflags.remove("-bundle")

            # get location of this file & use to include kdt files
            import inspect, os
            this_file = inspect.currentframe().f_code.co_filename
            installDir = os.path.dirname(this_file)
            self.mod.add_library("pycombblas",
                                 [installDir+"/include"])

            #FIXME: pass correct types, or try all types, or do SOMETHING that's smarter than this hardwired crap
            self.mod.add_function("myfunc", PcbOperatorConvert().convert(sm, types=["bool", "Obj2"]))
            self.mod.add_function("get_predicate", self.gen_get_predicate())

            print self.mod.generate()

        except:
            print "WARNING: Specialization failed, proceeding with pure Python."
            raise

    def gen_get_predicate(self):
        # this function generates the code that passes a specialized UnaryPredicateObj back to Python for later use
        # TODO: this should actually generate all the filled possible customFuncs for all datatypes
        import asp.codegen.templating.template as template
        t = template.Template("""


            PyObject* get_predicate()
            {
              using namespace op;
              swig_module_info* module = SWIG_Python_GetModule(NULL);

              swig_type_info* ty = SWIG_TypeQueryModule(module, module, "op::UnaryPredicateObj *");

              UnaryPredicateObj_SEJITS* retf = new UnaryPredicateObj_SEJITS();
              retf->customFuncO2 = &myfunc;

              UnaryPredicateObj* retO = new UnaryPredicateObj();
              retO->worker = *retf;

              PyObject* ret_obj = SWIG_NewPointerObj((void*)(retO), ty, SWIG_POINTER_OWN | 0);

              return ret_obj;
            }
            """, disable_unicode=True)

        return t.render()

    def get_predicate(self):
        try:
            pred =  self.mod.get_predicate()
        except:
            print "WARNING: Specialization failed, returning pure Python object."
            pred = self
            raise
        return pred



class PcbBinaryPredicate(PcbUnaryPredicate):
    """
    Top-level for PCB binary predicate functions.

    """

    def __init__(self):
        #FIXME: consider moving all of this to get_predicate() so that we can ensure it runs.  currently, users have
        # to make sure to call super(..).__init__() in their init function *last* to make self.foo lookups work.
        # problem is, how do we then save the vars of the instance so they can be passed in to translation machinery?

        #FIXME: catch any exceptions and resort to using pure python if this fails
        try:
            # create semantic model
            import ast, inspect
            from pcb_predicate_frontend import *
            sm = PcbBinaryPredicateFrontEnd().parse(ast.parse(inspect.getsource(self.__call__).lstrip()), env=vars(self))

            include_files = ["pyOperationsObj.h"]
            self.mod = asp_module.ASPModule(specializer="kdt")

            # add some include directories
            for x in include_files:
                self.mod.add_header(x)
            self.mod.backends["c++"].toolchain.cflags = ["-g", "-fPIC", "-shared"]
            self.mod.backends["c++"].toolchain.cflags.append("-O3")
            self.mod.backends["c++"].toolchain.defines.append("USESEJITS")
            self.mod.backends["c++"].toolchain.defines.append("FAST_64BIT_ARITHMETIC")
            self.mod.backends["c++"].toolchain.defines.append("PYCOMBBLAS_MPIOK=0")
            self.mod.backends["c++"].toolchain.defines.append("SWIG_TYPE_TABLE=pyCombBLAS")

            if "-bundle" in self.mod.backends["c++"].toolchain.ldflags:
                self.mod.backends["c++"].toolchain.ldflags.remove("-bundle")

            # get location of this file & use to include kdt files
            import inspect, os
            this_file = inspect.currentframe().f_code.co_filename
            installDir = os.path.dirname(this_file)
            self.mod.add_library("pycombblas",
                                 [installDir+"/include"])

            #FIXME: pass correct types, or try all types, or do SOMETHING that's smarter than this hardwired crap
            self.mod.add_function("myfunc", PcbOperatorConvert().convert(sm, types=["bool", "double", "double"]))
            self.mod.add_function("get_predicate", self.gen_get_predicate())

            print self.mod.generate()

        except:
            print "WARNING: Specialization failed, proceeding with pure Python."
            raise

    def gen_get_predicate(self):
        # this function generates the code that passes a specialized UnaryPredicateObj back to Python for later use
        # TODO: this should actually generate all the filled possible customFuncs for all datatypes
        import asp.codegen.templating.template as template
        t = template.Template("""


            PyObject* get_predicate()
            {
              using namespace op;
              swig_module_info* module = SWIG_Python_GetModule(NULL);

              swig_type_info* ty = SWIG_TypeQueryModule(module, module, "op::BinaryPredicateObj *");

              BinaryPredicateObj_SEJITS* retf = new BinaryPredicateObj_SEJITS();
              retf->customFuncDD = &myfunc;

              BinaryPredicateObj* retO = new BinaryPredicateObj();
              retO->worker = *retf;

              PyObject* ret_obj = SWIG_NewPointerObj((void*)(retO), ty, SWIG_POINTER_OWN | 0);

              return ret_obj;
            }
            """, disable_unicode=True)

        return t.render()
