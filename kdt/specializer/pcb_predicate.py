from pcb_predicate_sm import *
from pcb_operator_convert import *

from asp.jit import asp_module
from asp.codegen import cpp_ast
import kdt

from pdo_utils import get_CPP_types, add_PDO_stubs

class PcbUnaryPredicate(object):
    """
    Top-level for PCB predicate functions.

    """

    def gen_get_predicate(self, types):
        # this function generates the code that passes a specialized UnaryPredicateObj back to Python for later use
        # TODO: this should actually generate all the filled possible customFuncs for all datatypes
        import asp.codegen.templating.template as template
        
        specialized_function_slot = "customFunc%s" % (types[1])

        t = template.Template("""


            PyObject* get_predicate()
            {
              using namespace op;
              swig_module_info* module = SWIG_Python_GetModule(NULL);

              swig_type_info* ty = SWIG_TypeQueryModule(module, module, "op::UnaryPredicateObj *");

              UnaryPredicateObj_SEJITS* retf = new UnaryPredicateObj_SEJITS();
              retf->${specialized_function_slot} = &myfunc;

              UnaryPredicateObj* retO = new UnaryPredicateObj();
              retO->worker = *retf;

              PyObject* ret_obj = SWIG_NewPointerObj((void*)(retO), ty, SWIG_POINTER_OWN | 0);

              return ret_obj;
            }
            """, disable_unicode=True)

        return t.render(specialized_function_slot=specialized_function_slot)

    def get_predicate(self, types=["bool", "Obj2"], PDO=False):
        #FIXME: do we save the vars of the instance so they can be passed in to translation machinery?

        # see if the result is cached
        if not hasattr(self, '_func_cache'):
            self._func_cache = {}
            
        if str(types) in self._func_cache:
            return self._func_cache[str(types)]

        try:
            # create semantic model
            intypes = types
            import ast, inspect
            from pcb_predicate_frontend import *
            sm = PcbUnaryPredicateFrontEnd().parse(ast.parse(inspect.getsource(self.__call__).lstrip()), env=vars(self))
            types = intypes

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

            converted = PcbOperatorConvert().convert(sm, types=get_CPP_types(types))
            
            if PDO:
                add_PDO_stubs(self.mod, converted, types)
            
            #FIXME: try all types?
            self.mod.add_function("myfunc", converted)
            self.mod.add_function("get_predicate", self.gen_get_predicate(get_CPP_types(types)))

            kdt.p_debug(self.mod.generate())
            pred = self.mod.get_predicate()
            pred.setCallback(self)

            # cache the result
            self._func_cache[str(types)] = pred

        except Exception as ex:
            kdt.p("WARNING: Specialization failed, proceeding with pure Python.")
            kdt.p_debug(str(ex))
            pred = self
        return pred



class PcbBinaryPredicate(PcbUnaryPredicate):
    """
    Top-level for PCB binary predicate functions.

    """


    def gen_get_predicate(self, types):
        # this function generates the code that passes a specialized UnaryPredicateObj back to Python for later use
        # TODO: this should actually generate all the filled possible customFuncs for all datatypes

        specialized_function_slot = "customFunc%s%s" % (types[1], types[2])

        import asp.codegen.templating.template as template
        t = template.Template("""


            PyObject* get_predicate()
            {
              using namespace op;
              swig_module_info* module = SWIG_Python_GetModule(NULL);

              swig_type_info* ty = SWIG_TypeQueryModule(module, module, "op::BinaryPredicateObj *");

              BinaryPredicateObj_SEJITS* retf = new BinaryPredicateObj_SEJITS();
              retf->${specialized_function_slot} = &myfunc;

              BinaryPredicateObj* retO = new BinaryPredicateObj();
              retO->worker = *retf;

              PyObject* ret_obj = SWIG_NewPointerObj((void*)(retO), ty, SWIG_POINTER_OWN | 0);

              return ret_obj;
            }
            """, disable_unicode=True)

        return t.render(specialized_function_slot=specialized_function_slot)

    def get_predicate(self, types=["bool", "double", "double"], PDO=False):
        #FIXME: do we then save the vars of the instance so they can be passed in to translation machinery?

        # see if the result is cached
        if not hasattr(self, '_func_cache'):
            self._func_cache = {}
            
        if str(types) in self._func_cache:
            return self._func_cache[str(types)]

        try:
            # create semantic model
            intypes = types
            import ast, inspect
            from pcb_predicate_frontend import *
            types = intypes
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

            converted = PcbOperatorConvert().convert(sm, types=get_CPP_types(types))
            
            if PDO:
                add_PDO_stubs(self.mod, converted, types)

            #FIXME: try all types?
            self.mod.add_function("myfunc", converted)
            self.mod.add_function("get_predicate", self.gen_get_predicate(get_CPP_types(types)))

            kdt.p_debug(self.mod.generate())
            pred = self.mod.get_predicate()
            pred.setCallback(self)

            # cache the result
            self._func_cache[str(types)] = pred

        except Exception as ex:
            kdt.p("WARNING: Specialization failed, proceeding with pure Python.")
            kdt.p_debug(str(ex))
            pred = self
        return pred


