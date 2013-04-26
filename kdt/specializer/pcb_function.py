from pcb_function_sm import *
from pcb_operator_convert import *

from asp.jit import asp_module
from asp.codegen import cpp_ast
import kdt

from pdo_utils import get_CPP_types, add_PDO_stubs

class PcbBinaryFunction(object):
    """
    Top-level class for BinaryFunctions.
    """


    def gen_get_function(self, types=["double", "double", "double"]):
        # this function generates the code that passes a specialized BinaryFunctionObj back to python
        #TODO: this needs to generate the proper customFunc given the input datatypes
        # IN FACT, needs to generate ALL possible type specializations

        specialized_function_slot = "%s%s_%s" % (types[1], types[2], types[0])

        import asp.codegen.templating.template as template
        t = template.Template("""


            PyObject* get_function()
            {
              using namespace op;
              swig_module_info* module = SWIG_Python_GetModule(NULL);

              swig_type_info* ty = SWIG_TypeQueryModule(module, module, "op::BinaryFunctionObj *");

              BinaryFunctionObj_SEJITS* retf = new BinaryFunctionObj_SEJITS(Py_None);
              retf->customFunc_${specialized_function_slot} = &myfunc;

              BinaryFunctionObj* retO = new BinaryFunctionObj();
              retO->worker = *retf;

              PyObject* ret_obj = SWIG_NewPointerObj((void*)(retO), ty, SWIG_POINTER_OWN | 0);

              return ret_obj;
            }
            """, disable_unicode=True)

        return t.render(specialized_function_slot=specialized_function_slot)


    def get_function(self, types=["double", "double", "double"], PDO=False):
        # see if the result is cached
        if not hasattr(self, '_func_cache'):
            self._func_cache = {}
            
        if str(types) in self._func_cache:
            return self._func_cache[str(types)]

        try:
            # create semantic model
            intypes = types
            import ast, inspect
            #from pcb_function_frontend import * #AL: this produces a SyntaxWarning
            from pcb_function_frontend import PcbUnaryFunctionFrontEnd, PcbBinaryFunctionFrontEnd

            types = intypes
            sm = PcbBinaryFunctionFrontEnd().parse(ast.parse(inspect.getsource(self.__call__).lstrip()), env=vars(self))

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

            while "-bundle" in self.mod.backends["c++"].toolchain.cflags:
                self.mod.backends["c++"].toolchain.cflags.remove("-bundle")

            while "-bundle" in self.mod.backends["c++"].toolchain.ldflags:
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
                        
            #FIXME: pass correct types, or try all types, or do SOMETHING that's smarter than this hardwired crap
            self.mod.add_function("myfunc", converted)
            self.mod.add_function("get_function", self.gen_get_function(types=get_CPP_types(types)))

            kdt.p_debug(self.mod.generate())
            ret = self.mod.get_function()
            ret.setCallback(self)
            
            # cache the result
            self._func_cache[str(types)] = ret
            return ret
        except Exception as e:
            kdt.p("WARNING: Specialization failed, proceeding with pure Python.")
            kdt.p_debug(str(e))
            return self



class PcbUnaryFunction(object):
    """
    Top-level class for UnaryFunctions.
    """

    def gen_get_function(self, types=["double", "double"]):
        # this function generates the code that passes a specialized BinaryFunctionObj back to python
        #TODO: this needs to generate the proper customFunc given the input datatypes
        # IN FACT, needs to generate ALL possible type specializations

        specialized_function_slot = "%s_%s" % (types[1], types[0])

        import asp.codegen.templating.template as template
        t = template.Template("""


            PyObject* get_function()
            {
              using namespace op;
              swig_module_info* module = SWIG_Python_GetModule(NULL);

              swig_type_info* ty = SWIG_TypeQueryModule(module, module, "op::UnaryFunctionObj *");

              UnaryFunctionObj_SEJITS* retf = new UnaryFunctionObj_SEJITS(Py_None);
              retf->customFunc_${specialized_function_slot} = &myfunc;

              UnaryFunctionObj* retO = new UnaryFunctionObj();
              retO->worker = *retf;

              PyObject* ret_obj = SWIG_NewPointerObj((void*)(retO), ty, SWIG_POINTER_OWN | 0);

              return ret_obj;
            }
            """, disable_unicode=True)

        return t.render(specialized_function_slot=specialized_function_slot)



    def get_function(self, types=["double", "double"], PDO=False):
        # see if the result is cached
        if not hasattr(self, '_func_cache'):
            self._func_cache = {}
            
        if str(types) in self._func_cache:
            return self._func_cache[str(types)]

        try:
            #FIXME: need to refactor "types" everywhere to a different name because it gets aliased ove
            intypes = types
            # create semantic model
            import ast, inspect
            from pcb_function_frontend import PcbUnaryFunctionFrontEnd, PcbBinaryFunctionFrontEnd
            types = intypes
            sm = PcbUnaryFunctionFrontEnd().parse(ast.parse(inspect.getsource(self.__call__).lstrip()), env=vars(self))

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

            while "-bundle" in self.mod.backends["c++"].toolchain.cflags:
                self.mod.backends["c++"].toolchain.cflags.remove("-bundle")

            while "-bundle" in self.mod.backends["c++"].toolchain.ldflags:
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

            kdt.p_debug(types)
            #FIXME: pass correct types, or try all types, or do SOMETHING that's smarter than this hardwired crap
            self.mod.add_function("myfunc", converted)
            self.mod.add_function("get_function", self.gen_get_function(types=get_CPP_types(types)))

            kdt.p_debug(self.mod.generate())
            ret = self.mod.get_function()
            ret.setCallback(self)

            # cache the result
            self._func_cache[str(types)] = ret
            return ret

        except Exception as ex:
            kdt.p_debug(ex.message)
            kdt.p("WARNING: Specialization failed, proceeding with pure Python.")
            return self
