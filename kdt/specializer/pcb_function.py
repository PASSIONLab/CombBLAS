from pcb_function_sm import *
from pcb_operator_convert import *

from asp.jit import asp_module
from asp.codegen import cpp_ast
import kdt

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


    def get_function(self, types=["double", "double", "double"]):
        # see if the result is cached
        if not hasattr(self, '_func_cache'):
            self._func_cache = {}
            
        if str(types) in self._func_cache:
            return self._func_cache[str(types)]

        try:
            # create semantic model
            intypes = types
            import ast, inspect
            from pcb_function_frontend import *
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

            #FIXME: pass correct types, or try all types, or do SOMETHING that's smarter than this hardwired crap
            self.mod.add_function("myfunc", PcbOperatorConvert().convert(sm, types=types))
            self.mod.add_function("get_function", self.gen_get_function(types=types))

            kdt.p_debug(self.mod.generate())
            ret = self.mod.get_function()
            ret.setCallback(self)
            
            # cache the result
            self._func_cache[str(types)] = ret
            return ret
        except:
            kdt.p("WARNING: Specialization failed, proceeding with pure Python.")
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



    def get_function(self, types=["double", "double"]):
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

            kdt.p_debug(types)
            #FIXME: pass correct types, or try all types, or do SOMETHING that's smarter than this hardwired crap
            self.mod.add_function("myfunc", PcbOperatorConvert().convert(sm, types=types))
            self.mod.add_function("get_function", self.gen_get_function(types=types))

            kdt.p_debug(self.mod.generate())
            ret = self.mod.get_function()
            ret.setCallback(self)

            # cache the result
            self._func_cache[str(types)] = ret
            return ret

        except:
            kdt.p("WARNING: Specialization failed, proceeding with pure Python.")
            return self

    def get_function_PDO(self, types=[[None, "Obj1"], [None, "Obj1"]]):
        # see if the result is cached
        if not hasattr(self, '_func_cache'):
            self._func_cache = {}

        if str(types) in self._func_cache:
            return self._func_cache[str(types)]

        # parse out the types
        PDO_types = []
        PDO_type_names = []
        CPP_types = []
        for t in types:
            PDO_types.append(t[0])
            PDO_type_names.append(t[0].__name__)
            CPP_types.append(t[1])

        try:

            # create semantic model
            import ast, inspect
            from pcb_function_frontend import PcbUnaryFunctionFrontEnd, PcbBinaryFunctionFrontEnd
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


            kdt.p_debug("types:")
            kdt.p_debug(PDO_types)
            kdt.p_debug(PDO_type_names)
            kdt.p_debug(CPP_types)

            import asp.codegen.ast_tools as ast_tools
            import asp.codegen.cpp_ast as cpp_ast
            
            converted = PcbOperatorConvert().convert(sm, types=CPP_types)
            
            # define the type structs
            self.mod.add_to_module(PDO_types[0].get_c())
            
            # rename argument
            arg1_name = converted.fdecl.arg_decls[0].subdecl.name.name
            converted.fdecl.arg_decls[0].subdecl.name.name += "_in"

            # declare buffer
            arg1_PDO_ref = cpp_ast.Reference(cpp_ast.Value("const %s" % PDO_type_names[1], cpp_ast.CName(arg1_name)))
            arg1_get_buffer = cpp_ast.Call("%s_in.getConstDataPtr"%(arg1_name), '')
            arg1_typecast = cpp_ast.TypeCast(cpp_ast.Pointer(cpp_ast.Value("const %s" % PDO_type_names[1], '')), arg1_get_buffer)
            arg1_dereference = cpp_ast.Dereference(arg1_typecast)

            converted.body.contents.insert(0, cpp_ast.Assign(arg1_PDO_ref, arg1_dereference))
            
            # convert return statement to produce a CPP object of the return type
            ret_stmt = converted.body.contents[-1]
            retval = ret_stmt.retval
            ret_stmt.retval = cpp_ast.Call(CPP_types[0], ["&%s"%(str(retval)), "sizeof(%s)"%(str(retval))])
            # TODO: retval's type clearly needs to be checked!
            
            print converted

            #import sys
            #sys.exit(2)
            
            #FIXME: pass correct types, or try all types, or do SOMETHING that's smarter than this hardwired crap
            self.mod.add_function("myfunc", converted)
            #self.mod.add_function("myfunc", PcbOperatorConvert().convert(sm, types=CPP_types))
            self.mod.add_function("get_function", self.gen_get_function(types=CPP_types))

            kdt.p_debug(self.mod.generate())
            #return self

            ret = self.mod.get_function()
            ret.setCallback(self)

            # cache the result
            self._func_cache[str(types)] = ret
            return ret

        except Exception, ex:
            print ex.message
            kdt.p("WARNING: Specialization failed, proceeding with pure Python.")
            return self
