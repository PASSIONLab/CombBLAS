from pcb_function_sm import *
from pcb_operator_convert import *

from asp.jit import asp_module
from asp.codegen import cpp_ast

class PcbBinaryFunction(object):
    """
    Top-level class for BinaryFunctions.
    """

    def __init__(self, sm, types=["double", "double", "double"]):
        try:
            # create semantic model
            #import ast, inspect
            #from pcb_predicate_frontend import *
            #sm = PcbUnaryPredicateFrontEnd().parse(ast.parse(inspect.getsource(self.__call__).lstrip()), env=vars(self))

            include_files = ["pyOperationsObj.h"]
            self.mod = asp_module.ASPModule(specializer="kdt")

            # add some include directories
            for x in include_files:
                self.mod.add_header(x)
            #self.mod.backends["c++"].toolchain.cc = "mpicxx"
            self.mod.backends["c++"].module.add_to_preamble([cpp_ast.Line("#include <tr1/memory>")])
            self.mod.backends["c++"].module.add_to_preamble([cpp_ast.Line("#define COMBBLAS_TR1")])
            self.mod.backends["c++"].toolchain.cflags = ["-O3", "-fPIC", "-shared", "-DCOMBBLAS_TR1", "-DUSESEJITS", "-DFAST_64BIT_ARITHMETIC"]
            self.mod.backends["c++"].toolchain.cflags.append("-DGRAPH_GENERATOR_SEQ=1 -DMPICH_IGNORE_CXX_SEEK -D__STDC_CONSTANT_MACROS -D__STDC_LIMIT_MACROS -Drestrict= -DNDEBUG=1")

            # Adam's tests
            self.mod.backends["c++"].toolchain.cflags.append("-DSWIG_TYPE_TABLE=pyCombBLAS")
            self.mod.backends["c++"].toolchain.cflags.append("-g")

            if "-bundle" in self.mod.backends["c++"].toolchain.ldflags:
                self.mod.backends["c++"].toolchain.ldflags.remove("-bundle")
            self.mod.backends["c++"].toolchain.defines.append("COMBBLAS_TR1")
            # get location of this file & use to include kdt files

            import inspect, os
            this_file = inspect.currentframe().f_code.co_filename
            installDir = os.path.dirname(this_file)
            self.mod.add_library("pycombblas",
                                 [installDir+"/../pyCombBLAS"],
                                 #library_dirs=[installDir+"/../build/lib.linux-x86_64-2.6"])#,
                                 library_dirs=[installDir+"/../../build/lib.macosx-10.8-intel-2.7"])#,
            #libraries=["mpichcxx"])
            #FIXME: pass correct types, or try all types, or do SOMETHING that's smarter than this hardwired crap
            self.mod.add_function("myfunc", PcbOperatorConvert().convert(sm, types=types))
            self.mod.add_function("get_function", self.gen_get_function(types=types))

            print self.mod.generate()

        except:
            print "WARNING: Specialization failed, proceeding with pure Python."
            raise

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
              swig_module_info* module = SWIG_Python_GetModule();

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



    def get_function(self):
        return self.mod.get_function()
        #return self

    def __call__(self, x, y):
        print "CALL?!?!?!"
        return x

class PcbUnaryFunction(object):
    """
    Top-level class for UnaryFunctions.
    """

    def __init__(self, sm, types=["double", "double", "double"]):
        try:
            # create semantic model
            #import ast, inspect
            #from pcb_predicate_frontend import *
            #sm = PcbUnaryPredicateFrontEnd().parse(ast.parse(inspect.getsource(self.__call__).lstrip()), env=vars(self))

            include_files = ["pyOperationsObj.h"]
            self.mod = asp_module.ASPModule(specializer="kdt")

            # add some include directories
            for x in include_files:
                self.mod.add_header(x)
            #self.mod.backends["c++"].toolchain.cc = "mpicxx"
            self.mod.backends["c++"].module.add_to_preamble([cpp_ast.Line("#include <tr1/memory>")])
            self.mod.backends["c++"].module.add_to_preamble([cpp_ast.Line("#define COMBBLAS_TR1")])
            self.mod.backends["c++"].toolchain.cflags = ["-O3", "-fPIC", "-shared", "-DCOMBBLAS_TR1", "-DUSESEJITS", "-DFAST_64BIT_ARITHMETIC"]
            self.mod.backends["c++"].toolchain.cflags.append("-DGRAPH_GENERATOR_SEQ=1 -DMPICH_IGNORE_CXX_SEEK -D__STDC_CONSTANT_MACROS -D__STDC_LIMIT_MACROS -Drestrict= -DNDEBUG=1")
            if "-bundle" in self.mod.backends["c++"].toolchain.ldflags:
                self.mod.backends["c++"].toolchain.ldflags.remove("-bundle")
            self.mod.backends["c++"].toolchain.defines.append("COMBBLAS_TR1")

            # Adam's tests
            self.mod.backends["c++"].toolchain.cflags.append("-DSWIG_TYPE_TABLE=pyCombBLAS")
            self.mod.backends["c++"].toolchain.cflags.append("-g")

            # get location of this file & use to include kdt files
            import inspect, os
            this_file = inspect.currentframe().f_code.co_filename
            installDir = os.path.dirname(this_file)
            self.mod.add_library("pycombblas",
                                 [installDir+"/../pyCombBLAS"],
                                 library_dirs=[installDir+"/../../build/lib.linux-x86_64-2.7/kdt"],
                                 libraries=["pyCombBLAS"])
            #libraries=["mpichcxx"])
            #FIXME: pass correct types, or try all types, or do SOMETHING that's smarter than this hardwired crap
            self.mod.add_function("myfunc", PcbOperatorConvert().convert(sm, types=types))
            self.mod.add_function("get_function", self.gen_get_function(types=types))

            print self.mod.generate()

        except:
            print "WARNING: Specialization failed, proceeding with pure Python."
            raise

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
              swig_module_info* module = SWIG_Python_GetModule();

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



    def get_function(self):
        return self.mod.get_function()
        #return self

    def __call__(self, x):
        print "CALL?!?!?!"
        return x
