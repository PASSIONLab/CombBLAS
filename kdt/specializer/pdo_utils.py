def get_CPP_types(types):
    CPP_types = []
    for t in types:
        if isinstance(t, list):
            CPP_types.append(t[1])
        else:
            CPP_types.append(t)
    return CPP_types

def add_PDO_stubs(module, converted, types):

    import asp.codegen.ast_tools as ast_tools
    import asp.codegen.cpp_ast as cpp_ast
    import ctypes

    # parse out the types
    PDO_types = []
    PDO_type_names = []
    PDO_sizeof = []
    CPP_types = []
    for t in types:
        PDO_types.append(t[0])
        if t[0] is None:
        	# this is a basic type like 'bool' or 'double'
            PDO_type_names.append(t[0])
            PDO_sizeof.append(-1)
        else:
            PDO_type_names.append(t[0].__name__)
            PDO_sizeof.append(ctypes.sizeof(t[0]))
        CPP_types.append(t[1])
    
    #kdt.p_debug("types:")
    #kdt.p_debug(PDO_types)
    #kdt.p_debug(PDO_type_names)
    #kdt.p_debug(CPP_types)

    # define the type structs
    types_added_to_mod = []
    for t in PDO_types:
        if not t in types_added_to_mod and t is not None:
            module.add_to_module(t.get_c())
            types_added_to_mod.append(t)

    # shim the argument(s)
    num_args = len(PDO_types) - 1
    for arg in range(num_args):
        if PDO_types[arg+1] is not None:
            # rename argument(s)
            arg1_name = converted.fdecl.arg_decls[arg].subdecl.name.name
            converted.fdecl.arg_decls[arg].subdecl.name.name += "_in"

            # declare buffer
            arg1_PDO_ref = cpp_ast.Reference(cpp_ast.Value("const %s" % PDO_type_names[arg+1], cpp_ast.CName(arg1_name)))
            arg1_get_buffer = cpp_ast.Call("%s_in.getConstDataPtr"%(arg1_name), '')
            arg1_typecast = cpp_ast.TypeCast(cpp_ast.Pointer(cpp_ast.Value("const %s" % PDO_type_names[arg+1], '')), arg1_get_buffer)
            arg1_dereference = cpp_ast.Dereference(arg1_typecast)

            converted.body.contents.insert(arg, cpp_ast.Assign(arg1_PDO_ref, arg1_dereference))

    # convert return statement to produce a CPP object of the return type
    if PDO_types[0] is not None:
        ret_stmt = converted.body.contents[-1]
        retval = ret_stmt.retval
        ret_stmt.retval = cpp_ast.Call(CPP_types[0], ["&%s"%(str(retval)), "%d"%(PDO_sizeof[0])])
        # TODO: retval's type clearly needs to be checked!

    #print converted
