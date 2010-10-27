
%module DiGraph

// This block gets copied verbatim into the header area of the generated wrapper. DiGraph has to
// be defined here somehow. Prefereably we'd #include "DiGraph.h", but that brings in templates which
// cause duplicate definition linker errors. Unless that gets resolved, we just redefine DiGraph
// omitting the templated protected members.

%{
//#include "DiGraph.h"
#define SWIG_FILE_WITH_INIT

class DiGraph {
public:
	static void init();
	static void finalize();

public:
	DiGraph();

public:
	int nedges();
	int nverts();
	
public:	
	void load(const char* filename);
};

%}


// This block gets called when the module is loaded
%init %{
DiGraph::init();
%}


// This class will get a wrapper created
class DiGraph {
public:
	static void init();
	static void finalize();

public:
	DiGraph();

public:
	int nedges();
	int nverts();
	
public:	
	void load(const char* filename);

};

// It's possible to have the generated python code also include some custom code.
// This may be a good place to add an atexit() to call mpi finalize.
%pragma(python) code="import atexit"
%pragma(python) code="atexit.register(DiGraph.finalize())"
