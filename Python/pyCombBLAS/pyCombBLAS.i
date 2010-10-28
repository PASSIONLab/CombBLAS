
%module pyCombBLAS

// This block gets copied verbatim into the header area of the generated wrapper. DiGraph has to
// be defined here somehow. Prefereably we'd #include "DiGraph.h", but that brings in templates which
// cause duplicate definition linker errors. Unless that gets resolved, we just redefine DiGraph
// omitting the templated protected members.

%{
#define SWIG_FILE_WITH_INIT

#include "DiGraph.h"
#include "SpVectList.h"
#include "VectList.h"
%}


// This block gets called when the module is loaded
%init %{
init_pyCombBLAS_MPI();
%}

// It's possible to have the generated python code also include some custom code.
// This may be a good place to add an atexit() to call mpi finalize.
%pragma(python) code="import atexit"
%pragma(python) code="atexit.register(DiGraph.finalize())"


// This class will get a wrapper created
class DiGraph {
public:
	DiGraph();

public:
	int nedges();
	int nverts();
	
public:	
	void load(const char* filename);

public:
	void SpMV_SelMax(const SpVectList& v);

};

class SpVectList {

public:
	SpVectList();

public:
	int length();
	
public:	
	void load(const char* filename);
	
};

class VectList {
public:
	VectList();

public:
	int length() const;
	
public:	
	void load(const char* filename);
	
};

//void init();
void finalize();
