#ifndef PCB_OBJ_H
#define PCB_OBJ_H


/* Note to users:
 It's ok to change the members of these two structures.
*/
//INTERFACE_INCLUDE_BEGIN
class EDGETYPE {
public:
	int64_t i; // reseserved for internal use
public:
	double weight;
	int type;
	
//INTERFACE_INCLUDE_END
	operator int64_t() const { return i; }
	EDGETYPE(int64_t val): i(val) {}
	EDGETYPE(): i(-1) {}
//INTERFACE_INCLUDE_BEGIN
};

class VERTEXTYPE {
public:
	int64_t i; // reserved for internal use
public:
	double weight;
	int type;

	char *__repr__() {
		static char temp[256];
		sprintf(temp,"[ %lf, %d ]", weight,type);
		return &temp[0];
	}

//INTERFACE_INCLUDE_END
	operator int64_t() const { return i; }
	VERTEXTYPE(int64_t val): i(val) {}
	VERTEXTYPE(): i(-1) {}
	
	bool operator==(const VERTEXTYPE& other)
	{
		return this == &other;
	}
//INTERFACE_INCLUDE_BEGIN
};
//INTERFACE_INCLUDE_END

// From CombBLAS/promote.h:
DECLARE_PROMOTE(EDGETYPE, EDGETYPE, EDGETYPE)
DECLARE_PROMOTE(VERTEXTYPE, VERTEXTYPE, VERTEXTYPE)

// From CombBLAS/MPIType.h
#ifndef PYCOMBBLAS_CPP
// forward declarations
extern MPI::Datatype EDGETYPE_MPI_datatype;
extern MPI::Datatype VERTEXTYPE_MPI_datatype;

//extern "C" {
void create_EDGE_and_VERTEX_MPI_Datatypes();
//}

template<> MPI::Datatype MPIType< EDGETYPE >( void );
template<> MPI::Datatype MPIType< VERTEXTYPE >( void );

#else
// definitions
MPI::Datatype EDGETYPE_MPI_datatype;
MPI::Datatype VERTEXTYPE_MPI_datatype;

// called from init_pyCombBLAS_MPI() in pyCombBLAS.cpp
//extern "C" {
void create_EDGE_and_VERTEX_MPI_Datatypes()
{
	EDGETYPE_MPI_datatype = MPI::CHAR.Create_contiguous(sizeof(EDGETYPE));
	EDGETYPE_MPI_datatype.Commit();

	VERTEXTYPE_MPI_datatype = MPI::CHAR.Create_contiguous(sizeof(VERTEXTYPE));
	VERTEXTYPE_MPI_datatype.Commit();
}
//}

template<> MPI::Datatype MPIType< EDGETYPE >( void )
{
	return EDGETYPE_MPI_datatype;
}
template<> MPI::Datatype MPIType< VERTEXTYPE >( void )
{
	return VERTEXTYPE_MPI_datatype;
}

#endif




#endif