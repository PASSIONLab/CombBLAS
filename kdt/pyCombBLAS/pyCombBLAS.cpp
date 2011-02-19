#include "pyCombBLAS.h"

////////////////// OPERATORS

pySpParVec EWiseMult(const pySpParVec& a, const pySpParVec& b, bool exclude)
{
	pySpParVec ret;
	//ret->v = ::EWiseMult(a.v, b.v, exclude);
	cout << "EWiseMult(sparse, sparse) not implemented!" << endl;
	return ret;
}

pySpParVec EWiseMult(const pySpParVec& a, const pyDenseParVec& b, bool exclude, double zero)
{
	/*
	pySpParVec ret = new pySpParVec();
	FullyDistSpVec<pySpParVec::INDEXTYPE, doubleint> result = EWiseMult(a.v, b.v, exclude, doubleint(zero));
	ret->v.stealFrom(result);
	return ret;
	*/
	return pySpParVec(EWiseMult(a.v, b.v, exclude, doubleint(zero)));
}

void EWiseMult_inplacefirst(pySpParVec& a, const pyDenseParVec& b, bool exclude, double zero)
{
	a.v = EWiseMult(a.v, b.v, exclude, doubleint(zero));
}

////////////////////////// INITALIZATION/FINALIZE

void init_pyCombBLAS_MPI()
{
	//cout << "calling MPI::Init" << endl;
	MPI::Init();
	
	MPI::Datatype type[1] = {MPI::DOUBLE};
	int blocklen[1] = {1};
	MPI::Aint disp[1];
	
	doubleint data;
	disp[0] = (MPI::Get_address(&data.d) - MPI::Get_address(&data));

	doubleint_MPI_datatype = MPI::Datatype::Create_struct(1,blocklen,disp,type);
	doubleint_MPI_datatype.Commit();
}

void finalize()
{
	//cout << "calling MPI::Finalize" << endl;
	MPI::Finalize();
}

bool root()
{
	return MPI::COMM_WORLD.Get_rank() == 0;
}

int _nprocs()
{
	return MPI::COMM_WORLD.Get_size();
}

MPI::Datatype doubleint_MPI_datatype;

template<> MPI::Datatype MPIType< doubleint >( void )
{
	//cout << "returning doubleint MPIType" << endl;
	return doubleint_MPI_datatype;
}; 
