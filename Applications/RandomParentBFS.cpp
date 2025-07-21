#define DETERMINISTIC
#include "CombBLAS/CombBLAS.h"
#include <mpi.h>
#include <sys/time.h> 
#include <iostream>
#include <functional>
#include <algorithm>
#include <vector>
#include <string>
#include <sstream>
#ifdef THREADED
	#ifndef _OPENMP
	#define _OPENMP
	#endif
#endif

#ifdef _OPENMP
	#include <omp.h>
#endif


double cblas_alltoalltime;
double cblas_allgathertime;
double cblas_mergeconttime;
double cblas_transvectime;
double cblas_localspmvtime;
#ifdef _OPENMP
int cblas_splits = omp_get_max_threads(); 
#else
int cblas_splits = 1;
#endif

#define ITERS 16
#define EDGEFACTOR 16
using namespace std;
using namespace combblas;


MTRand GlobalMT(123); // for reproducable result


template <typename PARMAT>
void Symmetricize(PARMAT & A)
{
	// boolean addition is practically a "logical or"
	// therefore this doesn't destruct any links
	PARMAT AT = A;
	AT.Transpose();
	A += AT;
}



struct ParentType
{
public:
	ParentType(){parent=-1; p = 0;};
	ParentType(int64_t x){parent=(x); p = 0;};
    friend ostream& operator<<(ostream& os, const ParentType & vertex ){os << "Parent=" << vertex.parent << " p=" << vertex.p; return os;};
    //private:
    int64_t parent;
    float p;
    
};






struct Edge_randomizer : public std::unary_function<std::pair<bool, float>, std::pair<bool, float>>
{
    const std::pair<bool, float> operator()(const std::pair<bool, float> & x) const
    {
        float edgeRand = static_cast<float>(rand());	// random range(0,1)
        return std::pair<bool, float>(x.first, edgeRand);
    }
};



static void MPI_randuniq(void * invec, void * inoutvec, int * len, MPI_Datatype *datatype)
{
    RandReduce<int64_t> RR;
    int64_t * inveccast = (int64_t *) invec;
    int64_t * inoutveccast = (int64_t *) inoutvec;
    for (int i=0; i<*len; i++ )
        inoutveccast[i] = RR(inveccast[i], inoutveccast[i]);
}


struct SelectRandSRing
{
    static MPI_Op MPI_BFSRAND;
	typedef int64_t T_promote;
	static ParentType id(){ return ParentType(); };
	static bool returnedSAID() { return false; }
	//static MPI_Op mpi_op() { return MPI_MAX; }; // do we need this?
    
	static ParentType add(const ParentType & arg1, const ParentType & arg2)
	{
        //cout << arg1 << " ;;; " << arg2 << endl;
        if(arg1.p < arg2.p) return arg1;
        else return arg2;
	}
    
	static ParentType multiply(const T_promote & arg1, const ParentType & arg2)
	{
        ParentType temp;
        temp.parent = arg2.parent;
        temp.p = GlobalMT.rand();
		return temp;
	}
    
     static void axpy(T_promote a, const ParentType & x, ParentType & y)
     {
         y = add(y, multiply(a, x));
     }
};



// This one is used for BFS iteration
struct SelectMinSRing1
{
	typedef int64_t T_promote;
	static T_promote id(){ return -1; };
	static bool returnedSAID() { return false; }
	//static MPI_Op mpi_op() { return MPI_MAX; };
    
	static T_promote add(const T_promote & arg1, const T_promote & arg2)
	{
        cout << arg1 << " a " << arg2 << endl;
		return std::max(arg1, arg2);
	}
    
	static T_promote multiply(const bool & arg1, const T_promote & arg2)
	{
        cout << arg1 << " m " << arg2 << endl;
		return arg2;
	}
    /*
	static void axpy(bool a, const T_promote & x, T_promote & y)
	{
		y = std::max(y, x);
	}*/
};

typedef SpParMat < int64_t, bool, SpDCCols<int64_t,bool> > PSpMat_Bool;
typedef SpParMat < int64_t, bool, SpDCCols<int32_t,bool> > PSpMat_s32p64;
typedef SpParMat < int64_t, int64_t, SpDCCols<int64_t,int64_t> > PSpMat_Int64;


void RandomParentBFS(PSpMat_Bool & Aeff)
{
    
    int nprocs, myrank;
	MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    
    
    FullyDistSpVec<int64_t, ParentType> fringe(Aeff.getcommgrid(), Aeff.getncol());
    
    fringe.SetElement(0, ParentType(0));
    fringe.SetElement(1, ParentType(1));
    fringe.SetElement(5, ParentType(5));
    fringe.SetElement(6, ParentType(6));
    fringe.SetElement(7, ParentType(7));
    
    PSpMat_Int64  A = Aeff;
    //A.PrintInfo();
    SpMV<SelectRandSRing>(A, fringe, fringe, false);
    fringe.DebugPrint();
}



int main(int argc, char* argv[])
{
	int nprocs, myrank;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
	if(argc < 2)
	{
		if(myrank == 0)
		{
			cout << "Usage: ./rpbfs <Scale>" << endl;
			cout << "Example: mpirun -np 4 ./spbfs 20" << endl;
		}
		MPI_Finalize();
		return -1;
	}		
	{
    
		// Declare objects
		PSpMat_Bool A;
		FullyDistVec<int64_t, int64_t> nonisov;	// id's of non-isolated (connected) vertices
		unsigned scale;

		scale = static_cast<unsigned>(atoi(argv[1]));
        double initiator[4] = {.57, .19, .19, .05};

        
        DistEdgeList<int64_t> * DEL = new DistEdgeList<int64_t>();
        DEL->GenGraph500Data(initiator, scale, EDGEFACTOR, true, true );
        MPI_Barrier(MPI_COMM_WORLD);
        
        
        PSpMat_Bool * ABool = new PSpMat_Bool(*DEL, false);
        delete DEL;
        int64_t removed  = ABool->RemoveLoops();
        ABool->PrintInfo();
        
        FullyDistVec<int64_t, int64_t> * ColSums = new FullyDistVec<int64_t, int64_t>(ABool->getcommgrid());
        FullyDistVec<int64_t, int64_t> * RowSums = new FullyDistVec<int64_t, int64_t>(ABool->getcommgrid());
        
        ABool->Reduce(*ColSums, Column, plus<int64_t>(), static_cast<int64_t>(0));
        ABool->Reduce(*RowSums, Row, plus<int64_t>(), static_cast<int64_t>(0));
        ColSums->EWiseApply(*RowSums, plus<int64_t>());
        delete RowSums;
        nonisov = ColSums->FindInds([](int64_t val){return val > 0;});
        delete ColSums;
        nonisov.RandPerm();	// so that A(v,v) is load-balanced (both memory and time wise)
#ifndef NOPERMUTE
        ABool->operator()(nonisov, nonisov, true);	// in-place permute to save memory
#endif

        
        RandomParentBFS(*ABool);
        
        
	}
	MPI_Finalize();
	return 0;
}









