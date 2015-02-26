#define DETERMINISTIC
#include "../CombBLAS.h"
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

#define EDGEFACTOR 16  /// changed to 8
#define GREEDY 1
#define KARP_SIPSER 2
#define KS_GREEDY 3
using namespace std;


MTRand GlobalMT(123); // for reproducable result



struct VertexType
{
public:
    VertexType(){parent=-1; degree = 0; prob=0;};
    VertexType(int64_t pa){parent=pa; degree=0; prob=0;}; // this constructor is called when we assign vertextype=number. Called from ApplyInd function
    VertexType(int64_t pa, int64_t deg){parent=pa; degree = deg; prob=0;};
    VertexType(int64_t pa, int64_t deg, int64_t p){parent=pa; degree = deg; prob=p;};
    
    friend bool operator==(const VertexType & vtx1, const VertexType & vtx2 ){return vtx1.parent==vtx2.parent;};
    friend ostream& operator<<(ostream& os, const VertexType & vertex ){os << "(" << vertex.parent << "," << vertex.degree << "," << vertex.prob << ")"; return os;};
    //private:
    int64_t parent;
    int64_t degree;
    int64_t prob;
};


// for degree
struct SelectPlusSRing
{
    typedef int64_t T_promote;
    static T_promote id(){ return -1; };
    static bool returnedSAID() { return false; }
    //static MPI_Op mpi_op() { return MPI_MAX; };
    
    static T_promote add(const T_promote & arg1, const T_promote & arg2)
    {
        return std::plus<T_promote>()(arg1, arg2);
    }
    
    static T_promote multiply(const T_promote & arg1, const T_promote & arg2)
    {
        return static_cast<int64_t> (1);
        //return arg1;
    }
    
    static void axpy(T_promote a, const T_promote & x, T_promote & y)
    {
        y = add(y, multiply(a, x));
    }
};


struct SelectMinSRing1
{
	typedef int64_t T_promote;
	static T_promote id(){ return -1; };
	static bool returnedSAID() { return false; }
	//static MPI_Op mpi_op() { return MPI_MAX; };
    
	static T_promote add(const T_promote & arg1, const T_promote & arg2)
	{
		return std::min(arg1, arg2);
	}
    
	static T_promote multiply(const bool & arg1, const T_promote & arg2)
	{
		return arg2;
	}
    
    static void axpy(bool a, const T_promote & x, T_promote & y)
    {
        y = add(y, multiply(a, x));
    }
};


struct SelectMaxSRing1
{
    typedef int64_t T_promote;
    static T_promote id(){ return -1; };
    static bool returnedSAID() { return false; }
    //static MPI_Op mpi_op() { return MPI_MAX; };
    
    static T_promote add(const T_promote & arg1, const T_promote & arg2)
    {
        return std::max(arg1, arg2);
    }
    
    static T_promote multiply(const bool & arg1, const T_promote & arg2)
    {
        return arg2;
    }
    
    static void axpy(bool a, const T_promote & x, T_promote & y)
    {
        y = add(y, multiply(a, x));
    }
};



struct GreedySR
{
    typedef int64_t T_promote;
    static VertexType id(){ return VertexType(); };
    static bool returnedSAID() { return false; }
    //static MPI_Op mpi_op() { return MPI_MIN; };
    
    static VertexType add(const VertexType & arg1, const VertexType & arg2)
    {
        if(arg1.parent < arg2.parent) return arg1;
        else return arg2;
    }
    static VertexType multiply(const T_promote & arg1, const VertexType & arg2)
    {
        return VertexType(arg2.parent, arg2.degree, arg1);
    }
    
    
    static void axpy(T_promote a, const VertexType & x, VertexType & y)
    {
        y = add(y, multiply(a, x));
    }
};



struct GreedyRandSR
{
    typedef int64_t T_promote;
    static VertexType id(){ return VertexType(); };
    static bool returnedSAID() { return false; }
    //static MPI_Op mpi_op() { return MPI_MIN; };
    
    static VertexType add(const VertexType & arg1, const VertexType & arg2)
    {
        //if(arg1.prob > arg2.prob)
        //if((arg1.degree/arg1.prob) < (arg2.degree/arg2.prob))
        if((arg1.degree) < (arg2.degree))
            return arg1;
        else return arg2;
    }
    static VertexType multiply(const T_promote & arg1, const VertexType & arg2)
    {
        return VertexType(arg2.parent, arg2.degree, arg1);
    }
    
    
    static void axpy(T_promote a, const VertexType & x, VertexType & y)
    {
        y = add(y, multiply(a, x));
    }
};




struct KSGreedySR
{
    typedef int64_t T_promote;
    static VertexType id(){ return VertexType(); };
    static bool returnedSAID() { return false; }
    //static MPI_Op mpi_op() { return MPI_MIN; };
    
    static VertexType add(const VertexType & arg1, const VertexType & arg2)
    {
        if(arg1.degree == 1) return arg1;
        else if(arg2.degree == 1) return arg2;
        else if(arg1.parent < arg2.parent)
            return arg1;
        else
            return arg2;
    }
    static VertexType multiply(const T_promote & arg1, const VertexType & arg2)
    {
        return arg2;
    }
    
    
    static void axpy(T_promote a, const VertexType & x, VertexType & y)
    {
        y = add(y, multiply(a, x));
    }
};



struct KSGreedyRandSR
{
    typedef int64_t T_promote;
    static VertexType id(){ return VertexType(); };
    static bool returnedSAID() { return false; }
    //static MPI_Op mpi_op() { return MPI_MIN; };
    
    static VertexType add(const VertexType & arg1, const VertexType & arg2)
    {
        if(arg1.degree == 1) return arg1;
        else if(arg2.degree == 1) return arg2;
        if((arg1.degree) < (arg2.degree)) return arg1;
        //else if(arg1.prob > arg2.prob) return arg1;
        else return arg2;
    }
    static VertexType multiply(const T_promote & arg1, const VertexType & arg2)
    {
        return VertexType(arg2.parent, arg2.degree, arg1);
    }
    
    static void axpy(T_promote a, const VertexType & x, VertexType & y)
    {
        y = add(y, multiply(a, x));
    }
};


// not good at all
struct SelectMinDegSR
{
    typedef int64_t T_promote;
    static VertexType id(){ return VertexType(); };
    static bool returnedSAID() { return false; }
    //static MPI_Op mpi_op() { return MPI_MIN; };
    
    static VertexType add(const VertexType & arg1, const VertexType & arg2)
    {
        if(arg1.degree < arg2.degree)
            return arg1;
        else
            return arg2;
    }
    
    static VertexType multiply(const T_promote & arg1, const VertexType & arg2)
    {
        return arg2;
    }
    
    
    static void axpy(T_promote a, const VertexType & x, VertexType & y)
    {
        y = add(y, multiply(a, x));
    }
};





template<typename T>
struct unmatched : public std::unary_function<T, bool>
{
    bool operator()(const T& x) const
    {
        return (x==-1);
    }
};



template <typename PARMAT>
void Symmetricize(PARMAT & A)
{
	// boolean addition is practically a "logical or"
	// therefore this doesn't destruct any links
	PARMAT AT = A;
	AT.Transpose();
	A += AT;
}


typedef SpParMat < int64_t, bool, SpDCCols<int64_t,bool> > PSpMat_Bool;
typedef SpParMat < int64_t, bool, SpDCCols<int32_t,bool> > PSpMat_s32p64;
typedef SpParMat < int64_t, int64_t, SpDCCols<int64_t,int64_t> > PSpMat_Int64;
void greedyMatching_old(PSpMat_Int64 & Aeff);
void greedyMatching(PSpMat_Int64 & A, PSpMat_Int64 & AT, FullyDistVec<int64_t, int64_t>& mateRow2Col,
                    FullyDistVec<int64_t, int64_t>& mateCol2Row, bool removeIsolate);
void greedyMatching1(PSpMat_Int64 & A, PSpMat_Int64 & AT, FullyDistVec<int64_t, int64_t>& mateRow2Col,
                    FullyDistVec<int64_t, int64_t>& mateCol2Row, bool removeIsolate);

void KS(PSpMat_Int64 & A, PSpMat_Int64 & AT, FullyDistVec<int64_t, int64_t>& mateRow2Col,
        FullyDistVec<int64_t, int64_t>& mateCol2Row);
void DMD(PSpMat_Int64 & A, PSpMat_Int64 & AT, FullyDistVec<int64_t, int64_t>& mateRow2Col,
         FullyDistVec<int64_t, int64_t>& mateCol2Row);
void hybrid(PSpMat_Int64 & A, PSpMat_Int64 & AT, FullyDistVec<int64_t, int64_t>& mateRow2Col,
            FullyDistVec<int64_t, int64_t>& mateCol2Row, int type, bool rand);
template <class IT, class NT>
bool isMaximalmatching(PSpMat_Int64 & A, FullyDistVec<IT,NT> & mateRow2Col, FullyDistVec<IT,NT> & mateCol2Row);
template <class IT, class NT>
bool isMatching(FullyDistVec<IT,NT> & mateCol2Row, FullyDistVec<IT,NT> & mateRow2Col);



/*
 Remove isolated vertices and purmute
 */
void removeIsolated(PSpMat_Int64 & A, bool perm)
{
    
    int nprocs, myrank;
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    
    
    FullyDistVec<int64_t, int64_t> * ColSums = new FullyDistVec<int64_t, int64_t>(A.getcommgrid());
    FullyDistVec<int64_t, int64_t> * RowSums = new FullyDistVec<int64_t, int64_t>(A.getcommgrid());
    FullyDistVec<int64_t, int64_t> nonisoRowV;	// id's of non-isolated (connected) Row vertices
    FullyDistVec<int64_t, int64_t> nonisoColV;	// id's of non-isolated (connected) Col vertices
    FullyDistVec<int64_t, int64_t> nonisov;	// id's of non-isolated (connected) vertices
    
    A.Reduce(*ColSums, Column, plus<int64_t>(), static_cast<int64_t>(0));
    A.Reduce(*RowSums, Row, plus<int64_t>(), static_cast<int64_t>(0));
    
    // this steps for general graph
    /*
     ColSums->EWiseApply(*RowSums, plus<int64_t>()); not needed for bipartite graph
     nonisov = ColSums->FindInds(bind2nd(greater<int64_t>(), 0));
     nonisov.RandPerm();	// so that A(v,v) is load-balanced (both memory and time wise)
     A.operator()(nonisov, nonisov, true);	// in-place permute to save memory
    */

    // this steps for bipartite graph
    nonisoColV = ColSums->FindInds(bind2nd(greater<int64_t>(), 0));
    nonisoRowV = RowSums->FindInds(bind2nd(greater<int64_t>(), 0));
    delete ColSums;
    delete RowSums;
    
    if(perm)
    {
        nonisoColV.RandPerm();
        nonisoRowV.RandPerm();
    }
    
    
    int64_t nrows1=A.getnrow(), ncols1=A.getncol(), nnz1 = A.getnnz();
    double avgDeg1 = (double) nnz1/(nrows1+ncols1);

    
    A.operator()(nonisoRowV, nonisoColV, true);
   
    int64_t nrows2=A.getnrow(), ncols2=A.getncol(), nnz2 = A.getnnz();
    double avgDeg2 = (double) nnz2/(nrows2+ncols2);
    
    
    if(myrank == 0)
    {
        cout << "ncol nrows  nedges deg \n";
        cout << nrows1 << " " << ncols1 << " " << nnz1 << " " << avgDeg1 << " \n";
        cout << nrows2 << " " << ncols2 << " " << nnz2 << " " << avgDeg2 << " \n";
    }
    
    MPI_Barrier(MPI_COMM_WORLD);

    
}


/*
 Remove isolated vertices and purmute
 */
void graphStats(PSpMat_Int64 & A)
{
    int nprocs, myrank;
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    int64_t nrows=A.getnrow(), ncols=A.getncol(), nnz = A.getnnz();
    
    
    FullyDistVec<int64_t, int64_t> * ColSums = new FullyDistVec<int64_t, int64_t>(A.getcommgrid());
    FullyDistVec<int64_t, int64_t> * RowSums = new FullyDistVec<int64_t, int64_t>(A.getcommgrid());
    A.Reduce(*ColSums, Column, plus<int64_t>(), static_cast<int64_t>(0));
    A.Reduce(*RowSums, Row, plus<int64_t>(), static_cast<int64_t>(0));
    
    
    int64_t isolatedCols = ColSums->Count([](int64_t deg){return deg==0;});
    int64_t isolatedRows = RowSums->Count([](int64_t deg){return deg==0;});
    
    double fracIsolated = 100 * ((double)isolatedRows+isolatedCols)/(nrows+ncols);
    double avgDeg = (double) nnz/(nrows+ncols);
    delete ColSums;
    delete RowSums;
    
    
    if(myrank == 0)
    {
        cout << "nproc nrows  ncols %isolated isorow isocol nedges deg \n";
        cout << nprocs << " " << nrows << " " << ncols << " " <<  fracIsolated << " " << isolatedRows << " " << isolatedCols << " " <<  nnz << " " << avgDeg << " \n";
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
}



/*
    Randomly permute rows and columns of a matrix (assume bipartite graph)
 */
void RandPermMat(PSpMat_Int64 & A)
{
    FullyDistVec<int64_t, int64_t> I(A.getcommgrid(), A.getnrow(), (int64_t) -1);
    FullyDistVec<int64_t, int64_t> J(A.getcommgrid(), A.getncol(), (int64_t) -1);
    I.ApplyInd([](int64_t val, int64_t idx){return idx;});
    J.ApplyInd([](int64_t val, int64_t idx){return idx;});
    
    I.RandPerm();
    J.RandPerm();
    A.operator()(I, J, true);
   
}




int main(int argc, char* argv[])
{
	int nprocs, myrank;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
	if(argc < 3)
	{
		if(myrank == 0)
		{
			cout << "Usage: ./BPMaximalMatching <rmat|er|input> <scale|filename> <w|uw> " << endl;
			cout << "Example: mpirun -np 4 ./BPMaximalMatching rmat 20" << endl;
            cout << "Example: mpirun -np 4 ./BPMaximalMatching er 20" << endl;
            cout << "Example: mpirun -np 4 ./BPMaximalMatching input a.mtx uw sym" << endl;
            
		}
		MPI_Finalize();
		return -1;
	}		
	{
        PSpMat_Bool * ABool;
        
        if(string(argv[1]) == string("input")) // input option
        {
            ABool = new PSpMat_Bool();
            if(argc>=4 && string(argv[3]) == string("uw"))
                ABool->ReadDistribute(string(argv[2]), 0, true);	// unweighted
            else
                ABool->ReadDistribute(string(argv[2]), 0, false);	// weighted
            if(argc>=5 && string(argv[4]) == string("sym"))
                Symmetricize(*ABool);
            SpParHelper::Print("Read input\n");

        }
        else if(string(argv[1]) == string("rmat"))
        {
            unsigned scale;
            scale = static_cast<unsigned>(atoi(argv[2]));
            double initiator[4] = {.57, .19, .19, .05};
            DistEdgeList<int64_t> * DEL = new DistEdgeList<int64_t>();
            DEL->GenGraph500Data(initiator, scale, EDGEFACTOR, true, false );
            MPI_Barrier(MPI_COMM_WORLD);
            
            ABool = new PSpMat_Bool(*DEL, false);
            Symmetricize(*ABool);
            delete DEL;
        }
        else if(string(argv[1]) == string("er"))
        {
            unsigned scale;
            scale = static_cast<unsigned>(atoi(argv[2]));
            double initiator[4] = {.25, .25, .25, .25};
            DistEdgeList<int64_t> * DEL = new DistEdgeList<int64_t>();
            DEL->GenGraph500Data(initiator, scale, EDGEFACTOR, true, false );
            MPI_Barrier(MPI_COMM_WORLD);
            
            ABool = new PSpMat_Bool(*DEL, false);
            Symmetricize(*ABool);
            delete DEL;
        }
        else
        {
            SpParHelper::Print("Unknown input option\n");
            MPI_Finalize();
            return -1;
        }
        
        //int64_t removed  = ABool->RemoveLoops(); // loop means an edges (i,i+NU) in a bipartite graph
        PSpMat_Int64  A = *ABool;
        
        

        removeIsolated(A, true);
        
        
        PSpMat_Int64 AT= A;
        AT.Transpose();
        
        // always create tall matrix
        
        if(A.getnrow() < A.getncol() )
        {
            PSpMat_Int64 t= A;
            A= AT;
            AT = t;
            //graphStats(A);
            SpParHelper::Print("row-column swaped\n");
        }
        
        int64_t nrows=A.getnrow(), ncols=A.getncol(), nnz = A.getnnz();
        
        double tstart = MPI_Wtime();
        
        if(myrank==0) cout << "\n*********** Greedy Matching ***********\n";
        FullyDistVec<int64_t, int64_t> mateRow2Col ( A.getcommgrid(), A.getnrow(), (int64_t) -1);
        FullyDistVec<int64_t, int64_t> mateCol2Row ( A.getcommgrid(), A.getncol(), (int64_t) -1);
        greedyMatching(A, AT, mateRow2Col, mateCol2Row, false);
        MPI_Barrier(MPI_COMM_WORLD);
        double t = MPI_Wtime()-tstart; tstart = MPI_Wtime();
        
        /*
        if(myrank==0) cout << "\n*********** Greedy Matching remove isolated ***********\n";
        FullyDistVec<int64_t, int64_t> mateRow2Col1 ( A.getcommgrid(), A.getnrow(), (int64_t) -1);
        FullyDistVec<int64_t, int64_t> mateCol2Row1 ( A.getcommgrid(), A.getncol(), (int64_t) -1);
        greedyMatching1(A, AT, mateRow2Col1, mateCol2Row1, false);
        MPI_Barrier(MPI_COMM_WORLD);
        double t1 = MPI_Wtime()-tstart; tstart = MPI_Wtime();
        
        
        if(myrank==0) cout << "\n*********** Greedy-Rand ***********\n";
        FullyDistVec<int64_t, int64_t> mateRow2Col5 ( A.getcommgrid(), A.getnrow(), (int64_t) -1);
        FullyDistVec<int64_t, int64_t> mateCol2Row5 ( A.getcommgrid(), A.getncol(), (int64_t) -1);
        hybrid(A, AT, mateRow2Col5, mateCol2Row5, GREEDY, true);
        MPI_Barrier(MPI_COMM_WORLD);
        double t5 = MPI_Wtime()-tstart; tstart = MPI_Wtime();
        */
        
        if(myrank==0) cout << "\n*********** Karp-Sipser ***********\n";
        FullyDistVec<int64_t, int64_t> mateRow2Col2 ( A.getcommgrid(), A.getnrow(), (int64_t) -1);
        FullyDistVec<int64_t, int64_t> mateCol2Row2 ( A.getcommgrid(), A.getncol(), (int64_t) -1);
        KS(A, AT, mateRow2Col2, mateCol2Row2);
        MPI_Barrier(MPI_COMM_WORLD);
        double t2 = MPI_Wtime()-tstart; tstart = MPI_Wtime();
        
        /*
        if(myrank==0) cout << "\n*********** KS-Rand ***********\n";
        FullyDistVec<int64_t, int64_t> mateRow2Col3 ( A.getcommgrid(), A.getnrow(), (int64_t) -1);
        FullyDistVec<int64_t, int64_t> mateCol2Row3 ( A.getcommgrid(), A.getncol(), (int64_t) -1);
        hybrid(A, AT, mateRow2Col3, mateCol2Row3, KARP_SIPSER, true);
        MPI_Barrier(MPI_COMM_WORLD);
        double t3 = MPI_Wtime()-tstart; tstart = MPI_Wtime();
         */
        
        if(myrank==0) cout << "\n*********** DMD ***********\n";
        FullyDistVec<int64_t, int64_t> mateRow2Col4 ( A.getcommgrid(), A.getnrow(), (int64_t) -1);
        FullyDistVec<int64_t, int64_t> mateCol2Row4 ( A.getcommgrid(), A.getncol(), (int64_t) -1);
        DMD(A, AT, mateRow2Col4, mateCol2Row4);
        MPI_Barrier(MPI_COMM_WORLD);
        double t4 = MPI_Wtime()-tstart; tstart = MPI_Wtime();
        
         //hybrid(A, AT, mateRow2Col2, mateCol2Row2, KARP_SIPSER, false);
        
        
        //isMatching(mateCol2Row2, mateRow2Col2); //todo there is a better way to check this
        
        // print summary
        int64_t matched = mateCol2Row.Count([](int64_t mate){return mate!=-1;});
        //int64_t matched1 = mateCol2Row1.Count([](int64_t mate){return mate!=-1;});
        //int64_t matched5 = mateCol2Row5.Count([](int64_t mate){return mate!=-1;});
        int64_t matched2 = mateCol2Row2.Count([](int64_t mate){return mate!=-1;});
        //int64_t matched3 = mateCol2Row3.Count([](int64_t mate){return mate!=-1;});
        int64_t matched4 = mateCol2Row4.Count([](int64_t mate){return mate!=-1;});
        
        
        
       if(myrank==0)
       {
           cout << "matched %cols time\n";
           printf("%lld %lf %lf \n",matched, 100*(double)matched/(ncols), t);
           //printf("%lld %lf %lf \n",matched1, 100*(double)matched1/(ncols), t1);
           //printf("%lld %lf %lf ",matched5, 100*(double)matched5/(ncols), t5);
           printf("%lld %lf %lf \n",matched2, 100*(double)matched2/(ncols), t2);
           //printf("%lld %lf %lf ",matched3, 100*(double)matched3/(ncols), t3);
           printf("%lld %lf %lf \n",matched4, 100*(double)matched4/(ncols),  t4);
       }
	}
	MPI_Finalize();
	return 0;
}




void KS(PSpMat_Int64 & A, PSpMat_Int64 & AT, FullyDistVec<int64_t, int64_t>& mateRow2Col,
                    FullyDistVec<int64_t, int64_t>& mateCol2Row)
{
    
    int nprocs, myrank;
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    
    //unmatched row and column vertices
    FullyDistSpVec<int64_t, int64_t> unmatchedRow(mateRow2Col, [](int64_t mate){return mate==-1;});
    FullyDistSpVec<int64_t, int64_t> unmatchedCol(mateCol2Row, [](int64_t mate){return mate==-1;});
    unmatchedRow.setNumToInd();
    unmatchedCol.setNumToInd();
    
    FullyDistSpVec<int64_t, int64_t> degColSG(A.getcommgrid(), A.getncol());
    FullyDistVec<int64_t, int64_t> degCol(A.getcommgrid());
    
    // update initial degree of unmatched column vertices
    A.Reduce(degCol, Column, plus<int64_t>(), static_cast<int64_t>(0));
    unmatchedCol.Select(degCol, [](int64_t deg){return deg>0;}); // remove degree-0 columns

    //fringe vector to store the result of SpMV
    FullyDistSpVec<int64_t, int64_t> fringeRow(A.getcommgrid(), A.getnrow());
    
    
    int64_t curUnmatchedCol = unmatchedCol.getnnz();
    int64_t curUnmatchedRow = unmatchedRow.getnnz();
    int64_t newlyMatched = 1; // ensure the first pass of the while loop
    int iteration = 0;
    double tStart = MPI_Wtime();
    vector<vector<double> > timing;
    if(myrank == 0)
    {
        cout << "=======================================================\n";
        cout << "@@@@@@ Number of processes: " << nprocs << endl;
        cout << "=======================================================\n";
        cout  << "It   |  UMRow   |  UMCol   |  newlyMatched   |  Time "<< endl;
        cout << "=======================================================\n";
    }
    MPI_Barrier(MPI_COMM_WORLD);
    
    
    while(curUnmatchedCol !=0 && curUnmatchedRow!=0 && newlyMatched != 0 )
    {
        
        // ======================== step1: One step of BFS =========================
        vector<double> times;
        double t1 = MPI_Wtime();
        
        FullyDistSpVec<int64_t, int64_t> deg1Col = unmatchedCol;
        deg1Col.Select(degCol, [](int64_t deg){return deg==1;});
        
        if(deg1Col.getnnz()>9)
            SpMV<SelectMinSRing1>(A, deg1Col, fringeRow, false);
        else
            SpMV<SelectMinSRing1>(A, unmatchedCol, fringeRow, false);
       
        // Remove matched row vertices
        fringeRow.Select(mateRow2Col, [](int64_t mate){return mate==-1;});
        if(myrank == 0){times.push_back(MPI_Wtime()-t1); t1 = MPI_Wtime();}
        // ===========================================================================
        
        // ======================== step2: Update matching  =========================
        FullyDistSpVec<int64_t, int64_t> newMatchedCols = fringeRow.Invert(A.getncol());
        FullyDistSpVec<int64_t, int64_t> newMatchedRows = newMatchedCols.Invert(A.getnrow());
        mateCol2Row.Set(newMatchedCols);
        mateRow2Col.Set(newMatchedRows);
        if(myrank == 0){times.push_back(MPI_Wtime()-t1); t1 = MPI_Wtime();}
        // ===========================================================================
        
        // =============== step3: Update degree of unmatched columns =================
        unmatchedRow.Select(mateRow2Col, [](int64_t mate){return mate==-1;});
        unmatchedCol.Select(mateCol2Row, [](int64_t mate){return mate==-1;});
        
        // update degree
        SpMV< SelectPlusSRing>(AT, newMatchedRows, degColSG, false);  // degree of column vertices to matched rows
        // subtract degree of column vertices
        degCol.EWiseApply(degColSG,
                              [](int64_t old_deg, int64_t new_deg, bool a, bool b){return old_deg-new_deg;},
                              [](int64_t old_deg, int64_t new_deg, bool a, bool b){return true;},
                              false, static_cast<int64_t>(0), false);
        unmatchedCol.Select(degCol, [](int64_t deg){return deg>0;});
        if(myrank == 0){times.push_back(MPI_Wtime()-t1); t1 = MPI_Wtime();}
        // ===========================================================================
        
        
        ++iteration;
        newlyMatched = newMatchedCols.getnnz();
        if(myrank == 0)
        {
            times.push_back(std::accumulate(times.begin(), times.end(), 0.0));
            timing.push_back(times);
            printf("%3d %10lld %10lld %10lld %18lf\n", iteration , curUnmatchedRow, curUnmatchedCol, newlyMatched, times.back());
        }
        
        curUnmatchedCol = unmatchedCol.getnnz();
        curUnmatchedRow = unmatchedRow.getnnz();
        MPI_Barrier(MPI_COMM_WORLD);
        
    }
    
    
    // print statistics
    if(myrank == 0)
    {
        cout << "==========================================================\n";
        cout << "\n================individual timings =======================\n";
        cout  << "     SpMV      Update-Match   Update-Deg    Total "<< endl;
        cout << "==========================================================\n";
        
        vector<double> totalTimes(timing[0].size(),0);
        for(int i=0; i<timing.size(); i++)
        {
            for(int j=0; j<timing[i].size(); j++)
            {
                totalTimes[j] += timing[i][j];
                printf("%12.5lf ", timing[i][j]);
            }
            cout << endl;
        }
        
        cout << "-------------------------------------------------------\n";
        for(int i=0; i<totalTimes.size(); i++)
             printf("%12.5lf ", totalTimes[i]);
        cout << endl;
    }
    
    
}





void greedyMatching(PSpMat_Int64 & A, PSpMat_Int64 & AT, FullyDistVec<int64_t, int64_t>& mateRow2Col,
                    FullyDistVec<int64_t, int64_t>& mateCol2Row, bool removeIsolate)
{
    
    int nprocs, myrank;
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    
    
    //unmatched row and column vertices
    FullyDistSpVec<int64_t, int64_t> unmatchedRow(mateRow2Col, [](int64_t mate){return mate==-1;});
    FullyDistSpVec<int64_t, int64_t> unmatchedCol(mateCol2Row, [](int64_t mate){return mate==-1;});
    unmatchedRow.setNumToInd();
    unmatchedCol.setNumToInd();
    
    
    //fringe vector (sparse)
    FullyDistSpVec<int64_t, int64_t> fringeRow(A.getcommgrid(), A.getnrow());
    FullyDistSpVec<int64_t, int64_t> fringeCol(A.getcommgrid(), A.getncol());
    
    FullyDistSpVec<int64_t, int64_t> degColSG(A.getcommgrid(), A.getncol());
    FullyDistVec<int64_t, int64_t> degCol(A.getcommgrid());
    if(removeIsolate)
    {
        
        
        // update initial degree of unmatched column vertices
        A.Reduce(degCol, Column, plus<int64_t>(), static_cast<int64_t>(0));
        unmatchedCol.Select(degCol, [](int64_t deg){return deg>0;}); // remove degree-0 columns
    }
    
    
    
    int64_t curUnmatchedCol = unmatchedCol.getnnz();
    int64_t curUnmatchedRow = unmatchedRow.getnnz();
    int64_t newlyMatched = 1; // ensure the first pass of the while loop
    int iteration = 0;
    double tStart = MPI_Wtime();
    vector<vector<double> > timing;
    if(myrank == 0)
    {
        cout << "=======================================================\n";
        cout << "@@@@@@ Number of processes: " << nprocs << endl;
        cout << "=======================================================\n";
        cout  << "It   |  UMRow   |  UMCol   |  newlyMatched   |  Time "<< endl;
        cout << "=======================================================\n";
    }
    MPI_Barrier(MPI_COMM_WORLD);
    
    //PSpMat_Int64  A = Aeff;
    //A.PrintInfo();
    
    while(curUnmatchedCol !=0 && curUnmatchedRow!=0 && newlyMatched != 0 )
    {
        
        // ======================== step1: One step of BFS =========================
        vector<double> times;
        double t1 = MPI_Wtime();
        SpMV<SelectMinSRing1>(A, unmatchedCol, fringeRow, false);
        fringeRow.Select(mateRow2Col, [](int64_t mate){return mate==-1;});
        if(myrank == 0){times.push_back(MPI_Wtime()-t1); t1 = MPI_Wtime();}
        // ===========================================================================
        
        
        // ======================== step2: Update matching  =========================
        FullyDistSpVec<int64_t, int64_t> newMatchedCols = fringeRow.Invert(A.getncol());
        FullyDistSpVec<int64_t, int64_t> newMatchedRows = newMatchedCols.Invert(A.getnrow());
        mateCol2Row.Set(newMatchedCols);
        mateRow2Col.Set(newMatchedRows);
        if(myrank == 0){times.push_back(MPI_Wtime()-t1); t1 = MPI_Wtime();}
        // ===========================================================================
        
        
        // =============== step3: Update degree of unmatched columns =================
        unmatchedRow.Select(mateRow2Col, [](int64_t mate){return mate==-1;});
        unmatchedCol.Select(mateCol2Row, [](int64_t mate){return mate==-1;});
        
        if(removeIsolate)
        {
            SpMV< SelectPlusSRing>(AT, newMatchedRows, degColSG, false);  // degree of column vertices to matched rows
            // subtract degree of column vertices
            degCol.EWiseApply(degColSG,
                          [](int64_t old_deg, int64_t new_deg, bool a, bool b){return old_deg-new_deg;},
                          [](int64_t old_deg, int64_t new_deg, bool a, bool b){return true;},
                          false, static_cast<int64_t>(0), false);
            unmatchedCol.Select(degCol, [](int64_t deg){return deg>0;});
        }
        
        if(myrank == 0){times.push_back(MPI_Wtime()-t1); t1 = MPI_Wtime();}
        // ===========================================================================
        
        ++iteration;
        newlyMatched = newMatchedCols.getnnz();
        if(myrank == 0)
        {
            times.push_back(std::accumulate(times.begin(), times.end(), 0.0));
            timing.push_back(times);
            printf("%3d %10lld %10lld %10lld %18lf\n", iteration , curUnmatchedRow, curUnmatchedCol, newlyMatched, times.back());
        }
        curUnmatchedCol = unmatchedCol.getnnz();
        curUnmatchedRow = unmatchedRow.getnnz();
        MPI_Barrier(MPI_COMM_WORLD);
        
    }
    
    
    if(myrank == 0)
    {
        cout << "==========================================================\n";
        cout << "\n================individual timings =======================\n";
        cout  << "     SpMV      Update-Match   Update-UMC    Total "<< endl;
        cout << "==========================================================\n";
        
        vector<double> totalTimes(timing[0].size(),0);
        for(int i=0; i<timing.size(); i++)
        {
            for(int j=0; j<timing[i].size(); j++)
            {
                totalTimes[j] += timing[i][j];
                printf("%12.5lf ", timing[i][j]);
            }
            cout << endl;
        }
        
        cout << "-------------------------------------------------------\n";
        for(int i=0; i<totalTimes.size(); i++)
            printf("%12.5lf ", totalTimes[i]);
        cout << endl;
        printf("%lld %lf\n",curUnmatchedRow, totalTimes.back());
    }
    
}



void greedyMatching1(PSpMat_Int64 & A, PSpMat_Int64 & AT, FullyDistVec<int64_t, int64_t>& mateRow2Col,
                    FullyDistVec<int64_t, int64_t>& mateCol2Row, bool removeIsolate)
{
    
    int nprocs, myrank;
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    
    
    //unmatched row and column vertices
    FullyDistSpVec<int64_t, int64_t> unmatchedRow(mateRow2Col, [](int64_t mate){return mate==-1;});
    FullyDistSpVec<int64_t, int64_t> unmatchedCol(mateCol2Row, [](int64_t mate){return mate==-1;});
    unmatchedRow.setNumToInd();
    unmatchedCol.setNumToInd();
    
    
    //fringe vector (sparse)
    //FullyDistSpVec<int64_t, int64_t> fringeRow(A.getcommgrid(), A.getnrow());
    FullyDistSpVec<int64_t, int64_t> fringeCol(A.getcommgrid(), A.getncol());
    
    FullyDistSpVec<int64_t, int64_t> degColSG(A.getcommgrid(), A.getncol());
    FullyDistVec<int64_t, int64_t> degCol(A.getcommgrid());
    if(removeIsolate)
    {
        
        
        // update initial degree of unmatched column vertices
        A.Reduce(degCol, Column, plus<int64_t>(), static_cast<int64_t>(0));
        unmatchedCol.Select(degCol, [](int64_t deg){return deg>0;}); // remove degree-0 columns
    }
    
    
    
    int64_t curUnmatchedCol = unmatchedCol.getnnz();
    int64_t curUnmatchedRow = unmatchedRow.getnnz();
    int64_t newlyMatched = 1; // ensure the first pass of the while loop
    int iteration = 0;
    double tStart = MPI_Wtime();
    vector<vector<double> > timing;
    if(myrank == 0)
    {
        cout << "=======================================================\n";
        cout << "@@@@@@ Number of processes: " << nprocs << endl;
        cout << "=======================================================\n";
        cout  << "It   |  UMRow   |  UMCol   |  newlyMatched   |  Time "<< endl;
        cout << "=======================================================\n";
    }
    MPI_Barrier(MPI_COMM_WORLD);
    
    //PSpMat_Int64  A = Aeff;
    //A.PrintInfo();
    
    while(curUnmatchedCol !=0 && curUnmatchedRow!=0 && newlyMatched != 0 )
    {
        
        // ======================== step1: One step of BFS =========================
        vector<double> times;
       
        FullyDistSpVec<int64_t, int64_t> unmatchedCold(unmatchedCol);
        FullyDistSpVec<int64_t, int64_t> fringeRowd;
         double t1 = MPI_Wtime();
        fringeRowd = SpMV<SelectMinSRing1>(A, unmatchedCold);
        if(myrank == 0){times.push_back(MPI_Wtime()-t1); t1 = MPI_Wtime();}
        FullyDistSpVec<int64_t, int64_t> fringeRow(fringeRowd, [](int64_t parent){return parent!=0;});
        fringeRow.Select(mateRow2Col, [](int64_t mate){return mate==-1;});
        
        // ===========================================================================
        
        
        // ======================== step2: Update matching  =========================
        FullyDistSpVec<int64_t, int64_t> newMatchedCols = fringeRow.Invert(A.getncol());
        FullyDistSpVec<int64_t, int64_t> newMatchedRows = newMatchedCols.Invert(A.getnrow());
        mateCol2Row.Set(newMatchedCols);
        mateRow2Col.Set(newMatchedRows);
        if(myrank == 0){times.push_back(MPI_Wtime()-t1); t1 = MPI_Wtime();}
        // ===========================================================================
        
        
        // =============== step3: Update degree of unmatched columns =================
        unmatchedRow.Select(mateRow2Col, [](int64_t mate){return mate==-1;});
        unmatchedCol.Select(mateCol2Row, [](int64_t mate){return mate==-1;});
        
        if(removeIsolate)
        {
            SpMV< SelectPlusSRing>(AT, newMatchedRows, degColSG, false);  // degree of column vertices to matched rows
            // subtract degree of column vertices
            degCol.EWiseApply(degColSG,
                              [](int64_t old_deg, int64_t new_deg, bool a, bool b){return old_deg-new_deg;},
                              [](int64_t old_deg, int64_t new_deg, bool a, bool b){return true;},
                              false, static_cast<int64_t>(0), false);
            unmatchedCol.Select(degCol, [](int64_t deg){return deg>0;});
        }
        
        if(myrank == 0){times.push_back(MPI_Wtime()-t1); t1 = MPI_Wtime();}
        // ===========================================================================
        
        ++iteration;
        newlyMatched = newMatchedCols.getnnz();
        if(myrank == 0)
        {
            times.push_back(std::accumulate(times.begin(), times.end(), 0.0));
            timing.push_back(times);
            printf("%3d %10lld %10lld %10lld %18lf\n", iteration , curUnmatchedRow, curUnmatchedCol, newlyMatched, times.back());
        }
        curUnmatchedCol = unmatchedCol.getnnz();
        curUnmatchedRow = unmatchedRow.getnnz();
        MPI_Barrier(MPI_COMM_WORLD);
        
    }
    
    
    if(myrank == 0)
    {
        cout << "==========================================================\n";
        cout << "\n================individual timings =======================\n";
        cout  << "     SpMV      Update-Match   Update-UMC    Total "<< endl;
        cout << "==========================================================\n";
        
        vector<double> totalTimes(timing[0].size(),0);
        for(int i=0; i<timing.size(); i++)
        {
            for(int j=0; j<timing[i].size(); j++)
            {
                totalTimes[j] += timing[i][j];
                printf("%12.5lf ", timing[i][j]);
            }
            cout << endl;
        }
        
        cout << "-------------------------------------------------------\n";
        for(int i=0; i<totalTimes.size(); i++)
            printf("%12.5lf ", totalTimes[i]);
        cout << endl;
        printf("%lld %lf\n",curUnmatchedRow, totalTimes.back());
    }
    
}



// note: randomization in the matrix does not not help, because it is another fixed ordering
// randomizing on the fly might work, but it is too expensive
// hence we stick to mindegree approach

void DMD(PSpMat_Int64 & A, PSpMat_Int64 & AT, FullyDistVec<int64_t, int64_t>& mateRow2Col,
            FullyDistVec<int64_t, int64_t>& mateCol2Row)
{
    
    int nprocs, myrank;
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    
    
    //unmatched row and column vertices
    FullyDistSpVec<int64_t, int64_t> unmatchedRow(mateRow2Col, [](int64_t mate){return mate==-1;});
    unmatchedRow.setNumToInd();
    FullyDistSpVec<int64_t, int64_t> degColSG(A.getcommgrid(), A.getncol());
    FullyDistVec<int64_t, int64_t> degCol(A.getcommgrid());
    A.Reduce(degCol, Column, plus<int64_t>(), static_cast<int64_t>(0));
    
    
    FullyDistSpVec<int64_t, VertexType> unmatchedCol(A.getcommgrid(), A.getncol());
    unmatchedCol  = EWiseApply<VertexType>(unmatchedCol, mateCol2Row, [](VertexType vtx, int64_t mate){return VertexType(-1,0);},
                                           [](VertexType vtx, int64_t mate){return mate==-1;}, true, VertexType());
    unmatchedCol.Select(degCol, [](int64_t deg){return deg>0;});
    unmatchedCol.ApplyInd([](VertexType vtx, int64_t idx){return VertexType(idx,0);}); //  parent equals to index
    unmatchedCol.SelectApply(degCol, [](int64_t deg){return deg>0;},
                             [](VertexType vtx, int64_t deg){return VertexType(vtx.parent,deg);});
    
    //fringe vector (sparse)
    FullyDistSpVec<int64_t, VertexType> fringeRow(A.getcommgrid(), A.getnrow());
    
    
    
    int64_t curUnmatchedCol = unmatchedCol.getnnz();
    int64_t curUnmatchedRow = unmatchedRow.getnnz();
    int64_t newlyMatched = 1; // ensure the first pass of the while loop
    int iteration = 0;
    double tStart = MPI_Wtime();
    vector<vector<double> > timing;
    if(myrank == 0)
    {
        cout << "=======================================================\n";
        cout << "@@@@@@ Number of processes: " << nprocs << endl;
        cout << "=======================================================\n";
        cout  << "It   |  UMRow   |  UMCol   |  newlyMatched   |  Time "<< endl;
        cout << "=======================================================\n";
    }
    MPI_Barrier(MPI_COMM_WORLD);
    
    
    while(curUnmatchedCol !=0 && curUnmatchedRow!=0 && newlyMatched != 0 )
    {
        
        // ======================== step1: One step of BFS =========================
        vector<double> times;
        double t1 = MPI_Wtime();

        SpMV<SelectMinDegSR>(A, unmatchedCol, fringeRow, false);
        // Remove matched row vertices
        fringeRow.Select(mateRow2Col, [](int64_t mate){return mate==-1;});
        if(myrank == 0){times.push_back(MPI_Wtime()-t1); t1 = MPI_Wtime();}
        // ===========================================================================
        
        
        // ======================== step2: Update matching  =========================
        FullyDistSpVec<int64_t, int64_t> fringeRow2(A.getcommgrid(), A.getnrow());
        
        fringeRow2  = EWiseApply<int64_t>(fringeRow, mateRow2Col, [](VertexType vtx, int64_t mate){return vtx.parent;},
                                          [](VertexType vtx, int64_t mate){return true;}, false, VertexType());
        
        FullyDistSpVec<int64_t, int64_t> newMatchedCols = fringeRow2.Invert(A.getncol());
        FullyDistSpVec<int64_t, int64_t> newMatchedRows = newMatchedCols.Invert(A.getnrow());
        mateCol2Row.Set(newMatchedCols);
        mateRow2Col.Set(newMatchedRows);
        if(myrank == 0){times.push_back(MPI_Wtime()-t1); t1 = MPI_Wtime();}
        // ===========================================================================
        
        
        // =============== step3: Update degree of unmatched columns =================
        unmatchedRow.Select(mateRow2Col, [](int64_t mate){return mate==-1;});
        unmatchedCol.Select(mateCol2Row, [](int64_t mate){return mate==-1;});
        
        // update degree
        SpMV< SelectPlusSRing>(AT, newMatchedRows, degColSG, false);  // degree of column vertices to matched rows
        // subtract degree of column vertices
        degCol.EWiseApply(degColSG,
                          [](int64_t old_deg, int64_t new_deg, bool a, bool b){return old_deg-new_deg;},
                          [](int64_t old_deg, int64_t new_deg, bool a, bool b){return true;},
                          false, static_cast<int64_t>(0), false);
        unmatchedCol.SelectApply(degCol, [](int64_t deg){return deg>0;},
                                 [](VertexType vtx, int64_t deg){return VertexType(vtx.parent,deg);});
        
        //A.DimApply(Row, mateRow2Col, [](int64_t val, int64_t mate){if(mate!=-1) return static_cast<int64_t> (0); else return val;});
        if(myrank == 0){times.push_back(MPI_Wtime()-t1); t1 = MPI_Wtime();}
        // ===========================================================================
        
        
        ++iteration;
        newlyMatched = newMatchedCols.getnnz();
        if(myrank == 0)
        {
            times.push_back(std::accumulate(times.begin(), times.end(), 0.0));
            timing.push_back(times);
            printf("%3d %10lld %10lld %10lld %18lf\n", iteration , curUnmatchedRow, curUnmatchedCol, newlyMatched, times.back());
        }
        
        curUnmatchedCol = unmatchedCol.getnnz();
        curUnmatchedRow = unmatchedRow.getnnz();
        MPI_Barrier(MPI_COMM_WORLD);
        
    }
   
    
    if(myrank == 0)
    {
        cout << "==========================================================\n";
        cout << "\n================individual timings =======================\n";
        cout  << "     SpMV      Update-Match   Update-UMC    Total "<< endl;
        cout << "==========================================================\n";
        
        vector<double> totalTimes(timing[0].size(),0);
        for(int i=0; i<timing.size(); i++)
        {
            for(int j=0; j<timing[i].size(); j++)
            {
                totalTimes[j] += timing[i][j];
                printf("%12.5lf ", timing[i][j]);
            }
            cout << endl;
        }
        
        cout << "-------------------------------------------------------\n";
        for(int i=0; i<totalTimes.size(); i++)
            printf("%12.5lf ", totalTimes[i]);
        cout << endl;
        
        printf("%lld %lf\n",curUnmatchedRow, totalTimes.back());
    }
}




void hybrid(PSpMat_Int64 & A, PSpMat_Int64 & AT, FullyDistVec<int64_t, int64_t>& mateRow2Col,
            FullyDistVec<int64_t, int64_t>& mateCol2Row, int type, bool rand)
{
    
    int nprocs, myrank;
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    
    
    
    
    //unmatched row and column vertices
    FullyDistSpVec<int64_t, int64_t> unmatchedRow(mateRow2Col, [](int64_t mate){return mate==-1;});
    unmatchedRow.setNumToInd();
    FullyDistSpVec<int64_t, int64_t> degColSG(A.getcommgrid(), A.getncol());
    FullyDistVec<int64_t, int64_t> degCol(A.getcommgrid());
    A.Reduce(degCol, Column, plus<int64_t>(), static_cast<int64_t>(0));
    
    
    FullyDistSpVec<int64_t, VertexType> unmatchedCol(A.getcommgrid(), A.getncol());
    unmatchedCol  = EWiseApply<VertexType>(unmatchedCol, mateCol2Row, [](VertexType vtx, int64_t mate){return VertexType(-1,0);},
                                           [](VertexType vtx, int64_t mate){return mate==-1;}, true, VertexType());
    unmatchedCol.Select(degCol, [](int64_t deg){return deg>0;});
    unmatchedCol.ApplyInd([](VertexType vtx, int64_t idx){return VertexType(idx,0);}); //  parent equals to index
    unmatchedCol.SelectApply(degCol, [](int64_t deg){return deg>0;},
                             [](VertexType vtx, int64_t deg){return VertexType(vtx.parent,deg);});
    
    //fringe vector (sparse)
    FullyDistSpVec<int64_t, VertexType> fringeRow(A.getcommgrid(), A.getnrow());
    
    
    if(rand)
    {
        A.Apply([](int64_t x){return static_cast<int64_t>((GlobalMT.rand() * 100000)+1);}); // perform randomization
    }
    
    int64_t curUnmatchedCol = unmatchedCol.getnnz();
    int64_t curUnmatchedRow = unmatchedRow.getnnz();
    int64_t newlyMatched = 1; // ensure the first pass of the while loop
    int iteration = 0;
    double tStart = MPI_Wtime();
    vector<vector<double> > timing;
    if(myrank == 0)
    {
        cout << "=======================================================\n";
        cout << "@@@@@@ Number of processes: " << nprocs << endl;
        cout << "=======================================================\n";
        cout  << "It   |  UMRow   |  UMCol   |  newlyMatched   |  Time "<< endl;
        cout << "=======================================================\n";
    }
    MPI_Barrier(MPI_COMM_WORLD);
    
    
    while(curUnmatchedCol !=0 && curUnmatchedRow!=0 && newlyMatched != 0 )
    {
        
        // ======================== step1: One step of BFS =========================
        vector<double> times;
        double t1 = MPI_Wtime();
        if(type==GREEDY)
        {
            if(rand) SpMV<GreedyRandSR>(A, unmatchedCol, fringeRow, false);
            else SpMV<GreedySR>(A, unmatchedCol, fringeRow, false);
        }
        else if(type==KARP_SIPSER)
        {
            FullyDistSpVec<int64_t, VertexType> deg1Col = unmatchedCol;
            deg1Col.Select(degCol, [](int64_t deg){return deg==1;});
            
            if(rand)
            {
                if(deg1Col.getnnz()>9)
                    SpMV<GreedyRandSR>(A, deg1Col, fringeRow, false);
                else
                    SpMV<GreedyRandSR>(A, unmatchedCol, fringeRow, false);
            }
            else
            {
                if(deg1Col.getnnz()>9)
                    SpMV<GreedySR>(A, deg1Col, fringeRow, false);
                else
                    SpMV<GreedySR>(A, unmatchedCol, fringeRow, false);
            }
            
        }
        else // if (type==KS_GREEDY)  this is default
        {
            if(rand) SpMV<KSGreedyRandSR>(A, unmatchedCol, fringeRow, false);
            else SpMV<KSGreedySR>(A, unmatchedCol, fringeRow, false);
        }
        
        // Remove matched row vertices
        fringeRow.Select(mateRow2Col, [](int64_t mate){return mate==-1;});
        if(myrank == 0){times.push_back(MPI_Wtime()-t1); t1 = MPI_Wtime();}
        // ===========================================================================
        
        
        // ======================== step2: Update matching  =========================
        FullyDistSpVec<int64_t, int64_t> fringeRow2(A.getcommgrid(), A.getnrow());
        
        fringeRow2  = EWiseApply<int64_t>(fringeRow, mateRow2Col, [](VertexType vtx, int64_t mate){return vtx.parent;},
                                          [](VertexType vtx, int64_t mate){return true;}, false, VertexType());
        
        FullyDistSpVec<int64_t, int64_t> newMatchedCols = fringeRow2.Invert(A.getncol());
        FullyDistSpVec<int64_t, int64_t> newMatchedRows = newMatchedCols.Invert(A.getnrow());
        mateCol2Row.Set(newMatchedCols);
        mateRow2Col.Set(newMatchedRows);
        if(myrank == 0){times.push_back(MPI_Wtime()-t1); t1 = MPI_Wtime();}
        // ===========================================================================
        
        
        // =============== step3: Update degree of unmatched columns =================
        unmatchedRow.Select(mateRow2Col, [](int64_t mate){return mate==-1;});
        unmatchedCol.Select(mateCol2Row, [](int64_t mate){return mate==-1;});
        
        // update degree
        SpMV< SelectPlusSRing>(AT, newMatchedRows, degColSG, false);  // degree of column vertices to matched rows
        // subtract degree of column vertices
        degCol.EWiseApply(degColSG,
                          [](int64_t old_deg, int64_t new_deg, bool a, bool b){return old_deg-new_deg;},
                          [](int64_t old_deg, int64_t new_deg, bool a, bool b){return true;},
                          false, static_cast<int64_t>(0), false);
        unmatchedCol.SelectApply(degCol, [](int64_t deg){return deg>0;},
                                 [](VertexType vtx, int64_t deg){return VertexType(vtx.parent,deg);});
        if(myrank == 0){times.push_back(MPI_Wtime()-t1); t1 = MPI_Wtime();}
        // ===========================================================================
        
        
        ++iteration;
        newlyMatched = newMatchedCols.getnnz();
        if(myrank == 0)
        {
            times.push_back(std::accumulate(times.begin(), times.end(), 0.0));
            timing.push_back(times);
            printf("%3d %10lld %10lld %10lld %18lf\n", iteration , curUnmatchedRow, curUnmatchedCol, newlyMatched, times.back());
        }
        
        curUnmatchedCol = unmatchedCol.getnnz();
        curUnmatchedRow = unmatchedRow.getnnz();
        MPI_Barrier(MPI_COMM_WORLD);
        
    }
    
    if(rand) // undo randomization
    {
        A.Apply([](int64_t x){return static_cast<int64_t>(1);});
    }
    
    if(myrank == 0)
    {
        cout << "==========================================================\n";
        cout << "\n================individual timings =======================\n";
        cout  << "     SpMV      Update-Match   Update-UMC    Total "<< endl;
        cout << "==========================================================\n";
        
        vector<double> totalTimes(timing[0].size(),0);
        for(int i=0; i<timing.size(); i++)
        {
            for(int j=0; j<timing[i].size(); j++)
            {
                totalTimes[j] += timing[i][j];
                printf("%12.5lf ", timing[i][j]);
            }
            cout << endl;
        }
        
        cout << "-------------------------------------------------------\n";
        for(int i=0; i<totalTimes.size(); i++)
            printf("%12.5lf ", totalTimes[i]);
        cout << endl;
        
        printf("%lld %lf\n",curUnmatchedRow, totalTimes.back());
    }
}




/*

void greedyMatching_old(PSpMat_Int64  & A)
{
    
    int nprocs, myrank;
	MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    
    
    //matching vector (dense)
    FullyDistVec<int64_t, int64_t> mateRow2Col ( A.getcommgrid(), A.getnrow(), (int64_t) -1);
    FullyDistVec<int64_t, int64_t> mateCol2Row ( A.getcommgrid(), A.getncol(), (int64_t) -1);
    
    //unmatched row and column vertices
    FullyDistSpVec<int64_t, int64_t> unmatchedRow(mateRow2Col, unmatched<int64_t>());
    FullyDistSpVec<int64_t, int64_t> unmatchedCol(mateCol2Row, unmatched<int64_t>());
    unmatchedRow.setNumToInd();
    unmatchedCol.setNumToInd();
    
    
    //fringe vector (sparse)
    FullyDistSpVec<int64_t, int64_t> fringeRow(A.getcommgrid(), A.getnrow());
    FullyDistSpVec<int64_t, int64_t> fringeCol(A.getcommgrid(), A.getncol());
    
    
    int64_t curUnmatchedCol = unmatchedCol.getnnz();
    int64_t curUnmatchedRow = unmatchedRow.getnnz();
    int64_t newlyMatched = 1; // ensure the first pass of the while loop
    int iteration = 0;
    double tStart = MPI_Wtime();
    vector<vector<double> > timing;
    if(myrank == 0)
    {
        cout << "=======================================================\n";
        cout << "@@@@@@ Number of processes: " << nprocs << endl;
        cout << "=======================================================\n";
        cout  << "It   |  UMRow   |  UMCol   |  newlyMatched   |  Time "<< endl;
        cout << "=======================================================\n";
    }
    MPI_Barrier(MPI_COMM_WORLD);


    
    while(curUnmatchedCol !=0 && curUnmatchedRow!=0 && newlyMatched != 0 )
    {
        vector<double> times;
        double t1 = MPI_Wtime();
        // step1: Find adjacent row vertices (col vertices parent, row vertices child)
        //fringeRow = SpMV(Aeff, unmatchedCol, optbuf);
        //SpMV<SelectMinSRing1>(Aeff, unmatchedCol, fringeRow, false, optbuf);
        SpMV<SelectMinSRing1>(A, unmatchedCol, fringeRow, false);
        
        
        // step2: Remove matched row vertices
        fringeRow = EWiseMult(fringeRow, mateRow2Col, true, (int64_t) -1);
        if(myrank == 0){times.push_back(MPI_Wtime()-t1); t1 = MPI_Wtime();}
        
        // step3: Remove duplicate row vertices
        fringeRow = fringeRow.Uniq();
        if(myrank == 0){times.push_back(MPI_Wtime()-t1); t1 = MPI_Wtime();}
        //fringeRow.DebugPrint();
        
        // step4: Update mateRow2Col with the newly matched row vertices
        mateRow2Col.Set(fringeRow);
        //mateRow2Col.DebugPrint();
        
        // step5: Update mateCol2Row with the newly matched col vertices
        FullyDistSpVec<int64_t, int64_t> temp = fringeRow.Invert(A.getncol());
        mateCol2Row.Set(temp);
        // mateCol2Row.SetInd2Val(fringeRow);
        if(myrank == 0){times.push_back(MPI_Wtime()-t1); t1 = MPI_Wtime();}
        
        // step6: Update unmatchedCol/unmatchedRow by removing newly matched columns/rows
        unmatchedCol = EWiseMult(unmatchedCol, mateCol2Row, true, (int64_t) -1);
        unmatchedRow = EWiseMult(unmatchedRow, mateRow2Col, true, (int64_t) -1);
        if(myrank == 0){times.push_back(MPI_Wtime()-t1); t1 = MPI_Wtime();}
        
        
        ++iteration;
        newlyMatched = fringeRow.getnnz();
        if(myrank == 0)
        {
            times.push_back(std::accumulate(times.begin(), times.end(), 0.0));
            timing.push_back(times);
            cout  << iteration <<  "  " << curUnmatchedRow  <<  "  " << curUnmatchedCol  <<  "  " << newlyMatched <<  "  " << times.back() << endl;
        }
        
        curUnmatchedCol = unmatchedCol.getnnz();
        curUnmatchedRow = unmatchedRow.getnnz();
        MPI_Barrier(MPI_COMM_WORLD);
        
    }
    
    
    
    
    //Check if this is a maximal matching
    //mateRow2Col.DebugPrint();
    //mateCol2Row.DebugPrint();
    isMaximalmatching(A, mateRow2Col, mateCol2Row);
    //isMatching(mateCol2Row, mateRow2Col); //todo there is a better way to check this
    
    
    // print statistics
    if(myrank == 0)
    {
        cout << "============================================================\n";
        cout << "\n================individual timings =========================\n";
        cout  << "SpMV  |  Uniq   |  Permute   |  Update matching  |  Total "<< endl;
        cout << "============================================================\n";
        
        vector<double> totalTimes(timing[0].size(),0);
        for(int i=0; i<timing.size(); i++)
        {
            for(int j=0; j<timing[i].size(); j++)
            {
                totalTimes[j] += timing[i][j];
                cout << timing[i][j] << "  ";
            }
            cout << endl;
        }
        
        cout << "=================== total timing ===========================\n";
        for(int i=0; i<totalTimes.size(); i++)
            cout<<totalTimes[i] << " ";
        cout << endl;
    }
    
    
}

*/


template <class IT, class NT>
bool isMaximalmatching(PSpMat_Int64 & A, FullyDistVec<IT,NT> & mateRow2Col, FullyDistVec<IT,NT> & mateCol2Row)
{
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    FullyDistSpVec<int64_t, int64_t> fringeRow(A.getcommgrid(), A.getnrow());
    FullyDistSpVec<int64_t, int64_t> fringeCol(A.getcommgrid(), A.getncol());
    FullyDistSpVec<int64_t, int64_t> unmatchedRow(mateRow2Col, [](int64_t mate){return mate==-1;});
    FullyDistSpVec<int64_t, int64_t> unmatchedCol(mateCol2Row, [](int64_t mate){return mate==-1;});
    unmatchedRow.setNumToInd();
    unmatchedCol.setNumToInd();
    
    
    SpMV<SelectMinSRing1>(A, unmatchedCol, fringeRow, false);
    fringeRow = EWiseMult(fringeRow, mateRow2Col, true, (int64_t) -1);
    if(fringeRow.getnnz() != 0)
    {
        if(myrank == 0)
            cout << "Not maximal matching!!\n";
        return false;
    }
    
    PSpMat_Int64 tA = A;
    tA.Transpose();
    SpMV<SelectMinSRing1>(tA, unmatchedRow, fringeCol, false);
    fringeCol = EWiseMult(fringeCol, mateCol2Row, true, (int64_t) -1);
    if(fringeCol.getnnz() != 0)
    {
        if(myrank == 0)
            cout << "Not maximal matching**!!\n";
        return false;
    }
    return true;
}


/*
 * Serial: Check the validity of the matching solution; 
 we need a better solution using invert
 */
template <class IT, class NT>
bool isMatching(FullyDistVec<IT,NT> & mateCol2Row, FullyDistVec<IT,NT> & mateRow2Col)
{
    
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    for(int i=0; i< mateRow2Col.glen ; i++)
    {
        int t = mateRow2Col[i];
        
        if(t!=-1 && mateCol2Row[t]!=i)
        {
            if(myrank == 0)
                cout << "Does not satisfy the matching constraints\n";
            return false;
        }
    }
    
    for(int i=0; i< mateCol2Row.glen ; i++)
    {
        int t = mateCol2Row[i];
        if(t!=-1 && mateRow2Col[t]!=i)
        {
            if(myrank == 0)
                cout << "Does not satisfy the matching constraints\n";
            return false;
        }
    }
    return true;
}






