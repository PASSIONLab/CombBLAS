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

#define ITERS 16
#define EDGEFACTOR 16
using namespace std;


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




struct VertexType
{
public:
	VertexType(){parent=-1; root = -1;};
    explicit VertexType(int64_t p){parent=p; root=-1;}; // why do we need this constructor ? otherwise compile error
	VertexType(int64_t p, int64_t r){parent=p; root = r;};
    friend ostream& operator<<(ostream& os, const VertexType & vertex ){os << "Parent=" << vertex.parent << " Root=" << vertex.root << "\n"; return os;};
    //private:
    int64_t parent;
    int64_t root;

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
    
     static void axpy(bool a, const T_promote & x, T_promote & y)
     {
     y = std::max(y, x);
     }
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
    //static MPI_Op MPI_BFSRAND;
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
struct SelectMinSRing2
{
	typedef int64_t T_promote;
	static VertexType id(){ return VertexType(); };
	static bool returnedSAID() { return false; }
	//static MPI_Op mpi_op() { return MPI_MIN; };
    
	static VertexType add(const VertexType & arg1, const VertexType & arg2)
	{
		if(arg1.parent < arg2.parent)
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




typedef SpParMat < int64_t, bool, SpDCCols<int64_t,bool> > PSpMat_Bool;
typedef SpParMat < int64_t, bool, SpDCCols<int32_t,bool> > PSpMat_s32p64;
typedef SpParMat < int64_t, int64_t, SpDCCols<int64_t,int64_t> > PSpMat_Int64;
void greedyMatching(PSpMat_Bool & Aeff);
void maximumMatching(PSpMat_Bool & Aeff);
template <class IT, class NT>
bool isMaximalmatching(PSpMat_Int64 & A, FullyDistVec<IT,NT> & mateRow2Col, FullyDistVec<IT,NT> & mateCol2Row,
                       FullyDistSpVec<int64_t, int64_t> unmatchedRow, FullyDistSpVec<int64_t, int64_t> unmatchedCol);

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
    //fringe.DebugPrint();
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
        
        
        FullyDistVec<int64_t, int64_t> * ColSums = new FullyDistVec<int64_t, int64_t>(ABool->getcommgrid());
        FullyDistVec<int64_t, int64_t> * RowSums = new FullyDistVec<int64_t, int64_t>(ABool->getcommgrid());
        
        ABool->Reduce(*ColSums, Column, plus<int64_t>(), static_cast<int64_t>(0));
        ABool->Reduce(*RowSums, Row, plus<int64_t>(), static_cast<int64_t>(0));
        ColSums->EWiseApply(*RowSums, plus<int64_t>());
        delete RowSums;
        nonisov = ColSums->FindInds(bind2nd(greater<int64_t>(), 0));
        delete ColSums;
        nonisov.RandPerm();	// so that A(v,v) is load-balanced (both memory and time wise)
#ifndef NOPERMUTE
        ABool->operator()(nonisov, nonisov, true);	// in-place permute to save memory
#endif

        // remove isolated vertice if necessary
        
        //RandomParentBFS(*ABool);
        //greedyMatching(*ABool);
        maximumMatching(*ABool);
        
        
	}
	MPI_Finalize();
	return 0;
}




template<typename T>
struct unmatched : public std::unary_function<T, bool>
{
    bool operator()(const T& x) const
    {
        return (x==-1);
    }
};



template<typename T1, typename T2>
struct unmatched1: public std::binary_function<T1, T2, bool>
{
    bool operator()(const T1& x, const T2 & y) const
    {
        return (y==-1);
    }
};


template<typename T1, typename T2>
struct sel1st: public std::binary_function<T1, T2, bool>
{
    const T1& operator()(const T1& x, const T2 & y) const
    {
        return x;
    }
};




void maximumMatching(PSpMat_Bool & Aeff)
{
    
    int nprocs, myrank;
	MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    
    
    //matching vector (dense)
    FullyDistVec<int64_t, int64_t> mateRow2Col ( Aeff.getcommgrid(), Aeff.getnrow(), (int64_t) -1);
    FullyDistVec<int64_t, int64_t> mateCol2Row ( Aeff.getcommgrid(), Aeff.getncol(), (int64_t) -1);
    //FullyDistSpVec<int64_t, VertexType> unmatchedCol(Aeff.getcommgrid(), Aeff.getncol());
    //cout << fringeCol[1];
    //FullyDistSpVec<int64_t, VertexType> fringeCol;
    //FullyDistSpVec<int64_t, VertexType> fringeRow;
    
    //FullyDistSpVec<int64_t, VertexType> unmatchedCol  = EWiseApply<VertexType, int64_t, VertexType, int64_t>(unmatchedCol1, mateCol2Row, sel1st<VertexType, int64_t>(), unmatched1<VertexType,int64_t>(), false, VertexType());
    //unmatchedCol  = EWiseApply<VertexType>(unmatchedCol, mateCol2Row, sel1st<VertexType, int64_t>(), unmatched1<VertexType,int64_t>(), false, VertexType());
    //fringeCol = unmatchedCol;
    //cout << fringeCol[1];
    

     FullyDistSpVec<int64_t, VertexType> fringeCol;
    fringeCol.SetElement(0, VertexType(0,-1));

    //FullyDistSpVec<int64_t, VertexType> fringeCol;
    //fringeCol.SetElement(0, VertexType());

    
    PSpMat_Int64  A = Aeff;
    //A.PrintInfo();
    //SpMV<SelectRandSRing>(A, fringeCol, fringeCol, false);
    
    SpMV<SelectMinSRing2>(A, fringeCol, fringeCol, false);
    
    //fringeCol.DebugPrint(); // does not work for vector of custom data type
    
    //FullyDistSpVec<int64_t, VertexType> fringeCol(mateCol2Row, unmatched<int64_t>());
    
    //FullyDistVec<int64_t, int64_t> parentsRow ( Aeff.getcommgrid(), Aeff.getnrow(), (int64_t) -1);
    
    //unmatched row and column vertices
    //FullyDistSpVec<int64_t, int64_t> unmatchedRow(mateRow2Col, unmatched<int64_t>());
    //FullyDistSpVec<int64_t, int64_t> unmatchedCol(mateCol2Row, unmatched<int64_t>());
    //unmatchedRow.setNumToInd();
    //unmatchedCol.setNumToInd();
    
    
    //fringe vector (sparse)
    //FullyDistSpVec<int64_t, int64_t> fringeRow(Aeff.getcommgrid(), Aeff.getnrow());
    //FullyDistSpVec<int64_t, int64_t> fringeCol(Aeff.getcommgrid(), Aeff.getncol());
    
    //FullyDistSpVec<int64_t, int64_t> unmatchedCol(mateCol2Row, unmatched<int64_t>());
    
    
    
    
    /*
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
    
    PSpMat_Int64  A = Aeff;
    A.PrintInfo();
    
    fringeCol = unmatchedCol;
    //while(fringeCol.getnnz() >0)
    {

        SpMV<SelectMinSRing1>(A, fringeCol, fringeRow, false);
        fringeRow.DebugPrint();
        fringeRow = EWiseMult(fringeRow, parentsRow, true, (int64_t) -1);
        fringeRow.DebugPrint();
        parentsRow.Set(fringeRow);
        fringeRow = EWiseMult(fringeRow, mateRow2Col, true, (int64_t) -1); //unmatched rows
        fringeRow.DebugPrint();
        fringeCol = fringeRow.Invert(Aeff.getncol());
        fringeCol.DebugPrint();
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    */
    /*
    
    //Check if this is a maximal matching
    //mateRow2Col.DebugPrint();
    //mateCol2Row.DebugPrint();
    isMaximalmatching(A, mateRow2Col, mateCol2Row, unmatchedRow, unmatchedCol);
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
    */
    
}


/*

void greedyMatching(PSpMat_Bool & Aeff)
{
    
    int nprocs, myrank;
	MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    
    
    //matching vector (dense)
    FullyDistVec<int64_t, int64_t> mateRow2Col ( Aeff.getcommgrid(), Aeff.getnrow(), (int64_t) -1);
    FullyDistVec<int64_t, int64_t> mateCol2Row ( Aeff.getcommgrid(), Aeff.getncol(), (int64_t) -1);
    
    //unmatched row and column vertices
    FullyDistSpVec<int64_t, int64_t> unmatchedRow(mateRow2Col, unmatched<int64_t>());
    FullyDistSpVec<int64_t, int64_t> unmatchedCol(mateCol2Row, unmatched<int64_t>());
    unmatchedRow.setNumToInd();
    unmatchedCol.setNumToInd();
    
    
    //fringe vector (sparse)
    FullyDistSpVec<int64_t, int64_t> fringeRow(Aeff.getcommgrid(), Aeff.getnrow());
    FullyDistSpVec<int64_t, int64_t> fringeCol(Aeff.getcommgrid(), Aeff.getncol());
    
    
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
    
    PSpMat_Int64  A = Aeff;
    A.PrintInfo();
    
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
        FullyDistSpVec<int64_t, int64_t> temp = fringeRow.Invert(Aeff.getncol());
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
    isMaximalmatching(A, mateRow2Col, mateCol2Row, unmatchedRow, unmatchedCol);
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

template <class IT, class NT>
bool isMaximalmatching(PSpMat_Int64 & A, FullyDistVec<IT,NT> & mateRow2Col, FullyDistVec<IT,NT> & mateCol2Row,
                       FullyDistSpVec<int64_t, int64_t> unmatchedRow, FullyDistSpVec<int64_t, int64_t> unmatchedCol)
{
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    FullyDistSpVec<int64_t, int64_t> fringeRow(A.getcommgrid(), A.getnrow());
    FullyDistSpVec<int64_t, int64_t> fringeCol(A.getcommgrid(), A.getncol());
    
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
 */

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






