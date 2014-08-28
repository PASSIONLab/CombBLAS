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
    VertexType(int64_t p){parent=p; root=-1;}; // this constructor is called when we assign vertextype=number. Called from ApplyInd function
	VertexType(int64_t p, int64_t r){parent=p; root = r;};
    friend ostream& operator<<(ostream& os, const VertexType & vertex ){os << "(" << vertex.parent << "," << vertex.root << ")"; return os;};
    //private:
    int64_t parent;
    int64_t root;

};




// This one is used for maximal matching
struct SelectMinSRing1
{
	typedef int64_t T_promote;
	static T_promote id(){ return -1; };
	static bool returnedSAID() { return false; }
	//static MPI_Op mpi_op() { return MPI_MAX; };
    
	static T_promote add(const T_promote & arg1, const T_promote & arg2)
	{
        //cout << arg1 << " a " << arg2 << endl;
		return std::max(arg1, arg2);
	}
    
	static T_promote multiply(const bool & arg1, const T_promote & arg2)
	{
        //cout << arg1 << " m " << arg2 << endl;
		return arg2;
	}
    
     static void axpy(bool a, const T_promote & x, T_promote & y)
     {
     y = std::max(y, x);
     }
};




// This one is used for maximum matching
struct SelectMinSRing2
{
	typedef int64_t T_promote;
	static VertexType id(){ return VertexType(); };
	static bool returnedSAID() { return false; }
	//static MPI_Op mpi_op() { return MPI_MIN; };
    
	static VertexType add(const VertexType & arg1, const VertexType & arg2)
	{
        cout << arg1 << " + " << arg2 << endl;
		if(arg1.parent < arg2.parent)
            return arg1;
        else
            return arg2;
	}
    
	static VertexType multiply(const T_promote & arg1, const VertexType & arg2)
	{
        cout << arg1 << " * " << arg2 << endl;
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
void maximumMatchingSimple(PSpMat_Bool & Aeff);
template <class IT, class NT>
bool isMaximalmatching(PSpMat_Int64 & A, FullyDistVec<IT,NT> & mateRow2Col, FullyDistVec<IT,NT> & mateCol2Row,
                       FullyDistSpVec<int64_t, int64_t> unmatchedRow, FullyDistSpVec<int64_t, int64_t> unmatchedCol);




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
			cout << "Usage: ./bpmm <Scale>" << endl;
			cout << "Example: mpirun -np 4 ./bpmm 20" << endl;
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
        //maximumMatchingSimple(*ABool);
        
        
        
	}
	MPI_Finalize();
	return 0;
}




template<typename T>
struct unmatched_unary : public std::unary_function<T, bool>
{
    bool operator()(const T& x) const
    {
        return (x==-1);
    }
};



template<typename T1, typename T2>
struct unmatched_binary: public std::binary_function<T1, T2, bool>
{
    bool operator()(const T1& x, const T2 & y) const
    {
        return (y==-1);
    }
};


// an unary operator would suffice. But the EWiseApply function takes a binary predicate
// this function when used as a predicate select the matched entried
template<typename T1, typename T2>
struct matched_binary: public std::binary_function<T1, T2, bool>
{
    bool operator()(const T1& x, const T2 & y) const
    {
        return (y!=-1);
    }
};


template<typename T1, typename T2>
struct select1st: public std::binary_function<T1, T2, bool>
{
    const T1& operator()(const T1& x, const T2 & y) const
    {
        return x;
    }
};


// returns the second argument
template<typename T1, typename T2>
struct select2nd: public std::binary_function<T1, T2, bool>
{
    const T2& operator()(const T1& x, const T2 & y) const
    {
        cout << y << "....\n";
        return y;
    }
};



// init
template<typename T1, typename T2>
struct init: public std::binary_function<T1, T2, bool>
{
    const T1 operator()(const T1& x, const T2 & y) const
    {
        return T1(y,y);
    }
};



// init
template<typename T>
struct binopInd: public std::binary_function<VertexType, T, T>
{
    const T operator()(const VertexType& vtx, const T & index) const
    {
        return vtx.parent;
    }
};



// init
template<typename T>
struct binopVal: public std::binary_function<VertexType, T, VertexType>
{
    const VertexType operator()(const VertexType& vtx, const T & index) const
    {
        return VertexType(index, vtx.root);
    }
};




void maximumMatchingSimple(PSpMat_Bool & Aeff)
{
    int nprocs, myrank;
	MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    
    PSpMat_Int64  A = Aeff;
    
    FullyDistVec<int64_t, int64_t> mateRow2Col ( A.getcommgrid(), A.getnrow(), (int64_t) -1);
    FullyDistVec<int64_t, int64_t> mateCol2Row ( A.getcommgrid(), A.getncol(), (int64_t) -1);
    FullyDistVec<int64_t, int64_t> parentsRow ( A.getcommgrid(), A.getnrow(), (int64_t) -1);
    FullyDistVec<int64_t, int64_t> rootsRow ( A.getcommgrid(), A.getnrow(), (int64_t) -1);
    FullyDistVec<int64_t, int64_t> rootsCol ( A.getcommgrid(), A.getncol(), (int64_t) -1);
    FullyDistSpVec<int64_t, int64_t> fringeRow(A.getcommgrid(), A.getnrow());
    FullyDistSpVec<int64_t, int64_t> umFringeRow(A.getcommgrid(), A.getnrow()); // unmatched vertices in the current fringeRow
    FullyDistSpVec<int64_t, int64_t> rootFringeRow(A.getcommgrid(), A.getnrow());
    FullyDistSpVec<int64_t, int64_t> rootFringeCol(A.getcommgrid(), A.getncol());
    
    
    FullyDistSpVec<int64_t, int64_t> fringeCol(mateCol2Row, unmatched_unary<int64_t>());
    fringeCol.ApplyInd(select2nd<VertexType, int64_t>());
    
    
    A.PrintInfo();
    while(fringeCol.getnnz() > 0)
    {
        fringeCol.setNumToInd();
        fringeCol.DebugPrint();
        SpMV<SelectMinSRing1>(A, fringeCol, fringeRow, false);
        fringeRow.DebugPrint();
        /*
        fringeRow = EWiseMult(fringeRow, parentsRow, true, (int64_t) -1);	// clean-up vertices that already have parents
        parentsRow.Set(fringeRow);
        
        // pass root information
        SpMV<SelectMinSRing1>(A, rootFringeCol, rootFringeRow, false); // this will not work... we need a sparse value based set operation
        fringeRow = EWiseMult(fringeRow, parentsRow, true, (int64_t) -1);	// clean-up vertices that already has parents
        parentsRow.Set(fringeRow);
        
        if(fringeRow.getnnz()>0) fringeRow.DebugPrint();
        
        umFringeRow = EWiseApply<int64_t>(fringeRow, mateRow2Col, select1st<int64_t, int64_t>(), unmatched_binary<int64_t,int64_t>(), false, (int64_t) 0);
        if(umFringeRow .getnnz()>0) break;
        
        // faster than using unary function in the constructor.
        // Here we are accessing the sparse vector, but in the constructor we access the dense vector
        fringeRow = EWiseApply<int64_t>(fringeRow, mateRow2Col, select2nd<int64_t, int64_t>(), matched_binary<int64_t,int64_t>(), false, (int64_t) 0);
        if(fringeRow.getnnz()>0) fringeRow.DebugPrint();
        
        fringeCol = fringeRow.Invert(A.getncol());
        //fringeCol.DebugPrint();
         */
        
        break;

    }
    
    //augment
    
    
}

void maximumMatching(PSpMat_Bool & Aeff)
{
    
    int nprocs, myrank;
	MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    
    PSpMat_Int64  A = Aeff;
    
    //matching vector (dense)
    FullyDistVec<int64_t, int64_t> mateRow2Col ( Aeff.getcommgrid(), Aeff.getnrow(), (int64_t) -1);
    FullyDistVec<int64_t, int64_t> mateCol2Row ( Aeff.getcommgrid(), Aeff.getncol(), (int64_t) -1);
    FullyDistVec<int64_t, int64_t> parentsRow ( Aeff.getcommgrid(), Aeff.getnrow(), (int64_t) -1);
    FullyDistSpVec<int64_t, VertexType> unmatchedCol(Aeff.getcommgrid(), Aeff.getncol());
    FullyDistSpVec<int64_t, VertexType> fringeCol(Aeff.getcommgrid(), Aeff.getncol());
    FullyDistSpVec<int64_t, VertexType> fringeRow(Aeff.getcommgrid(), Aeff.getnrow());
    
    
    
    //FullyDistSpVec<int64_t, VertexType> umFringeRow(Aeff.getcommgrid(), Aeff.getnrow());
    //FullyDistSpVec<int64_t, VertexType> mFringeRow(Aeff.getcommgrid(), Aeff.getnrow());
    
    fringeCol  = EWiseApply<VertexType>(fringeCol, mateCol2Row, select1st<VertexType, int64_t>(), unmatched_binary<VertexType,int64_t>(), true, VertexType()); // root & parent both =-1
    //fringeCol.SetElement(1,VertexType(1,1));
    fringeCol.ApplyInd(init<VertexType, int64_t>()); //  root & parent both equal to index
    fringeCol.DebugPrint();
    
    A.PrintInfo();
    
     while(fringeCol.getnnz() > 0)
     {
        SpMV<SelectMinSRing2>(A, fringeCol, fringeRow, false);
        fringeRow  = EWiseApply<VertexType>(fringeRow, parentsRow, select1st<VertexType, int64_t>(), unmatched_binary<VertexType,int64_t>(), false, VertexType());
        //parentsRow.Set(fringeRow);
        fringeRow.DebugPrint();
        
         fringeCol = fringeRow.Compose(Aeff.getncol(), binopInd<int64_t>(), binopVal<int64_t>());
         fringeCol.DebugPrint();
        //fringeRow  = EWiseApply<VertexType>(fringeCol, mateCol2Row, select1st<VertexType, int64_t>(), unmatched_binary<VertexType,int64_t>(), true, VertexType()); // root & parent both =-1
     
         break;
     //fringeCol.DebugPrint();
     
     }
    
    //fringeCol = unmatchedCol;
    //SpMV<SelectMinSRing2>(A, fringeCol, fringeRow, false);
    
    //umFringeRow = EWiseApply<VertexType>(fringeRow, mateRow2Col, select1st<VertexType, int64_t>(), unmatched_binary<VertexType,int64_t>(), false, VertexType());
    
    
    //fringeCol  = EWiseApply<VertexType>(fringeCol, mateCol2Row, select1st<VertexType, int64_t>(), unmatched_binary<VertexType,int64_t>(), false, VertexType()); // root & parent both =-1
    
    
    
    
    /*
    while(fringeCol.getnnz() > 0)
    {
        fringeCol.setNumToInd();
        fringeCol.DebugPrint();
        
        //fringeCol.DebugPrint();
        
    }*/
    
    //cout << fringeCol[1];
    //FullyDistSpVec<int64_t, VertexType> fringeCol;
    //FullyDistSpVec<int64_t, VertexType> fringeRow;
    
    //FullyDistSpVec<int64_t, VertexType> unmatchedCol  = EWiseApply<VertexType, int64_t, VertexType, int64_t>(unmatchedCol1, mateCol2Row, sel1st<VertexType, int64_t>(), unmatched1<VertexType,int64_t>(), false, VertexType());
    //unmatchedCol  = EWiseApply<VertexType>(unmatchedCol, mateCol2Row, sel1st<VertexType, int64_t>(), unmatched1<VertexType,int64_t>(), false, VertexType());
    //fringeCol = unmatchedCol;
    //cout << fringeCol[1];
    

    // FullyDistSpVec<int64_t, VertexType> fringeCol;
    //fringeCol.SetElement(0, VertexType(0,-1));

    //FullyDistSpVec<int64_t, VertexType> fringeCol;
    //fringeCol.SetElement(0, VertexType());

    
    //PSpMat_Int64  A = Aeff;
    //A.PrintInfo();
    //SpMV<SelectRandSRing>(A, fringeCol, fringeCol, false);
    
    //SpMV<SelectMinSRing2>(A, fringeCol, fringeCol, false);
    
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






