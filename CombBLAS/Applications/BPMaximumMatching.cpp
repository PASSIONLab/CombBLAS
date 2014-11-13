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
//#include "PothenFan.cpp"
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
#define EDGEFACTOR 8
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



struct VertexType
{
public:
	VertexType(){parent=-1; root = -1; prob = 1;};
    VertexType(int64_t p){parent=p; root=-1; prob = 1;}; // this constructor is called when we assign vertextype=number. Called from ApplyInd function
	VertexType(int64_t p, int64_t r){parent=p; root = r; prob = 1;};
    VertexType(int64_t p, int64_t r, int16_t pr){parent=p; root = r; prob = pr;};
    
    friend bool operator==(const VertexType & vtx1, const VertexType & vtx2 ){return vtx1.parent==vtx2.parent;};
    friend ostream& operator<<(ostream& os, const VertexType & vertex ){os << "(" << vertex.parent << "," << vertex.root << ")"; return os;};
    //private:
    int64_t parent;
    int64_t root;
    int16_t prob; // probability of selecting an edge

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



// Selct parent at random
struct SelectRandSRing
{
    static MPI_Op MPI_BFSRAND;
	typedef int64_t T_promote;
	static VertexType id(){ return VertexType(); };
	static bool returnedSAID() { return false; }
	//static MPI_Op mpi_op() { return MPI_MAX; }; // do we need this?
    
	static VertexType add(const VertexType & arg1, const VertexType & arg2)
	{
        if(arg1.prob < arg2.prob) return arg1;
        else return arg2;
	}
    
	static VertexType multiply(const T_promote & arg1, const VertexType & arg2)
	{
        return VertexType(arg2.parent, arg2.root, arg1); // think if we can use arg2.prob for a better prediction. may be based on degree
	}
    
    static void axpy(T_promote a, const VertexType & x, VertexType & y)
    {
        y = add(y, multiply(a, x));
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
        //cout << arg1 << " + " << arg2 << endl;
		if(arg1.parent < arg2.parent)
            return arg1;
        else
            return arg2;
	}
    
	static VertexType multiply(const T_promote & arg1, const VertexType & arg2)
	{
        //cout << arg1 << " * " << arg2 << endl;
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
typedef SpParMat < int64_t, float, SpDCCols<int64_t,float> > PSpMat_float;
void greedyMatching(PSpMat_Int64 & A, FullyDistVec<int64_t, int64_t>& mateRow2Col,
                    FullyDistVec<int64_t, int64_t>& mateCol2Row);
void maximumMatching(PSpMat_Int64 & Aeff, FullyDistVec<int64_t, int64_t>& mateRow2Col,
                     FullyDistVec<int64_t, int64_t>& mateCol2Row);
void maximumMatchingSimple(PSpMat_Bool & Aeff);
template <class IT, class NT>
bool isMaximalmatching(PSpMat_Int64 & A, FullyDistVec<IT,NT> & mateRow2Col, FullyDistVec<IT,NT> & mateCol2Row,
                       FullyDistSpVec<int64_t, int64_t> unmatchedRow, FullyDistSpVec<int64_t, int64_t> unmatchedCol);

/*
 Remove isolated vertices and purmute
 */
void removeIsolated(PSpMat_Int64 & A)
{
    FullyDistVec<int64_t, int64_t> * ColSums = new FullyDistVec<int64_t, int64_t>(A.getcommgrid());
    FullyDistVec<int64_t, int64_t> * RowSums = new FullyDistVec<int64_t, int64_t>(A.getcommgrid());
    FullyDistVec<int64_t, int64_t> nonisoRowV;	// id's of non-isolated (connected) Row vertices
    FullyDistVec<int64_t, int64_t> nonisoColV;	// id's of non-isolated (connected) Col vertices
    FullyDistVec<int64_t, int64_t> nonisov;	// id's of non-isolated (connected) vertices
    
    A.Reduce(*ColSums, Column, plus<int64_t>(), static_cast<int64_t>(0));
    A.Reduce(*RowSums, Row, plus<int64_t>(), static_cast<int64_t>(0));
    //ColSums->EWiseApply(*RowSums, plus<int64_t>());
    nonisov = ColSums->FindInds(bind2nd(greater<int64_t>(), 0));
    nonisoColV = ColSums->FindInds(bind2nd(greater<int64_t>(), 0));
    nonisoRowV = RowSums->FindInds(bind2nd(greater<int64_t>(), 0));
    //nonisoColV.iota(A.getncol(), 0);
    nonisov.RandPerm();	// so that A(v,v) is load-balanced (both memory and time wise)
    nonisoColV.RandPerm();
    nonisoRowV.RandPerm();
    
    delete ColSums;
    delete RowSums;
    
    //A(nonisoColV, nonisoColV, true);	// in-place permute to save memory
    A.operator()(nonisoRowV, nonisoColV, true);
    //A.PrintInfo();

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
            //cout << "Usage: ./bpmm <Scale> <Init> <Permute>" << endl;
			cout << "Example: mpirun -np 4 ./bpmm 20 1 (optional, 1=init, 0=dont init) 1 (optional, 1=permute 0=nopermute)" << endl;
		}
		MPI_Finalize();
		return -1;
	}		
	{
		
		unsigned scale;
		scale = static_cast<unsigned>(atoi(argv[1]));
        //double initiator[4] = {.57, .19, .19, .05};
        double initiator[4] = {.25, .25, .25, .25};

        
        DistEdgeList<int64_t> * DEL = new DistEdgeList<int64_t>();
        DEL->GenGraph500Data(initiator, scale, EDGEFACTOR, true, false ); // if packed=true the function does not use initiator
        MPI_Barrier(MPI_COMM_WORLD);
        
        PSpMat_Bool * ABool = new PSpMat_Bool(*DEL, false);
        delete DEL;
        //int64_t removed  = ABool->RemoveLoops(); // loop means an edges (i,i+NU) in a bipartite graph
        
       
        // remove isolated vertice if necessary
        
        //RandomParentBFS(*ABool);
        //greedyMatching(*ABool);
        // maximumMatching(*ABool);
        //maximumMatchingSimple(*ABool);
        
        
        
        
        //PSpMat_Bool A1;
        //A1.ReadDistribute("amazon0312.mtx", 0);	// read it from file
        //A1.ReadDistribute("coPapersDBLP.mtx", 0);	// read it from file
        //A1.PrintInfo();
        
        
        
        //////
        /*
        // Remove Isolated vertice
        FullyDistVec<int64_t, int64_t> * ColSums = new FullyDistVec<int64_t, int64_t>(A.getcommgrid());
        FullyDistVec<int64_t, int64_t> * RowSums = new FullyDistVec<int64_t, int64_t>(A.getcommgrid());
        FullyDistVec<int64_t, int64_t> nonisoRowV;	// id's of non-isolated (connected) Row vertices
        FullyDistVec<int64_t, int64_t> nonisoColV;	// id's of non-isolated (connected) Col vertices
        FullyDistVec<int64_t, int64_t> nonisov;	// id's of non-isolated (connected) vertices
        
        A1.Reduce(*ColSums, Column, plus<int64_t>(), static_cast<int64_t>(0));
        A1.Reduce(*RowSums, Row, plus<int64_t>(), static_cast<int64_t>(0));
        //ColSums->EWiseApply(*RowSums, plus<int64_t>());
        nonisov = ColSums->FindInds(bind2nd(greater<int64_t>(), 0));
        nonisoColV = ColSums->FindInds(bind2nd(greater<int64_t>(), 0));
        nonisoRowV = RowSums->FindInds(bind2nd(greater<int64_t>(), 0));
        //nonisoColV.iota(A.getncol(), 0);
        nonisov.RandPerm();	// so that A(v,v) is load-balanced (both memory and time wise)
        nonisoColV.RandPerm();
        nonisoRowV.RandPerm();
        
        delete ColSums;
        delete RowSums;
        
        //A(nonisoColV, nonisoColV, true);	// in-place permute to save memory
        A.operator()(nonisoRowV, nonisoColV, true);
        /////
        */
        
        
        
        
        PSpMat_Int64  A = *ABool;
        if(argc>=4 && static_cast<unsigned>(atoi(argv[3]))==1)
            removeIsolated(A);
        
        
        FullyDistVec<int64_t, int64_t> mateRow2Col ( A.getcommgrid(), A.getnrow(), (int64_t) -1);
        FullyDistVec<int64_t, int64_t> mateCol2Row ( A.getcommgrid(), A.getncol(), (int64_t) -1);
        if(argc>=3 && static_cast<unsigned>(atoi(argv[2]))==1)
            greedyMatching(A, mateRow2Col, mateCol2Row);
        

        A.Apply([](int64_t x){return static_cast<int64_t>(GlobalMT.rand() * 10000);}); // perform randomization
        
        
        //A1.Transpose();
        //varify_matching(*ABool);
        maximumMatching(A, mateRow2Col, mateCol2Row);
        
        //mateRow2Col.DebugPrint();
        
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



void Augment1(FullyDistVec<int64_t, int64_t>& mateRow2Col, FullyDistVec<int64_t, int64_t>& mateCol2Row,
             FullyDistVec<int64_t, int64_t>& parentsRow, FullyDistVec<int64_t, int64_t>& leaves)
{
 
    int64_t nrow = mateRow2Col.TotalLength();
    int64_t ncol = mateCol2Row.TotalLength();
    FullyDistSpVec<int64_t, int64_t> col(leaves, [](int64_t leaf){return leaf!=-1;});
    FullyDistSpVec<int64_t, int64_t> row(mateRow2Col.getcommgrid(), nrow);
    FullyDistSpVec<int64_t, int64_t> nextcol(col.getcommgrid(), ncol);
 
    while(col.getnnz()!=0)
    {
     
        row = col.Invert(nrow);
        
        row.SelectApply(parentsRow, [](int64_t parent){return true;},
                        [](int64_t root, int64_t parent){return parent;}); // this is a Set operation
        

        col = row.Invert(ncol); // children array

        nextcol = col.SelectApplyNew(mateCol2Row, [](int64_t mate){return mate!=-1;}, [](int64_t child, int64_t mate){return mate;});
        
        mateRow2Col.Set(row);
        mateCol2Row.Set(col);
        col = nextcol;
        
    }
}



template <typename IT, typename NT>
void Augment(FullyDistVec<int64_t, int64_t>& mateRow2Col, FullyDistVec<int64_t, int64_t>& mateCol2Row,
             FullyDistVec<int64_t, int64_t>& parentsRow, FullyDistVec<int64_t, int64_t>& leaves)
{

    MPI_Win win_mateRow2Col, win_mateCol2Row, win_parentsRow;
    MPI_Win_create(&mateRow2Col.arr[0], mateRow2Col.LocArrSize() * sizeof(int64_t), sizeof(int64_t), MPI_INFO_NULL, mateRow2Col.commGrid->GetWorld(), &win_mateRow2Col);
    MPI_Win_create(&mateCol2Row.arr[0], mateCol2Row.LocArrSize() * sizeof(int64_t), sizeof(int64_t), MPI_INFO_NULL, mateCol2Row.commGrid->GetWorld(), &win_mateCol2Row);
    MPI_Win_create(&parentsRow.arr[0], parentsRow.LocArrSize() * sizeof(int64_t), sizeof(int64_t), MPI_INFO_NULL, parentsRow.commGrid->GetWorld(), &win_parentsRow);

    //cout<< "Leaves: " ;
    //leaves.DebugPrint();
    //parentsRow.DebugPrint();
    
    MPI_Win_fence(0, win_mateRow2Col);
    MPI_Win_fence(0, win_mateCol2Row);
    MPI_Win_fence(0, win_parentsRow);

    int64_t row, col=100, nextrow;
    int owner_row, owner_col;
    IT locind_row, locind_col;
    int myrank;
    
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    
    //IT i=1;
    for(IT i=0; i<leaves.LocArrSize(); i++)
    {
        int depth=0;
        row = leaves.arr[i];
        while(row != - 1)
        {
            
            owner_row = mateRow2Col.Owner(row, locind_row);
            //cout << myrank << ": " << row << " " << owner_row << " " << locind_row << "@@@\n";
            MPI_Win_lock(MPI_LOCK_SHARED, owner_row, 0, win_parentsRow);
            MPI_Get(&col, 1, MPIType<NT>(), owner_row, locind_row, 1, MPIType<NT>(), win_parentsRow);
            MPI_Win_unlock(owner_row, win_parentsRow);
            
            owner_col = mateCol2Row.Owner(col, locind_col);
            //cout <<  myrank << ": " << col << " " << owner_col << " " << locind_col << "!!!\n";
            
            MPI_Win_lock(MPI_LOCK_SHARED, owner_col, 0, win_mateCol2Row);
            MPI_Fetch_and_op(&row, &nextrow, MPIType<NT>(), owner_col, locind_col, MPI_REPLACE, win_mateCol2Row);
            
            //MPI_Get(&nextrow, 1, MPIType<NT>(), owner_col, locind_col, 1, MPIType<NT>(), win_mateCol2Row);
            //MPI_Put(&row, 1, MPIType<NT>(), owner_col, locind_col, 1, MPIType<NT>(), win_mateCol2Row);
            MPI_Win_unlock(owner_col, win_mateCol2Row);
            
            MPI_Win_lock(MPI_LOCK_SHARED, owner_row, 0, win_mateRow2Col);
            MPI_Put(&col, 1, MPIType<NT>(), owner_row, locind_row, 1, MPIType<NT>(), win_mateRow2Col);
            MPI_Win_unlock(owner_row, win_mateRow2Col); // we need this otherwise col might get overwritten before communication!
            //depth++;
            //if(depth>5) {cout << "depth--------------\n";break;}
            
            row = nextrow;
            
        }
    }
    

    MPI_Win_fence(0, win_mateRow2Col);
    MPI_Win_fence(0, win_mateCol2Row);
    MPI_Win_fence(0, win_parentsRow);
    
    MPI_Win_free(&win_mateRow2Col);
    MPI_Win_free(&win_mateCol2Row);
    MPI_Win_free(&win_parentsRow);
    
    //mateCol2Row.DebugPrint();
    //mateRow2Col.DebugPrint();
}




void maximumMatching(PSpMat_Int64 & A, FullyDistVec<int64_t, int64_t>& mateRow2Col,
                     FullyDistVec<int64_t, int64_t>& mateCol2Row)
{
    
    int nprocs, myrank;
	MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    
    int64_t nrow = A.getnrow();
    int64_t ncol = A.getncol();
    //FullyDistSpVec<int64_t, VertexType> unmatchedCol(A.getcommgrid(), ncol);
    
    FullyDistSpVec<int64_t, VertexType> temp(A.getcommgrid(), ncol);
    FullyDistSpVec<int64_t, int64_t> temp1(A.getcommgrid(), ncol);
    FullyDistSpVec<int64_t, VertexType> fringeRow(A.getcommgrid(), nrow);
    FullyDistSpVec<int64_t, VertexType> umFringeRow(A.getcommgrid(), nrow);
    FullyDistSpVec<int64_t, int64_t> umFringeRow1(A.getcommgrid(), nrow);
    
    
    vector<vector<double> > timing;
    double t1, time_search, time_augment, time_phase;
   
    bool matched = true;
    int phase = 0;
    int totalLayer = 0;
    int64_t numUnmatchedCol;
    
    while(matched)
    {
        time_phase = MPI_Wtime();
        vector<double> phase_timing(9,0);
        FullyDistVec<int64_t, int64_t> leaves ( A.getcommgrid(), nrow, (int64_t) -1);
        FullyDistVec<int64_t, int64_t> parentsRow ( A.getcommgrid(), nrow, (int64_t) -1); // it needs to be cleared after each phase
        
        FullyDistVec<int64_t, int64_t> rootsRow ( A.getcommgrid(), nrow, (int64_t) -1); // just for test
        
        FullyDistSpVec<int64_t, VertexType> fringeCol(A.getcommgrid(), ncol);
        fringeCol  = EWiseApply<VertexType>(fringeCol, mateCol2Row, select1st<VertexType, int64_t>(),
                                            unmatched_binary<VertexType,int64_t>(), true, VertexType()); // root & parent both =-1
        fringeCol.ApplyInd([](VertexType vtx, int64_t idx){return VertexType(idx,idx);}); //  root & parent both equal to index
        //fringeCol.DebugPrint();
        
        ++phase;
        numUnmatchedCol = fringeCol.getnnz();
        int64_t tt;
        int layer = 0;
        
         double test1=0, test2=0;
        
        time_search = MPI_Wtime();
        while(fringeCol.getnnz() > 0)
        {
            layer++;
            t1 = MPI_Wtime();
            SpMV<SelectMinSRing2>(A, fringeCol, fringeRow, false);
            //SpMV<SelectRandSRing>(A, fringeCol, fringeRow, false);
            phase_timing[0] += MPI_Wtime()-t1;
            
            
            
            // remove vertices already having parents
            t1 = MPI_Wtime();
            fringeRow.Select(parentsRow, [](int64_t parent){return parent==-1;});
           
            
            // Set parent pointer
            // TODO: Write a general purpose FullyDistVec::Set
            t1 = MPI_Wtime();
            parentsRow.EWiseApply(fringeRow,
                                  [](int64_t dval, VertexType svtx, bool a, bool b){return svtx.parent;}, // return parent of the sparse vertex
                                  [](int64_t dval, VertexType svtx, bool a, bool b){return true;}, //always true; why do we have to pass the bools?
                                  false, VertexType(), false);
            
            
            
            //get unmatched row vertices
            t1 = MPI_Wtime();
            //umFringeRow = fringeRow.SelectNew(mateRow2Col, [](int64_t mate){return mate==-1;});
            umFringeRow1 = fringeRow.SelectNew1(mateRow2Col, [](int64_t mate){return mate==-1;}, [](VertexType& vtx){return vtx.root;});
            //umFringeRow1 = fringeRow.SelectApplyNew(mateRow2Col, [](int64_t mate){return mate==-1;},
            //                                        [](VertexType& vtx, int64_t mate){return vtx.root;});
            
            //cout << "umFringeRow1: ";
            //if(umFringeRow1.getnnz()>0)umFringeRow1.DebugPrint();
            phase_timing[1] += MPI_Wtime()-t1;
            t1 = MPI_Wtime();
            tt = umFringeRow1.getnnz();
            // get the unique leaves
            //MPI_Pcontrol(1,"Compose");
            if(umFringeRow1.getnnz()>0)
            {
                //temp = umFringeRow.Compose1(ncol,
                //                           [](VertexType& vtx, const int64_t & index){return vtx.root;}, // index is the root
                //                           [](VertexType& vtx, const int64_t & index){return VertexType(index, vtx.root);}); // value is the leaf
                
                //temp1 = umFringeRow1.Compose1(ncol,
                //                          [](VertexType& vtx, const int64_t & index){return vtx.root;}, // index is the root
                //                        [](VertexType& vtx, const int64_t & index){return index;}); // value is the leaf
                //temp1 = umFringeRow1.Invert
                /*temp1 = umFringeRow1.ComposeRMA(ncol,
                                            [](int64_t& val, const int64_t& index){return val;}, // index is the val
                                             [](int64_t& val, const int64_t& index){return index;}); // val is the index
                */
                temp1 = umFringeRow1.Invert(ncol);
                //temp1.DebugPrint();
            }
            
            //MPI_Pcontrol(-1,"Compose");
            phase_timing[2] += MPI_Wtime()-t1;
            
            //set leaf pointer
            t1 = MPI_Wtime();
            if(umFringeRow1.getnnz()>0)
            {
                
                //leaves.EWiseApply(temp,
                //             [](int64_t dval, VertexType svtx, bool a, bool b){return svtx.parent;}, // return parent of the sparse vertex
                //           [](int64_t dval, VertexType svtx, bool a, bool b){return dval==-1;}, //if no aug path is already found
                //         false, VertexType(), false);
                
                leaves.Set(temp1);
                
            }
            phase_timing[3] += MPI_Wtime()-t1;
            
            
            t1 = MPI_Wtime();
            fringeRow.SelectApply(mateRow2Col, [](int64_t mate){return mate!=-1;},
                                  [](VertexType vtx, int64_t mate){return VertexType(mate, vtx.root);});
            phase_timing[4] += MPI_Wtime()-t1;
            
            //if(temp1.getnnz() > 0)temp1.DebugPrint();
            
            //cout << temp1.getnnz() << " : " << fringeRow.getnnz() << " : ";
            //if(fringeRow.getnnz()>0)fringeRow.DebugPrint();
            //if(temp1.getnnz()>0 )
             //   fringeRow.FilterByVal (temp1,[](VertexType vtx){return vtx.root;});
            //if(fringeRow.getnnz()>0)fringeRow.DebugPrint();
            //cout << fringeRow.getnnz() << " \n";
            
            
            t1 = MPI_Wtime();
            
            if(fringeRow.getnnz() > 0)
            {
                //fringeRow.DebugPrint();
                // looks like we need fringeCol sorted!!
                
                FullyDistSpVec<int64_t, VertexType> fringeCol1(fringeCol);
                
                double t2 = MPI_Wtime();
                fringeCol1 = fringeRow.Compose(ncol,
                                              [](VertexType& vtx, const int64_t & index){return vtx.parent;}, // index is the parent (mate)
                                              [](VertexType& vtx, const int64_t & index){return VertexType(vtx.parent, vtx.root);}); // value
                
                test1 += MPI_Wtime()-t2;
                t2 = MPI_Wtime();
                //MPI_Abort(MPI_COMM_WORLD,-1);
                // I think this is only better for long paths / small number of vertices
                
                fringeCol = fringeRow.ComposeRMA(ncol,
                                              [](VertexType& vtx, const int64_t & index){return vtx.parent;}, // index is the parent (mate)
                                              [](VertexType& vtx, const int64_t & index){return VertexType(vtx.parent, vtx.root);}); // value
                
                test2 += MPI_Wtime()-t2;
                
            }
            else break;
            
            phase_timing[5] += MPI_Wtime()-t1;
            // TODO:do something for prunning
            
            
        }
        time_search = MPI_Wtime() - time_search;
        phase_timing[6] += time_search;

      
        int64_t numMatchedCol = leaves.Count([](int64_t leaf){return leaf!=-1;});
        time_augment = MPI_Wtime();
        if (numMatchedCol== 0) matched = false;
        else
        {
            Augment<int64_t,int64_t>(mateRow2Col, mateCol2Row,parentsRow, leaves);
            //Augment1(mateRow2Col, mateCol2Row,parentsRow, leaves);
        }
        time_augment = MPI_Wtime() - time_augment;
        phase_timing[7] += time_augment;
        
        
        time_phase = MPI_Wtime() - time_phase;
        phase_timing[8] += time_phase;
        timing.push_back(phase_timing);
        
        ostringstream tinfo;
        tinfo << "Phase: " << phase << " layers:" << layer << " Unmatched Columns: " << numUnmatchedCol << " Matched: " << numMatchedCol << " Time: "<< time_phase << " ::: "  << test1 << " , " << test2 << "\n";
        SpParHelper::Print(tinfo.str());
        //if(phase==2)break;
        totalLayer += layer;
        
    }
    
    
    
    
    //mateCol2Row.DebugPrint();
    //mateRow2Col.DebugPrint();

    
    
    //isMaximalmatching(A, mateRow2Col, mateCol2Row, unmatchedRow, unmatchedCol);
    //isMatching(mateCol2Row, mateRow2Col); //todo there is a better way to check this
    
    
    // print statistics
    if(myrank == 0)
    {
        cout << endl;
        cout << "===================== ========================================= ==============\n";
        cout << "                                        BFS Search               Aug    \n";
        cout << "===================== ========================================= ======= ======\n";
        cout  << "Phase Layer    UMCol   SpMV EWOpp CmUqL EWSetL EWMR CmMC  BFS           Total\n";
        cout << "===================== ========================================= ==============\n";
        
        vector<double> totalTimes(timing[0].size(),0);
        int nphases = timing.size();
        for(int i=0; i<timing.size(); i++)
        {
            //printf(" %3d   ", i+1);
            for(int j=0; j<timing[i].size(); j++)
            {
                totalTimes[j] += timing[i][j];
                //timing[i][j] /= timing[i].back();
                //printf("%.2lf  ", timing[i][j]);
            }
            
            //printf("\n");
        }
        
        double combTime = totalTimes.back();
        printf(" %3d  %3d  %8lld   ", nphases, totalLayer/nphases, numUnmatchedCol);
        for(int j=0; j<totalTimes.size()-1; j++)
        {
            printf("%.2lf  ", totalTimes[j]);
        }
        printf("%.2lf\n", combTime);
        
        //cout << "=================== total timing ===========================\n";
        //for(int i=0; i<totalTimes.size(); i++)
        //    cout<<totalTimes[i] << " ";
        //cout << endl;
    }
    
    
    
}



void greedyMatching(PSpMat_Int64 & A, FullyDistVec<int64_t, int64_t>& mateRow2Col,
                    FullyDistVec<int64_t, int64_t>& mateCol2Row)
{
    
    int nprocs, myrank;
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    
    //PSpMat_Int64  A = Aeff;
    //A.Apply([](int64_t x){return static_cast<int64_t>(GlobalMT.rand() * 10000);}); // perform randomization
    //A.PrintInfo();
    
    
    //matching vector (dense)
    
    //unmatched row and column vertices
    FullyDistSpVec<int64_t, int64_t> unmatchedRow(mateRow2Col, [](int64_t mate){return mate==-1;});
    FullyDistSpVec<int64_t, int64_t> unmatchedCol(mateCol2Row, [](int64_t mate){return mate==-1;});
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
    
    //PSpMat_Int64  A = Aeff;
    //A.PrintInfo();
    
    while(curUnmatchedCol !=0 && curUnmatchedRow!=0 && newlyMatched != 0 )
    {
        vector<double> times;
        double t1 = MPI_Wtime();
        // step1: Find adjacent row vertices (col vertices parent, row vertices child)
        //fringeRow = SpMV(Aeff, unmatchedCol, optbuf);
        //SpMV<SelectMinSRing1>(Aeff, unmatchedCol, fringeRow, false, optbuf);
        SpMV<SelectMinSRing1>(A, unmatchedCol, fringeRow, false);
        
        
        // step2: Remove matched row vertices
        fringeRow.Select(mateRow2Col, [](int64_t mate){return mate==-1;});
        if(myrank == 0){times.push_back(MPI_Wtime()-t1); t1 = MPI_Wtime();}
        
        
        // step3: Remove duplicate row vertices
        //fringeRow = fringeRow.Uniq();
        //if(myrank == 0){times.push_back(MPI_Wtime()-t1); t1 = MPI_Wtime();}
        //fringeRow.DebugPrint();
        
        // step4: Update mateRow2Col with the newly matched row vertices
        //mateRow2Col.Set(fringeRow);
        //fringeRow.DebugPrint();
        
        // step5: Update mateCol2Row with the newly matched col vertices
        FullyDistSpVec<int64_t, int64_t> temp = fringeRow.Invert(A.getncol());
        mateCol2Row.Set(temp);
        //temp.DebugPrint();
        if(myrank == 0){times.push_back(MPI_Wtime()-t1); t1 = MPI_Wtime();}
        
        FullyDistSpVec<int64_t, int64_t> temp1 = temp.Invert(A.getnrow());
        
        mateRow2Col.Set(temp1);
        //temp1.DebugPrint();
        if(myrank == 0){times.push_back(MPI_Wtime()-t1); t1 = MPI_Wtime();}
        
        // step6: Update unmatchedCol/unmatchedRow by removing newly matched columns/rows
        //unmatchedCol = EWiseMult(unmatchedCol, mateCol2Row, true, (int64_t) -1);
        //unmatchedRow = EWiseMult(unmatchedRow, mateRow2Col, true, (int64_t) -1);
        
        unmatchedRow.Select(mateRow2Col, [](int64_t mate){return mate==-1;});
        unmatchedCol.Select(mateCol2Row, [](int64_t mate){return mate==-1;});
        
        if(myrank == 0){times.push_back(MPI_Wtime()-t1); t1 = MPI_Wtime();}
        
        
        ++iteration;
        newlyMatched = temp.getnnz();
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
    //isMaximalmatching(A, mateRow2Col, mateCol2Row, unmatchedRow, unmatchedCol);
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




void maximumMatching1(PSpMat_Bool & Aeff)
{
    
    int nprocs, myrank;
	MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    
    PSpMat_Int64  A = Aeff;
    //PSpMat_float A = Aeff;
    A.Apply([](int64_t x){return static_cast<int64_t>(GlobalMT.rand() * 10000);}); // perform randomization
    A.PrintInfo();
    
    //matching vector (dense)
    FullyDistVec<int64_t, int64_t> mateRow2Col ( Aeff.getcommgrid(), Aeff.getnrow(), (int64_t) -1);
    FullyDistVec<int64_t, int64_t> mateCol2Row ( Aeff.getcommgrid(), Aeff.getncol(), (int64_t) -1);
    //mateCol2Row.DebugPrint();
    
    //FullyDistSpVec<int64_t, VertexType> unmatchedCol(Aeff.getcommgrid(), Aeff.getncol());
    
    FullyDistSpVec<int64_t, VertexType> temp(Aeff.getcommgrid(), Aeff.getncol());
    FullyDistSpVec<int64_t, int64_t> temp1(Aeff.getcommgrid(), Aeff.getncol());
    FullyDistSpVec<int64_t, VertexType> fringeRow(Aeff.getcommgrid(), Aeff.getnrow());
    FullyDistSpVec<int64_t, VertexType> umFringeRow(Aeff.getcommgrid(), Aeff.getnrow());
    FullyDistSpVec<int64_t, int64_t> umFringeRow1(Aeff.getcommgrid(), Aeff.getnrow());
    
    
    vector<vector<double> > timing;
    double t1, time_search, time_augment, time_phase;
    bool matched = true;
    int phase = 0;
    
    while(matched)
    {
        time_phase = MPI_Wtime();
        vector<double> phase_timing(16,0);
        FullyDistVec<int64_t, int64_t> leaves ( Aeff.getcommgrid(), Aeff.getnrow(), (int64_t) -1);
        FullyDistVec<int64_t, int64_t> parentsRow ( Aeff.getcommgrid(), Aeff.getnrow(), (int64_t) -1); // it needs to be cleared after each phase
        
        FullyDistVec<int64_t, int64_t> rootsRow ( Aeff.getcommgrid(), Aeff.getnrow(), (int64_t) -1); // just for test
        
        FullyDistSpVec<int64_t, VertexType> fringeCol(Aeff.getcommgrid(), Aeff.getncol());
        fringeCol  = EWiseApply<VertexType>(fringeCol, mateCol2Row, select1st<VertexType, int64_t>(),
                                            unmatched_binary<VertexType,int64_t>(), true, VertexType()); // root & parent both =-1
        fringeCol.ApplyInd([](VertexType vtx, int64_t idx){return VertexType(idx,idx);}); //  root & parent both equal to index
        //fringeCol.DebugPrint();
        
        ++phase;
        int64_t numUnmatchedCol = fringeCol.getnnz();
        int64_t tt;
        int layer = 0;
        time_search = MPI_Wtime();
        while(fringeCol.getnnz() > 0)
        {
            layer++;
            t1 = MPI_Wtime();
            //SpMV<SelectMinSRing2>(A, fringeCol, fringeRow, false);
            SpMV<SelectRandSRing>(A, fringeCol, fringeRow, false);
            phase_timing[0] += MPI_Wtime()-t1;
            
            // remove vertices already having parents
            t1 = MPI_Wtime();
            fringeRow.Select(parentsRow, [](int64_t parent){return parent==-1;});
            phase_timing[1] += MPI_Wtime()-t1;
            
            // Set parent pointer
            // TODO: Write a general purpose FullyDistVec::Set
            t1 = MPI_Wtime();
            parentsRow.EWiseApply(fringeRow,
                                  [](int64_t dval, VertexType svtx, bool a, bool b){return svtx.parent;}, // return parent of the sparse vertex
                                  [](int64_t dval, VertexType svtx, bool a, bool b){return true;}, //always true; why do we have to pass the bools?
                                  false, VertexType(), false);
            rootsRow.EWiseApply(fringeRow,
                                  [](int64_t dval, VertexType svtx, bool a, bool b){return svtx.root;}, // return parent of the sparse vertex
                                  [](int64_t dval, VertexType svtx, bool a, bool b){return true;}, //always true; why do we have to pass the bools?
                                  false, VertexType(), false); // just for testing
            phase_timing[2] += MPI_Wtime()-t1;
            
            //if(fringeCol.getnnz() > 0)fringeCol.DebugPrint();
            //if(fringeRow.getnnz() > 0)fringeRow.DebugPrint();
            
            
            
            //get unmatched row vertices
            t1 = MPI_Wtime();
            //umFringeRow = fringeRow.SelectNew(mateRow2Col, [](int64_t mate){return mate==-1;});
            umFringeRow1 = fringeRow.SelectNew1(mateRow2Col, [](int64_t mate){return mate==-1;}, [](VertexType& vtx){return vtx.root;});
            phase_timing[3] += MPI_Wtime()-t1;
            t1 = MPI_Wtime();
            tt = umFringeRow1.getnnz();
            // get the unique leaves
            //MPI_Pcontrol(1,"Compose");
            if(umFringeRow1.getnnz()>0)
            {
                //temp = umFringeRow.Compose1(Aeff.getncol(),
                //                           [](VertexType& vtx, const int64_t & index){return vtx.root;}, // index is the root
                //                           [](VertexType& vtx, const int64_t & index){return VertexType(index, vtx.root);}); // value is the leaf
                
                //temp1 = umFringeRow1.Compose1(Aeff.getncol(),
                  //                          [](VertexType& vtx, const int64_t & index){return vtx.root;}, // index is the root
                    //                        [](VertexType& vtx, const int64_t & index){return index;}); // value is the leaf
                //temp1 = umFringeRow1.Invert
                temp1 = umFringeRow1.Compose1(Aeff.getncol(),
                                   [](int64_t val, const int64_t index){return val;}, // index is the val
                                   [](int64_t val, const int64_t index){return index;}); // val is the index
            }
            
            //MPI_Pcontrol(-1,"Compose");
            phase_timing[4] += MPI_Wtime()-t1;
            
            //set leaf pointer
            t1 = MPI_Wtime();
            if(umFringeRow1.getnnz()>0)
            {
                
                //leaves.EWiseApply(temp,
                 //             [](int64_t dval, VertexType svtx, bool a, bool b){return svtx.parent;}, // return parent of the sparse vertex
                   //           [](int64_t dval, VertexType svtx, bool a, bool b){return dval==-1;}, //if no aug path is already found
                     //         false, VertexType(), false);
                
                leaves.Set(temp1);
                
            }
            phase_timing[5] += MPI_Wtime()-t1;
            
            
            t1 = MPI_Wtime();
            fringeRow.SelectApply(mateRow2Col, [](int64_t mate){return mate!=-1;},
                                  [](VertexType vtx, int64_t mate){return VertexType(mate, vtx.root);});
            phase_timing[6] += MPI_Wtime()-t1;
            
            if(temp1.getnnz() > 0)temp1.DebugPrint();
            
            cout << temp1.getnnz() << " : " << fringeRow.getnnz() << " : ";
            if(temp1.getnnz()>0)
                fringeRow.FilterByVal (temp1,[](VertexType vtx){return vtx.root;});
            cout << fringeRow.getnnz() << " \n";

  
            t1 = MPI_Wtime();
            // looks like we need fringeCol sorted!!
            fringeCol = fringeRow.Compose(Aeff.getncol(),
                                          [](VertexType& vtx, const int64_t & index){return vtx.parent;}, // index is the parent (mate)
                                          [](VertexType& vtx, const int64_t & index){return VertexType(vtx.parent, vtx.root);}); // value
            phase_timing[7] += MPI_Wtime()-t1;
            // TODO:do something for prunning
            
            
        }
        time_search = MPI_Wtime() - time_search;
        phase_timing[8] += time_search;
        //leaves.DebugPrint();
        //rootsRow.DebugPrint();
        
        // create a sparse vector of leaves
        FullyDistSpVec<int64_t, int64_t> col(leaves, [](int64_t leaf){return leaf!=-1;});
        FullyDistSpVec<int64_t, int64_t> row(A.getcommgrid(), A.getnrow());
        FullyDistSpVec<int64_t, int64_t> nextcol(A.getcommgrid(), A.getncol());
        
        int64_t numMatchedCol = col.getnnz();
        if (col.getnnz()== 0) matched = false;
        
        // Augment
        time_augment = MPI_Wtime();
        while(col.getnnz()!=0)
        {
            t1 = MPI_Wtime();
            row = col.Compose1(Aeff.getncol(),
                                        [](int64_t val, const int64_t index){return val;}, // index is the val
                                        [](int64_t val, const int64_t index){return index;}); // val is the index
            //row = col.Invert(A.getnrow());
            phase_timing[9] += MPI_Wtime()-t1;
            // Set parent pointer
            // TODO: Write a general purpose FullyDistSpVec::Set based on a FullyDistVec
            t1 = MPI_Wtime();
            /*
             row = EWiseApply<int64_t>(row, parentsRow,
             [](int64_t root, int64_t parent){return parent;},
             [](int64_t root, int64_t parent){return true;}, // must have a root
             false, (int64_t) -1);
             */
            row.SelectApply(parentsRow, [](int64_t parent){return true;},
                            [](int64_t root, int64_t parent){return parent;}); // this is a Set operation
            
            phase_timing[10] += MPI_Wtime()-t1;
            //if(row.getnnz()!=0)row.DebugPrint();
            
            t1 = MPI_Wtime();
            //col = row.Invert(A.getncol()); // children array
            col = row.Compose1(Aeff.getncol(),
                               [](int64_t val, const int64_t index){return val;}, // index is the val
                               [](int64_t val, const int64_t index){return index;}); // val is the index

            phase_timing[11] += MPI_Wtime()-t1;
            
            t1 = MPI_Wtime();
            /*
            nextcol = EWiseApply<int64_t>(col, mateCol2Row,
                                          [](int64_t child, int64_t mate){return mate;},
                                          [](int64_t child, int64_t mate){return mate!=-1;}, // mate==-1 when we have reached to the root
                                          false, (int64_t) -1);
             */
            nextcol = col.SelectApplyNew(mateCol2Row, [](int64_t mate){return mate!=-1;}, [](int64_t child, int64_t mate){return mate;});
            phase_timing[12] += MPI_Wtime()-t1;
            //col.DebugPrint();
            t1 = MPI_Wtime();
            mateRow2Col.Set(row);
            mateCol2Row.Set(col);
            phase_timing[13] += MPI_Wtime()-t1;
            col = nextcol;
            
        }
        
        //mateRow2Col.DebugPrint();
        //mateCol2Row.DebugPrint();
        
        time_augment = MPI_Wtime() - time_augment;
        phase_timing[14] += time_augment;
        time_phase = MPI_Wtime() - time_phase;
        phase_timing[15] += time_phase;
        timing.push_back(phase_timing);
        
        ostringstream tinfo;
        tinfo << "Phase: " << phase << " layers:" << layer << " Unmatched Columns: " << numUnmatchedCol << " Matched: " << numMatchedCol << " Time: "<< time_phase << " Comp: " <<phase_timing[4]<< " um: " << tt << "\n";
        SpParHelper::Print(tinfo.str());
        //break;
        
    }
    
    
    
    
    
    
    
    
    
    //isMaximalmatching(A, mateRow2Col, mateCol2Row, unmatchedRow, unmatchedCol);
    //isMatching(mateCol2Row, mateRow2Col); //todo there is a better way to check this
    
    
    // print statistics
    if(myrank == 0)
    {
        cout << endl;
        cout << "========================================================================================================\n";
        cout << "                         BFS Search                                        Augment    \n";
        cout << "============================================================ =================================== =======\n";
        cout  << "Phase  SpMV EWvis EWSetP EWUmR CmUqL EWSetL EWMR CmMC  BFS   Inv1   EW1  Inv2  EW2   SetM  Aug   Total \n";
        cout << "========================================================================================================\n";
        
        vector<double> totalTimes(timing[0].size(),0);
        int nphases = timing.size();
        for(int i=0; i<timing.size(); i++)
        {
            //printf(" %3d   ", i+1);
            for(int j=0; j<timing[i].size(); j++)
            {
                totalTimes[j] += timing[i][j];
                //timing[i][j] /= timing[i].back();
                //printf("%.2lf  ", timing[i][j]);
            }
            
            //printf("\n");
        }
        
        double combTime = totalTimes.back();
        printf(" %3d   ", nphases);
        for(int j=0; j<totalTimes.size()-1; j++)
        {
            printf("%.2lf  ", totalTimes[j]);
        }
        printf("%.2lf\n", combTime);
        
        //cout << "=================== total timing ===========================\n";
        //for(int i=0; i<totalTimes.size(); i++)
        //    cout<<totalTimes[i] << " ";
        //cout << endl;
    }
    
    
}





void maximumMatching_old(PSpMat_Bool & Aeff)
{
    
    int nprocs, myrank;
	MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    
    PSpMat_Int64  A = Aeff;
    A.PrintInfo();
    
    //matching vector (dense)
    FullyDistVec<int64_t, int64_t> mateRow2Col ( Aeff.getcommgrid(), Aeff.getnrow(), (int64_t) -1);
    FullyDistVec<int64_t, int64_t> mateCol2Row ( Aeff.getcommgrid(), Aeff.getncol(), (int64_t) -1);
    //mateCol2Row.DebugPrint();

    //FullyDistSpVec<int64_t, VertexType> unmatchedCol(Aeff.getcommgrid(), Aeff.getncol());
    
    FullyDistSpVec<int64_t, VertexType> temp(Aeff.getcommgrid(), Aeff.getncol());
    FullyDistSpVec<int64_t, VertexType> fringeRow(Aeff.getcommgrid(), Aeff.getnrow());
    FullyDistSpVec<int64_t, VertexType> umFringeRow(Aeff.getcommgrid(), Aeff.getnrow());
    
    
    vector<vector<double> > timing;
    double t1, time_search, time_augment, time_phase;
    bool matched = true;
    int phase = 0;
    
    while(matched)
    {
        time_phase = MPI_Wtime();
        vector<double> phase_timing(16,0);
        FullyDistVec<int64_t, int64_t> leaves ( Aeff.getcommgrid(), Aeff.getnrow(), (int64_t) -1);
        FullyDistVec<int64_t, int64_t> parentsRow ( Aeff.getcommgrid(), Aeff.getnrow(), (int64_t) -1); // it needs to be cleared after each phase
        FullyDistSpVec<int64_t, VertexType> fringeCol(Aeff.getcommgrid(), Aeff.getncol());
        fringeCol  = EWiseApply<VertexType>(fringeCol, mateCol2Row, select1st<VertexType, int64_t>(),
                                            unmatched_binary<VertexType,int64_t>(), true, VertexType()); // root & parent both =-1
        fringeCol.ApplyInd([](VertexType vtx, int64_t idx){return VertexType(idx,idx);}); //  root & parent both equal to index
        //fringeCol.DebugPrint();
        
        ++phase;
        int64_t numUnmatchedCol = fringeCol.getnnz();
        
        
        time_search = MPI_Wtime();
        while(fringeCol.getnnz() > 0)
        {
            t1 = MPI_Wtime();
            SpMV<SelectMinSRing2>(A, fringeCol, fringeRow, false);
            phase_timing[0] += MPI_Wtime()-t1;
            
            // remove vertices already having parents
            t1 = MPI_Wtime();
            //fringeRow  = EWiseApply<VertexType>(fringeRow, parentsRow, select1st<VertexType, int64_t>(), unmatched_binary<VertexType,int64_t>(), false, VertexType());
            fringeRow.Select(parentsRow, [](int64_t parent){return parent==-1;});
            /*fringeRow  = EWiseApply<VertexType>(fringeRow, parentsRow,
                                                  [](VertexType vtx, int64_t parent){return vtx;}, // return unvisited vertices
                                                  [](VertexType vtx, int64_t parent){return parent==-1;}, // select unvisited vertices
                                                  false, VertexType());
            */
             phase_timing[1] += MPI_Wtime()-t1;
            
            // Set parent pointer
            // TODO: Write a general purpose FullyDistVec::Set
            t1 = MPI_Wtime();
            parentsRow.EWiseApply(fringeRow,
                                  [](int64_t dval, VertexType svtx, bool a, bool b){return svtx.parent;}, // return parent of the sparse vertex
                                  [](int64_t dval, VertexType svtx, bool a, bool b){return true;}, //always true; why do we have to pass the bools?
                                  false, VertexType(), false);
            phase_timing[2] += MPI_Wtime()-t1;
            
            //if(fringeCol.getnnz() > 0)fringeCol.DebugPrint();
            //if(fringeRow.getnnz() > 0)fringeRow.DebugPrint();
            
            
            
            //get unmatched row vertices
            t1 = MPI_Wtime();
            umFringeRow  = EWiseApply<VertexType>(fringeRow, mateRow2Col,
                                                  [](VertexType vtx, int64_t mate){return vtx;}, // return unmatched vertices
                                                  [](VertexType vtx, int64_t mate){return mate==-1;}, // select unmatched vertices
                                                  false, VertexType());
            
            phase_timing[3] += MPI_Wtime()-t1;
            t1 = MPI_Wtime();
            // get the unique leaves
            temp = umFringeRow.Compose(Aeff.getncol(),
                                       [](VertexType& vtx, const int64_t & index){return vtx.root;}, // index is the root
                                       [](VertexType& vtx, const int64_t & index){return VertexType(index, vtx.root);}); // value
            phase_timing[4] += MPI_Wtime()-t1;
            
            //set leaf pointer
            t1 = MPI_Wtime();
            leaves.EWiseApply(temp,
                              [](int64_t dval, VertexType svtx, bool a, bool b){return svtx.parent;}, // return parent of the sparse vertex
                              [](int64_t dval, VertexType svtx, bool a, bool b){return dval==-1;}, //if no aug path is already found
                              false, VertexType(), false);
            phase_timing[5] += MPI_Wtime()-t1;
            
            
            //fringeRow  = EWiseApply<VertexType>(fringeRow, mateRow2Col, select1st<VertexType, int64_t>(), matched_binary<VertexType,int64_t>(), false, VertexType());
            // keep matched vertices
            // TODO: this can be merged into compose function for a complicated function to avoid creating unnecessary intermediate fringeRow
            t1 = MPI_Wtime();
            /*
            fringeRow  = EWiseApply<VertexType>(fringeRow, mateRow2Col,
                                                [](VertexType vtx, int64_t mate){return VertexType(mate, vtx.root);}, // return matched vertices with mate as parent
                                                [](VertexType vtx, int64_t mate){return mate!=-1;}, // select matched vertices
                                                false, VertexType());
             */
            fringeRow.SelectApply(mateRow2Col, [](int64_t mate){return mate!=-1;},
                                 [](VertexType vtx, int64_t mate){return VertexType(mate, vtx.root);});
             
            //fringeRow.Select(mateRow2Col, [](int64_t mate){return mate!=-1;});
            phase_timing[6] += MPI_Wtime()-t1;
            //if(fringeRow.getnnz() > 0)fringeRow.DebugPrint();
            //fringeCol = fringeRow.Compose(Aeff.getncol(), binopInd<int64_t>(), binopVal<int64_t>());
            t1 = MPI_Wtime();
            fringeCol = fringeRow.Compose(Aeff.getncol(),
                                          [](VertexType& vtx, const int64_t & index){return vtx.parent;}, // index is the
                                          [](VertexType& vtx, const int64_t & index){return VertexType(vtx.parent, vtx.root);});
            phase_timing[7] += MPI_Wtime()-t1;
            // TODO:do something for prunning
            
            
        }
        time_search = MPI_Wtime() - time_search;
        phase_timing[8] += time_search;
        //leaves.DebugPrint();
        
        // create a sparse vector of leaves
        FullyDistSpVec<int64_t, int64_t> col(leaves, [](int64_t leaf){return leaf!=-1;});
        FullyDistSpVec<int64_t, int64_t> row(A.getcommgrid(), A.getnrow());
        FullyDistSpVec<int64_t, int64_t> nextcol(A.getcommgrid(), A.getncol());
        
        int64_t numMatchedCol = col.getnnz();
        if (col.getnnz()== 0) matched = false;
        
        // Augment
        time_augment = MPI_Wtime();
        while(col.getnnz()!=0)
        {
            t1 = MPI_Wtime();
            row = col.Invert(A.getnrow());
            phase_timing[9] += MPI_Wtime()-t1; 
            // Set parent pointer
            // TODO: Write a general purpose FullyDistSpVec::Set based on a FullyDistVec
            t1 = MPI_Wtime();
            /*
            row = EWiseApply<int64_t>(row, parentsRow,
                                      [](int64_t root, int64_t parent){return parent;},
                                      [](int64_t root, int64_t parent){return true;}, // must have a root
                                      false, (int64_t) -1);
            */
            row.SelectApply(parentsRow, [](int64_t parent){return true;},
                            [](int64_t root, int64_t parent){return parent;}); // this is a Set operation
            
            phase_timing[10] += MPI_Wtime()-t1;
            //if(row.getnnz()!=0)row.DebugPrint();
            
            t1 = MPI_Wtime();
            col = row.Invert(A.getncol()); // children array
            phase_timing[11] += MPI_Wtime()-t1;
            
            t1 = MPI_Wtime();
            nextcol = EWiseApply<int64_t>(col, mateCol2Row,
                                          [](int64_t child, int64_t mate){return mate;},
                                          [](int64_t child, int64_t mate){return mate!=-1;}, // mate==-1 when we have reached to the root
                                          false, (int64_t) -1);
            phase_timing[12] += MPI_Wtime()-t1;
            //col.DebugPrint();
            t1 = MPI_Wtime();
            mateRow2Col.Set(row);
            mateCol2Row.Set(col);
            phase_timing[13] += MPI_Wtime()-t1;
            col = nextcol;
            
        }
        
        //mateRow2Col.DebugPrint();
        //mateCol2Row.DebugPrint();
        
        time_augment = MPI_Wtime() - time_augment;
        phase_timing[14] += time_augment;
        time_phase = MPI_Wtime() - time_phase;
        phase_timing[15] += time_phase;
        timing.push_back(phase_timing);
        
        ostringstream tinfo;
        tinfo << "Phase: " << phase << " Unmatched Columns: " << numUnmatchedCol << " Matched: " << numMatchedCol << " Time: "<< time_phase << "\n";
        SpParHelper::Print(tinfo.str());

        
    }
    
    
    
    
    
    
   
    
    
    //isMaximalmatching(A, mateRow2Col, mateCol2Row, unmatchedRow, unmatchedCol);
    //isMatching(mateCol2Row, mateRow2Col); //todo there is a better way to check this
    
    
    // print statistics
    if(myrank == 0)
    {
        cout << endl;
        cout << "========================================================================================================\n";
        cout << "                         BFS Search                                        Augment    \n";
        cout << "============================================================ =================================== =======\n";
        cout  << "Phase  SpMV EWvis EWSetP EWUmR CmUqL EWSetL EWMR CmMC  BFS   Inv1   EW1  Inv2  EW2   SetM  Aug   Total \n";
        cout << "========================================================================================================\n";
        
        vector<double> totalTimes(timing[0].size(),0);
        int nphases = timing.size();
        for(int i=0; i<timing.size(); i++)
        {
            //printf(" %3d   ", i+1);
            for(int j=0; j<timing[i].size(); j++)
            {
                totalTimes[j] += timing[i][j];
                //timing[i][j] /= timing[i].back();
                //printf("%.2lf  ", timing[i][j]);
            }
            
            //printf("\n");
        }
        
        double combTime = totalTimes.back();
        printf(" %3d   ", nphases);
        for(int j=0; j<totalTimes.size()-1; j++)
        {
            printf("%.2lf  ", totalTimes[j]);
        }
        printf("%.2lf\n", combTime);
        
        //cout << "=================== total timing ===========================\n";
        //for(int i=0; i<totalTimes.size(); i++)
        //    cout<<totalTimes[i] << " ";
        //cout << endl;
    }
    
    
}





/*
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






