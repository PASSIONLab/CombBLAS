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
void greedyMatching(PSpMat_Bool & Aeff);
void maximumMatching(PSpMat_Bool & Aeff);
void maximumMatchingSimple(PSpMat_Bool & Aeff);
template <class IT, class NT>
bool isMaximalmatching(PSpMat_Int64 & A, FullyDistVec<IT,NT> & mateRow2Col, FullyDistVec<IT,NT> & mateCol2Row,
                       FullyDistSpVec<int64_t, int64_t> unmatchedRow, FullyDistSpVec<int64_t, int64_t> unmatchedCol);

void removeIsolated(PSpMat_Bool & A)
{
   

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
			cout << "Usage: ./bpmm <Scale>" << endl;
			cout << "Example: mpirun -np 4 ./bpmm 20" << endl;
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
        FullyDistVec<int64_t, int64_t> * ColSums = new FullyDistVec<int64_t, int64_t>(A1.getcommgrid());
        FullyDistVec<int64_t, int64_t> * RowSums = new FullyDistVec<int64_t, int64_t>(A1.getcommgrid());
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
        A1.operator()(nonisoRowV, nonisoColV, true);
        /////
        */
        
        
        //removeIsolated(A1);
        
       
        //A1.Transpose();
        //varify_matching(*ABool);
        maximumMatching(*ABool);
        
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
            fringeRow.Select(parentsRow, [](int64_t parent){return parent==-1;});
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
            umFringeRow = fringeRow.SelectNew(mateRow2Col, [](int64_t mate){return mate==-1;});
            phase_timing[3] += MPI_Wtime()-t1;
            t1 = MPI_Wtime();
            // get the unique leaves
            MPI_Pcontrol(1,"Compose");
            temp = umFringeRow.Compose(Aeff.getncol(),
                                       [](VertexType& vtx, const int64_t & index){return vtx.root;}, // index is the root
                                       [](VertexType& vtx, const int64_t & index){return VertexType(index, vtx.root);}); // value
            MPI_Pcontrol(-1,"Compose");
            phase_timing[4] += MPI_Wtime()-t1;
            
            //set leaf pointer
            t1 = MPI_Wtime();
            leaves.EWiseApply(temp,
                              [](int64_t dval, VertexType svtx, bool a, bool b){return svtx.parent;}, // return parent of the sparse vertex
                              [](int64_t dval, VertexType svtx, bool a, bool b){return dval==-1;}, //if no aug path is already found
                              false, VertexType(), false);
            phase_timing[5] += MPI_Wtime()-t1;
            
            
            t1 = MPI_Wtime();
            fringeRow.SelectApply(mateRow2Col, [](int64_t mate){return mate!=-1;},
                                  [](VertexType vtx, int64_t mate){return VertexType(mate, vtx.root);});
            phase_timing[6] += MPI_Wtime()-t1;
            //if(fringeRow.getnnz() > 0)fringeRow.DebugPrint();

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






