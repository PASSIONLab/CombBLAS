#include "CombBLAS/CombBLAS.h"
#include <mpi.h>
#include <sys/time.h> 
#include <iostream>
#include <functional>
#include <algorithm>
#include <vector>
#include <string>
#include <sstream>


using namespace std;
using namespace combblas;

bool prune, mvInvertMate, randMM, moreSplit;
int init;
bool randMaximal;
bool fewexp;


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
    
    // careful: secondMaxProfit should be -\max wij by default
    // how can we pass that info to static VertexType id() so that SpMV can use it?
    VertexType(int64_t oi=-1, double p = 0, double smp=-9999999)
    {
        objID = oi;
        price = p;
        secondMaxProfit = smp;
    };
    
     
    friend bool operator<(const VertexType & vtx1, const VertexType & vtx2 ){return vtx1.price < vtx2.price;};
    friend ostream& operator<<(ostream& os, const VertexType & vertex ){os << "(" << vertex.objID << ", " << vertex.price << ", " << vertex.secondMaxProfit << ")"; return os;};
    
    // variables for the object
    int64_t objID; // ultimately it will be the object with the max profit for a bidder
    double price;  //ultimately it will be the max profit for a bidder
    double secondMaxProfit; //it will be the 2nd max profit for a bidder
    
    
};


// return the maximum and second maximum profits
struct max2 : public std::binary_function<VertexType, VertexType, VertexType>
{
    const VertexType operator()(const VertexType& x, const VertexType& y) const
    {
        VertexType ret;
        if(x.price > y.price)
        {
            ret = x;
            if(x.secondMaxProfit < y.price) ret.secondMaxProfit = y.price;
        }
        else
        {
            ret = y;
            if(y.secondMaxProfit < x.price) ret.secondMaxProfit = x.price;
        }
        
        return ret;
    }
};

        

struct SubMaxSR
{
    static VertexType id(){ return VertexType(); };
    static bool returnedSAID() { return false; }
    static MPI_Op mpi_op() { return MPIOp<max2, VertexType>::op(); };
    

    
    // addition compute the maximum and second maximum profit
    static VertexType add(const VertexType & arg1, const VertexType & arg2)
    {
        return max2()(arg1, arg2);
    }
    
    // multiplication computes profit of an object to a bidder
    // profit_j = cij - price_j
    static VertexType multiply(const double & arg1, const VertexType & arg2)
    {
        // if the matrix is boolean, arg1=1
        return VertexType(arg2.objID, arg1 - arg2.price, arg2.secondMaxProfit);
    }
    
    static void axpy(const double a, const VertexType & x, VertexType & y)
    {
        y = add(y, multiply(a, x));
    }
};



typedef SpParMat < int64_t, bool, SpDCCols<int64_t,bool> > PSpMat_Bool;
typedef SpParMat < int64_t, bool, SpDCCols<int32_t,bool> > PSpMat_s32p64;
typedef SpParMat < int64_t, int64_t, SpDCCols<int64_t,int64_t> > PSpMat_Int64;
typedef SpParMat < int64_t, float, SpDCCols<int64_t,float> > PSpMat_float;
void maximumMatching(PSpMat_s32p64 & Aeff, FullyDistVec<int64_t, int64_t>& mateRow2Col,
                     FullyDistVec<int64_t, int64_t>& mateCol2Row);
void auction(PSpMat_s32p64 & A, FullyDistVec<int64_t, int64_t>& mateRow2Col,
                     FullyDistVec<int64_t, int64_t>& mateCol2Row);
template <class IT, class NT>
bool isMaximalmatching(PSpMat_Int64 & A, FullyDistVec<IT,NT> & mateRow2Col, FullyDistVec<IT,NT> & mateCol2Row,
                       FullyDistSpVec<int64_t, int64_t> unmatchedRow, FullyDistSpVec<int64_t, int64_t> unmatchedCol);




/*
 Remove isolated vertices and purmute
 */
void removeIsolated(PSpMat_Bool & A)
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





int main(int argc, char* argv[])
{
	
    // ------------ initialize MPI ---------------
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
    if (provided < MPI_THREAD_SERIALIZED)
    {
        printf("ERROR: The MPI library does not have MPI_THREAD_SERIALIZED support\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int nprocs, myrank;
	MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    if(argc < 3)
    {
        //ShowUsage();
        MPI_Finalize();
        return -1;
    }

    
    // ------------ Process input arguments and build matrix ---------------
	{
        
        PSpMat_Bool * ABool;
        PSpMat_s32p64 ALocalT;
        ostringstream tinfo;
        double t01, t02;
        if(string(argv[1]) == string("input")) // input option
        {
            ABool = new PSpMat_Bool();
            
            string filename(argv[2]);
            tinfo.str("");
            tinfo << "**** Reading input matrix: " << filename << " ******* " << endl;
            SpParHelper::Print(tinfo.str());
            t01 = MPI_Wtime();
            ABool->ParallelReadMM(filename, false, maximum<bool>());
            t02 = MPI_Wtime();
            ABool->PrintInfo();
            tinfo.str("");
            tinfo << "Reader took " << t02-t01 << " seconds" << endl;
            SpParHelper::Print(tinfo.str());
            //GetOptions(argv+3, argc-3);

        }
        else if(argc < 4)
        {
            //ShowUsage();
            MPI_Finalize();
            return -1;
        }
        else
        {
            unsigned scale = (unsigned) atoi(argv[2]);
            unsigned EDGEFACTOR = (unsigned) atoi(argv[3]);
            double initiator[4];
            if(string(argv[1]) == string("er"))
            {
                initiator[0] = .25;
                initiator[1] = .25;
                initiator[2] = .25;
                initiator[3] = .25;
                cout << "ER ******** \n";
            }
            else if(string(argv[1]) == string("g500"))
            {
                initiator[0] = .57;
                initiator[1] = .19;
                initiator[2] = .19;
                initiator[3] = .05;
                 cout << "g500 ******** \n";
            }
            else if(string(argv[1]) == string("ssca"))
            {
                initiator[0] = .6;
                initiator[1] = .4/3;
                initiator[2] = .4/3;
                initiator[3] = .4/3;
                 cout << "ER ******** \n";
            }
            else
            {
                if(myrank == 0)
                    printf("The input type - %s - is not recognized.\n", argv[2]);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            
            SpParHelper::Print("Generating input matrix....\n");
            t01 = MPI_Wtime();
            DistEdgeList<int64_t> * DEL = new DistEdgeList<int64_t>();
            DEL->GenGraph500Data(initiator, scale, EDGEFACTOR, true, true);
            ABool = new PSpMat_Bool(*DEL, false);
            delete DEL;
            t02 = MPI_Wtime();
            ABool->PrintInfo();
            tinfo.str("");
            tinfo << "Generator took " << t02-t01 << " seconds" << endl;
            SpParHelper::Print(tinfo.str());
            
            Symmetricize(*ABool);
            //removeIsolated(*ABool);
            SpParHelper::Print("Generated matrix symmetricized....\n");
            ABool->PrintInfo();
            
            //GetOptions(argv+4, argc-4);

        }

  
        // randomly permute for load balance
        SpParHelper::Print("Performing random permutation of matrix.\n");
        FullyDistVec<int64_t, int64_t> prow(ABool->getcommgrid());
        FullyDistVec<int64_t, int64_t> pcol(ABool->getcommgrid());
        prow.iota(ABool->getnrow(), 0);
        pcol.iota(ABool->getncol(), 0);
        prow.RandPerm();
        pcol.RandPerm();
        (*ABool)(prow, pcol, true);
        SpParHelper::Print("Performed random permutation of matrix.\n");

     
        PSpMat_s32p64 A = *ABool;
        

        FullyDistVec<int64_t, int64_t> mateRow2Col ( A.getcommgrid(), A.getnrow(), (int64_t) -1);
        FullyDistVec<int64_t, int64_t> mateCol2Row ( A.getcommgrid(), A.getncol(), (int64_t) -1);
    
        auction(A, mateRow2Col, mateCol2Row);
        
        
       
	}
	MPI_Finalize();
	return 0;
}







void auction(PSpMat_s32p64 & A, FullyDistVec<int64_t, int64_t>& mateRow2Col,
             FullyDistVec<int64_t, int64_t>& mateCol2Row)
{
    
    int nprocs, myrank;
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    
    int64_t nrow = A.getnrow();
    int64_t ncol = A.getncol();
    FullyDistSpVec<int64_t, VertexType> fringeRow(A.getcommgrid(), nrow);
    FullyDistSpVec<int64_t, int64_t> umFringeRow(A.getcommgrid(), nrow);
    FullyDistVec<int64_t, int64_t> leaves ( A.getcommgrid(), ncol, (int64_t) -1);
    
    vector<vector<double> > timing;
    vector<int> layers;
    vector<int64_t> phaseMatched;
    double t1, time_search, time_augment, time_phase;
    
    bool matched = true;
    int phase = 0;
    int totalLayer = 0;
    int64_t numUnmatchedCol;
    
    

    FullyDistVec<int64_t, VertexType> objects ( A.getcommgrid(), ncol, VertexType());
    // set index to value 
    objects.ApplyInd([](VertexType vtx, int64_t idx){return VertexType(idx, vtx.price, vtx.secondMaxProfit);});
    FullyDistVec<int64_t, VertexType> bidders ( A.getcommgrid(), nrow, VertexType());
    
    // future: I need an SPMV that could return a dense vector of type other than the input
    bidders = SpMV<SubMaxSR>(A, objects);

    
    // Remove bidders without any profitable objects
    FullyDistSpVec<int64_t, VertexType> activeBidders ( bidders, [](VertexType bidder){return bidder.price>0;});
    
    // remove matched bidders
    // we have done unncessary computation for already matched bidders
    // option1: bottom up style
    // option2: masked SpMV

    activeBidders = EWiseApply<VertexType>(activeBidders, mateRow2Col,
                                            [](VertexType vtx, int64_t mate){return vtx;},
                                            [](VertexType vtx, int64_t mate){return mate==-1;},
                                            false, VertexType());

   
    // compute bid
    activeBidders.Apply([](VertexType bidder){return VertexType(bidder.objID, bidder.price - bidder.secondMaxProfit);}); // I don't care secondMaxProfit anymore
    
    
    // place bid
    // objects need to select the best bidder
    activeBidders.DebugPrint();
    FullyDistSpVec<int64_t, VertexType> bidObject =
                            activeBidders.Invert(ncol,
                            [](VertexType bidder, int64_t idx){return bidder.objID;},
                            [](VertexType bidder, int64_t idx){return VertexType(idx, bidder.price);},
                            [](VertexType bid1, VertexType bid2){return bid1.price>bid2.price? bid1: bid2;}); // I don't care secondMaxProfit anymore
    
    bidObject.DebugPrint();
    
    // just creating a simplified object with the highest bidder
    // mateCol2Row is used just as a mask
    FullyDistSpVec<int64_t, int64_t> successfullBids = EWiseApply<int64_t>(bidObject, mateCol2Row,
                                      [](VertexType vtx, int64_t mate){return vtx.objID;},
                                      [](VertexType vtx, int64_t mate){return true;},
                                      false, VertexType());
    
    
    //mateCol2Row.DebugPrint();
    // bidders previously matched to current successfull bids will become unmatched
    FullyDistSpVec<int64_t, int64_t> revokedBids = EWiseApply<int64_t>(successfullBids, mateCol2Row,
                                                                            [](int64_t newbidder, int64_t mate){return mate;},
                                                                            [](int64_t newbidder, int64_t mate){return mate!=-1;},
                                                                            false, (int64_t)-1);
    
    
    mateCol2Row.Set(successfullBids);
    
    //cout << " djkfhksjdfh \n";
    //successfullBids.DebugPrint();
    //revokedBids.DebugPrint();
    
    // previously unmatched bidders that will be matched
    FullyDistSpVec<int64_t, int64_t> successfullBidders = successfullBids.Invert(nrow);
   
    // previously matched bidders that will be unmatched
    FullyDistSpVec<int64_t, int64_t> revokedBidders = revokedBids.Invert(nrow);
    // they are mutually exclusive
    successfullBidders.DebugPrint();
    revokedBidders.DebugPrint();
    
    mateRow2Col.Set(successfullBidders);
    revokedBidders.Apply([](int64_t prevmate){return (int64_t)-1;});
    
    //mateRow2Col.Set(revokedBidders);
    
   
   
    
    
    //objects.DebugPrint();
    
}

