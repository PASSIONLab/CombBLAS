#include <mpi.h>
#include <sys/time.h>
#include <iostream>
#include <functional>
#include <algorithm>
#include <vector>
#include <sstream>
#include <cstdlib>
#include "CombBLAS/CombBLAS.h"
#include "CombBLAS/ParFriends.h"
using namespace std;
using namespace combblas;

#define EPS 0.0001

typedef struct
{
    //Input/Output file
    string ifilename;
    string dirname; // Patch to dump per iteration matrix into the directory
    bool isInputMM;
    int base; // only useful for matrix market files
    
    string ofilename;
    
    //Preprocessing
    int randpermute;
    bool remove_isolated;

    //inflation
    double inflation;
    
    //pruning
    double prunelimit;
    int64_t select;
    int64_t recover_num;
    double recover_pct;
    int kselectVersion; // 0: adapt based on k, 1: kselect1, 2: kselect2
    bool preprune;
    
    //HipMCL optimization
    int phases;
    int perProcessMem;
    bool isDoublePrecision; // true: double, false: float
    bool is64bInt; // true: int64_t for local indexing, false: int32_t (for local indexing)
    int layers; // Number of layers to use in communication avoiding SpGEMM. 
    int compute;
    
    //debugging
    bool show;
    
    // Options related to incremental setting
    int maxIter; // -ve value specifies number of iterations necessary to get intended summary
    int summaryIter; // If set, MCL state of that particular iteration is saved as summary
    double summaryThresholdNNZ; // If summaryIter is not set then summary is saved when nnz of MCL state drops below this percent of nnz in the beginning MCL state
    bool normalizedAssign;

    
}HipMCLParam;

void InitParam(HipMCLParam & param)
{
    //Input/Output file
    param.ifilename = "";
    param.dirname = "";
    param.isInputMM = false;
    param.ofilename = "mclinc";
    param.base = 1;
    
    //Preprocessing
    // mcl removes isolated vertices by default,
    // we don't do this because it will create different ordering of vertices!
    param.remove_isolated = false;
    param.randpermute = 0;
    
    //inflation
    param.inflation = 2.0;
    
    //pruning
    param.prunelimit = 1.0/10000.0;
    param.select = 1100;
    param.recover_num = 1400;
    param.recover_pct = .9; // we allow both 90 or .9 as input. Internally, we keep it 0.9
    param.kselectVersion = 1;
    param.preprune = false;
    
    //HipMCL optimization
    param.layers = 1;
    param.compute = 1; // 1 means hash-based computation, 2 means heap-based computation
    param.phases = 1;
    param.perProcessMem = 0;
    param.isDoublePrecision = true;
    param.is64bInt = true;
    
    //debugging
    param.show = true;

    //Options related to incremental setting
    param.maxIter = 1000000; // Arbitrary large number means no limit on number of iterations 
    param.summaryIter = 0;
    param.summaryThresholdNNZ = 0.7;
    param.normalizedAssign = true;
}



// base: base of items
// clusters are always numbered 0-based
template <typename IT, typename NT, typename DER>
FullyDistVec<IT, IT> Interpret(SpParMat<IT,NT,DER> & A)
{
    IT nCC;
    // A is a directed graph
    // symmetricize A
    
    SpParMat<IT,NT,DER> AT = A;
    AT.Transpose();
    A += AT;
    SpParHelper::Print("Finding connected components....\n");
    
    FullyDistVec<IT, IT> cclabels = CC(A, nCC);
    return cclabels;
}


template <typename IT, typename NT, typename DER>
void MakeColStochastic(SpParMat<IT,NT,DER> & A)
{
    FullyDistVec<IT, NT> colsums = A.Reduce(Column, plus<NT>(), 0.0);
    colsums.Apply(safemultinv<NT>());
    A.DimApply(Column, colsums, multiplies<NT>());    // scale each "Column" with the given vector
}

template <typename IT, typename NT, typename DER>
void MakeColStochastic3D(SpParMat3D<IT,NT,DER> & A3D)
{
    //SpParMat<IT, NT, DER> * ALayer = A3D.GetLayerMat();
    std::shared_ptr< SpParMat<IT, NT, DER> > ALayer = A3D.GetLayerMat();
    FullyDistVec<IT, NT> colsums = ALayer->Reduce(Column, plus<NT>(), 0.0);
    colsums.Apply(safemultinv<NT>());
    ALayer->DimApply(Column, colsums, multiplies<NT>());    // scale each "Column" with the given vector
}

template <typename IT, typename NT, typename DER>
NT Chaos(SpParMat<IT,NT,DER> & A)
{   
    // sums of squares of columns
    FullyDistVec<IT, NT> colssqs = A.Reduce(Column, plus<NT>(), 0.0, bind2nd(exponentiate(), 2));
    // Matrix entries are non-negative, so max() can use zero as identity
    FullyDistVec<IT, NT> colmaxs = A.Reduce(Column, maximum<NT>(), 0.0);
    colmaxs -= colssqs;
    
    // multiplu by number of nonzeros in each column
    FullyDistVec<IT, NT> nnzPerColumn = A.Reduce(Column, plus<NT>(), 0.0, [](NT val){return 1.0;});
    colmaxs.EWiseApply(nnzPerColumn, multiplies<NT>());
    
    return colmaxs.Reduce(maximum<NT>(), 0.0);
}

template <typename IT, typename NT, typename DER>
NT Chaos3D(SpParMat3D<IT,NT,DER> & A3D)
{
    //SpParMat<IT, NT, DER> * ALayer = A3D.GetLayerMat();
    std::shared_ptr< SpParMat<IT, NT, DER> > ALayer = A3D.GetLayerMat();

    // sums of squares of columns
    FullyDistVec<IT, NT> colssqs = ALayer->Reduce(Column, plus<NT>(), 0.0, bind2nd(exponentiate(), 2));
    // Matrix entries are non-negative, so max() can use zero as identity
    FullyDistVec<IT, NT> colmaxs = ALayer->Reduce(Column, maximum<NT>(), 0.0);
    colmaxs -= colssqs;

    // multiply by number of nonzeros in each column
    FullyDistVec<IT, NT> nnzPerColumn = ALayer->Reduce(Column, plus<NT>(), 0.0, [](NT val){return 1.0;});
    colmaxs.EWiseApply(nnzPerColumn, multiplies<NT>());
    
    NT layerChaos = colmaxs.Reduce(maximum<NT>(), 0.0);

    NT totalChaos = 0.0;
    MPI_Allreduce( &layerChaos, &totalChaos, 1, MPIType<NT>(), MPI_MAX, A3D.getcommgrid3D()->GetFiberWorld());
    return totalChaos;
}

template <typename IT, typename NT, typename DER>
void Inflate(SpParMat<IT,NT,DER> & A, double power)
{
    A.Apply(bind2nd(exponentiate(), power));
}

template <typename IT, typename NT, typename DER>
void Inflate3D(SpParMat3D<IT,NT,DER> & A3D, double power)
{
    //SpParMat<IT, NT, DER> * ALayer = A3D.GetLayerMat();
    std::shared_ptr< SpParMat<IT, NT, DER> > ALayer = A3D.GetLayerMat();
    ALayer->Apply(bind2nd(exponentiate(), power));
}

// default adjustloop setting
// 1. Remove loops
// 2. set loops to max of all arc weights
template <typename IT, typename NT, typename DER>
void AdjustLoops(SpParMat<IT,NT,DER> & A)
{

    A.RemoveLoops();
    FullyDistVec<IT, NT> colmaxs = A.Reduce(Column, maximum<NT>(), numeric_limits<NT>::min());
    A.Apply([](NT val){return val==numeric_limits<NT>::min() ? 1.0 : val;}); // for isolated vertices
    A.AddLoops(colmaxs);
    ostringstream outs;
    outs << "Adjusting loops" << endl;
    SpParHelper::Print(outs.str());
}

template <typename IT, typename NT, typename DER>
void RemoveIsolated(SpParMat<IT,NT,DER> & A, HipMCLParam & param)
{
    ostringstream outs;
    FullyDistVec<IT, NT> ColSums = A.Reduce(Column, plus<NT>(), 0.0);
    FullyDistVec<IT, IT> nonisov = ColSums.FindInds(bind2nd(greater<NT>(), 0));
    IT numIsolated = A.getnrow() - nonisov.TotalLength();
    outs << "Number of isolated vertices: " << numIsolated << endl;
    SpParHelper::Print(outs.str());
    
    A(nonisov, nonisov, true);
    SpParHelper::Print("Removed isolated vertices.\n");
    //if(param.show)
    //{
        //A.PrintInfo();
    //}
    
}

//TODO: handle reordered cluster ids
template <typename IT, typename NT, typename DER>
void RandPermute(SpParMat<IT,NT,DER> & A, HipMCLParam & param)
{
    // randomly permute for load balance
    if(A.getnrow() == A.getncol())
    {
        FullyDistVec<IT, IT> p( A.getcommgrid());
        p.iota(A.getnrow(), 0);
        p.RandPerm();
        (A)(p,p,true);// in-place permute to save memory
        SpParHelper::Print("Applied symmetric permutation.\n");
    }
    else
    {
        SpParHelper::Print("Rectangular matrix: Can not apply symmetric permutation.\n");
    }
}

template <typename IT, typename NT, typename DER>
void HipMCL(SpParMat<IT,NT,DER> & A, HipMCLParam & param, FullyDistVec<IT, IT> & clustAsn, SpParMat<IT, NT, DER> & Asummary)
{
    if(param.remove_isolated)
        RemoveIsolated(A, param);
    
    // BecausePermutation would be taken care outside
    //if(param.randpermute)
        //RandPermute(A, param);

    // Adjust self loops
    AdjustLoops(A);
    
    // Make stochastic
    MakeColStochastic(A);
    
    IT nnz = A.getnnz();
    IT nv = A.getnrow();
    IT avgDegree = nnz/nv;
    if(avgDegree > std::max(param.select, param.recover_num))
    {
        SpParHelper::Print("Average degree of the input graph is greater than max{S,R}.\n");
        param.preprune = true;
    }
    if(param.preprune)
    {
        SpParHelper::Print("Applying the prune/select/recovery logic before the first iteration\n\n");
        MCLPruneRecoverySelect(A, (NT)param.prunelimit, (IT)param.select, (IT)param.recover_num, (NT)param.recover_pct, param.kselectVersion);
    }

    if(param.show)
    {
        A.PrintInfo();
    }
    
    NT chaos = Chaos(A);
    double newBalance = A.LoadImbalance();
    stringstream s;
    s << "Iteration# "  << setw(3) << -1 << " : "  << " chaos: " << setprecision(3) << chaos << "  load-balance: "<< newBalance << " Time: " << -1 << endl;
    SpParHelper::Print(s.str());

    int it=1;
    double tInflate = 0;
    double tExpand = 0;
    double tTotal = 0;
    typedef PlusTimesSRing<NT, NT> PTFF;
	SpParMat3D<IT,NT,DER> A3D_cs(param.layers);
	if(param.layers > 1) {
    	SpParMat<IT,NT,DER> A2D_cs = SpParMat<IT, NT, DER>(A);
		A3D_cs = SpParMat3D<IT,NT,DER>(A2D_cs, param.layers, true, false);    // Non-special column split
	}

    bool summarySaved = false;
    bool stopIter = false;
    IT nnzStart = A.getnnz();

    // while there is an epsilon improvement
    while( (chaos > EPS) && (stopIter == false) )
    {
		SpParMat3D<IT,NT,DER> A3D_rs(param.layers);
		if(param.layers > 1) {
			A3D_rs  = SpParMat3D<IT,NT,DER>(A3D_cs, false); // Create new rowsplit copy of matrix from colsplit copy
		}

        double t1 = MPI_Wtime();
		if(param.layers == 1){
			A = MemEfficientSpGEMM<PTFF, NT, DER>(A, A, param.phases, param.prunelimit, (IT)param.select, (IT)param.recover_num, param.recover_pct, param.kselectVersion, param.compute, param.perProcessMem);
		}
		else{
			A3D_cs = MemEfficientSpGEMM3D<PTFF, NT, DER, IT, NT, NT, DER, DER >(
                A3D_cs, A3D_rs, 
                param.phases, 
                param.prunelimit, 
                (IT)param.select, 
                (IT)param.recover_num, 
                param.recover_pct, 
                param.kselectVersion,
                param.compute,
                param.perProcessMem
         	);
		}
        
		if(param.layers == 1){
			MakeColStochastic(A);
		}
		else{
            MakeColStochastic3D(A3D_cs);
		}
        tExpand += (MPI_Wtime() - t1);
        
        if(param.show)
        {
            A.PrintInfo();
        }
        if(param.layers == 1) chaos = Chaos(A);
        else chaos = Chaos3D(A3D_cs);

        //double tInflate1 = MPI_Wtime();
        if (param.layers == 1) Inflate(A, param.inflation);
        else Inflate3D(A3D_cs, param.inflation);

        if(param.layers == 1) MakeColStochastic(A);
        else MakeColStochastic3D(A3D_cs);
        
        newBalance = A.LoadImbalance();
        double t3=MPI_Wtime();
        stringstream s;
        s << "Iteration# "  << setw(3) << it << " : "  << " chaos: " << setprecision(3) << chaos << "  load-balance: "<< newBalance << " Time: " << (t3-t1) << endl;
        SpParHelper::Print(s.str());

        tTotal += (t3-t1);

        // Save Markov state as summary
        if( param.summaryIter > 0 && param.summaryIter == it){
            // If a particular iteration is specified save that point
            Asummary.FreeMemory();
            Asummary = A;
            summarySaved = true; // To prevent overwritting of summary
            if(param.show){
                stringstream s;
                s << "Summary saved" << endl;
                SpParHelper::Print(s.str());
            }
        }
        else if (param.summaryIter == 0 & summarySaved == false ) {
            // If particular iteration is not specified and summary is not saved yet
            if (A.getnnz() < (IT)(param.summaryThresholdNNZ * nnzStart)){
                // Save summary when nnz drops below the threshold
                Asummary.FreeMemory();
                Asummary = A;
                summarySaved = true; // To prevent overwritting of summary
                if(param.show){
                    stringstream s;
                    s << "Summary saved" << endl;
                    SpParHelper::Print(s.str());
                }
            }
        }

        // Manage stopIter flag
        if(param.maxIter >= 0){
            // If maximum number of iteration is reached
            if (it == param.maxIter) stopIter = true;
        }
        else{
            // If no maximum iteration is specified as number of iterations necessary for summary to be saved
            if (summarySaved) stopIter = true;
        }
        it++;
    }
    
#ifdef TIMING
    double tcc1 = MPI_Wtime();
#endif
    
    // bool can not be used because
    // bool does not work in A.AddLoops(1) used in LACC: can not create a fullydist vector with Bool
    // SpParMat<IT,NT,DER> A does not work because int64_t and float promote trait not defined
    // hence, we are forcing this with IT and double
    SpParMat<IT,double, SpDCCols < IT, double >> ADouble(MPI_COMM_WORLD);
    if(param.layers == 1) ADouble = A;
    else ADouble = A3D_cs.Convert2D();
    clustAsn = Interpret(ADouble);

    IT nclusters = clustAsn.Reduce(maximum<IT>(), (IT) 0 ) ;
    stringstream s2;
    s2 << "Number of clusters: " << nclusters << endl;
    s2 << "Total MCL time: " << tTotal << endl;
    //s2 << "=================================================\n" << endl ;
    SpParHelper::Print(s2.str());
}

template <class IT, class NT, class LBL, class DER>
void IncClustV1(SpParMat<IT, NT, DER> &Mpp, SpParMat<IT, NT, DER> &Mpn, SpParMat<IT, NT, DER> &Mnp, SpParMat<IT, NT, DER> &Mnn, FullyDistVec<IT, LBL> &pVtxLbl, FullyDistVec<IT, LBL> &nVtxLbl, 
        FullyDistVec<IT, LBL> &aVtxLbl, FullyDistVec<IT, IT> &clustAsn, SpParMat<IT, NT, DER> &Mstar, HipMCLParam &param){
    int nprocs, myrank, nthreads = 1;
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
#ifdef THREADED
#pragma omp parallel
    {
        nthreads = omp_get_num_threads();
    }
#endif
    
    auto commGrid = pVtxLbl.getcommgrid();

    double t0, t1, t2, t3;
    
    if(myrank == 0) printf("[Start] Preparing incremental\n");
    t0 = MPI_Wtime();

    FullyDistVec<IT, IT> pVtxMap( commGrid );
    pVtxMap.iota(pVtxLbl.TotalLength(), 0); // Intialize with consecutive numbers
    FullyDistVec<IT, IT> nVtxMap( commGrid );
    nVtxMap.iota(nVtxLbl.TotalLength(), pVtxLbl.TotalLength()); // Initialize with consecutive numbers

    // All prev and new vectors are assumed to be of same length as the respective vertex, label and mapping vector
    IT pLocLen = pVtxLbl.LocArrSize();     
    IT nLocLen = nVtxLbl.LocArrSize();
    IT aLocLen = aVtxLbl.LocArrSize();
           
    IT minLocLen = std::min(pLocLen, nLocLen);
    IT maxLocLen = std::max(pLocLen, nLocLen);

    // Boolean flags for each element to keep track of which elements have been swapped between prev and new
    std::vector<bool> pLocFlag(pLocLen, true);
    std::vector<bool> nLocFlag(nLocLen, true);

    // Initialize two uniform random number generators
    // one is a real generator in a range of [0.0-1.0] to do coin toss to decide whether a swap will happen or not
    // another is an integer generator to randomly pick positions to swap,
    std::mt19937 rng;
    rng.seed(myrank);
    std::uniform_real_distribution<float> urdist(0, 1.0);
    std::uniform_int_distribution<IT> uidist(0, 999999999); // TODO: Choose this range adaptively

    // MTH: Enable multi-threading?
    for (IT i = 0; i < minLocLen; i++){ // Run as many attempts as minimum of the candidate array lengths
        if(urdist(rng) < double(maxLocLen)/(maxLocLen + minLocLen)){ // If the picked random real number is less than the ratio of new and previous vertices
            // Randomly select an index from the previous vertex list
            IT idxPrev = uidist(rng) % pLocLen; 

            // If the selected index is already swapped then cyclicly probe the indices after that 
            // until an index is found which has not been swapped
            while(pLocFlag[idxPrev] == false) idxPrev = (idxPrev + 1) % pLocLen;  
            
            // Mark the index as swapped
            pLocFlag[idxPrev] = false;

            // Randomly select an index from the new vertex list
            IT idxNew = uidist(rng) % nLocLen;

            // If the selected index is already swapped then cyclicly probe the indices after that 
            // until an index is found which has not been swapped
            while(nLocFlag[idxNew] == false) idxNew = (idxNew + 1) % nLocLen;

            // Mark the index as swapped
            nLocFlag[idxNew] = false;

            IT pVtxRM = pVtxMap.GetLocalElement(idxPrev);
            IT nVtxRM = nVtxMap.GetLocalElement(idxNew);
            pVtxMap.SetLocalElement(idxPrev, nVtxRM);
            nVtxMap.SetLocalElement(idxNew, pVtxRM);
        }
    }
    
    // Global permutation after local shuffle will result in true shuffle
    pVtxMap.RandPerm(31415929535);
    nVtxMap.RandPerm(31415929535);

    const std::vector<IT> pVtxMapLoc = pVtxMap.GetLocVec();
    const std::vector<LBL> pVtxLblLoc = pVtxLbl.GetLocVec();
    const std::vector<IT> nVtxMapLoc = nVtxMap.GetLocVec();
    const std::vector<LBL> nVtxLblLoc = nVtxLbl.GetLocVec();

    // Must be array of `int` for MPI requirements
    std::vector<int> sendcnt(nprocs, 0);
    std::vector<int> sdispls(nprocs+1);
    std::vector<int> recvcnt(nprocs, 0);
    std::vector<int> rdispls(nprocs+1);

    for (IT i = 0; i < pLocLen; i++){
        IT rLocIdx; // Index of the local array in the receiver side
        int owner = aVtxLbl.Owner(pVtxMapLoc[i], rLocIdx);
        sendcnt[owner] = sendcnt[owner] + 1;
    }
    for (IT i = 0; i < nLocLen; i++){
        IT rLocIdx; // Index of the local array in the receiver side
        int owner = aVtxLbl.Owner(nVtxMapLoc[i], rLocIdx);
        sendcnt[owner] = sendcnt[owner] + 1;
    }

    MPI_Alltoall(sendcnt.data(), 1, MPI_INT, recvcnt.data(), 1, MPI_INT, commGrid->GetWorld());

    sdispls[0] = 0;
    rdispls[0] = 0;
    std::partial_sum(sendcnt.begin(), sendcnt.end(), sdispls.begin()+1);
    std::partial_sum(recvcnt.begin(), recvcnt.end(), rdispls.begin()+1);

    int totsend = sdispls[sdispls.size()-1]; // Can be safely assumed to be int because MPI forces the array to be of int
    int totrecv = rdispls[rdispls.size()-1];

    std::vector< std::tuple<IT, LBL> > sendTuples(totsend);
    std::vector< std::tuple<IT, LBL> > recvTuples(totrecv);
    std::vector<int> sidx(sdispls); // Copy sdispls array to use for preparing sendTuples
                                    
    for (IT i = 0; i < pLocLen; i++){
        IT rLocIdx; // Index of the local array in the receiver side
        int owner = aVtxLbl.Owner(pVtxMapLoc[i], rLocIdx);
        sendTuples[sidx[owner]] = std::make_tuple(rLocIdx, pVtxLblLoc[i]);
        sidx[owner]++;
    }
    for (IT i = 0; i < nLocLen; i++){
        IT rLocIdx; // Index of the local array in the receiver side
        int owner = aVtxLbl.Owner(nVtxMapLoc[i], rLocIdx);
        sendTuples[sidx[owner]] = std::make_tuple(rLocIdx, nVtxLblLoc[i]);
        sidx[owner]++;
    }

    MPI_Datatype MPI_Custom;
    MPI_Type_contiguous(sizeof(std::tuple<IT,LBL>), MPI_CHAR, &MPI_Custom);
    MPI_Type_commit(&MPI_Custom);
    MPI_Alltoallv(sendTuples.data(), sendcnt.data(), sdispls.data(), MPI_Custom, recvTuples.data(), recvcnt.data(), rdispls.data(), MPI_Custom, commGrid->GetWorld());

    for(int i = 0; i < totrecv; i++){
        IT rLocIdx = std::get<0>(recvTuples[i]);
        LBL rLocLbl = std::get<1>(recvTuples[i]);
        aVtxLbl.SetLocalElement(rLocIdx, rLocLbl);
    }

    t1 = MPI_Wtime();
    if(myrank == 0) printf("Time to calculate vertex mapping: %lf\n", t1-t0);

    t0 = MPI_Wtime();
    SpParMat<IT, NT, DER> Mall = SpParMat<IT,NT,DER>(Mpp.getnrow() + Mnn.getnrow(), 
                 Mpp.getncol() + Mnn.getncol(), 
                 FullyDistVec<IT,IT>(commGrid), 
                 FullyDistVec<IT,IT>(commGrid), 
                 FullyDistVec<IT,IT>(commGrid), true); 
    SpParMat<IT, NT, DER> Mall_Mpn = SpParMat<IT,NT,DER>(Mall);
    SpParMat<IT, NT, DER> Mall_Mnp = SpParMat<IT,NT,DER>(Mall);
    SpParMat<IT, NT, DER> Mall_Mnn = SpParMat<IT,NT,DER>(Mall);
    
    // Assign each piece of incremental matrix to empty matrix
    if (param.normalizedAssign){
        MakeColStochastic(Mpp);
        Mpp.Apply(bind1st(multiplies<double>(), Mpp.getnrow()));
        MakeColStochastic(Mpn);
        Mpn.Apply(bind1st(multiplies<double>(), Mpn.getnrow()));
        MakeColStochastic(Mnp);
        Mnp.Apply(bind1st(multiplies<double>(), Mnp.getnrow()));
        MakeColStochastic(Mnn);
        Mnn.Apply(bind1st(multiplies<double>(), Mnn.getnrow()));
    }
    Mall.SpAsgn(pVtxMap, pVtxMap, Mpp);
    Mall_Mpn.SpAsgn(pVtxMap, nVtxMap, Mpn);
    Mall_Mnp.SpAsgn(nVtxMap, pVtxMap, Mnp);
    Mall_Mnn.SpAsgn(nVtxMap, nVtxMap, Mnn);
    // Sum them up
    Mall += Mall_Mpn;
    Mall += Mall_Mnp;
    Mall += Mall_Mnn;
    t1 = MPI_Wtime();
    if(myrank == 0) printf("Time to assign submatrices: %lf\n", t1-t0);
    if(myrank == 0) printf("[End] Preparing incremental\n");
    
    if(myrank == 0) printf("[Start] Clustering incremental\n");
    HipMCL(Mall, param, clustAsn, Mstar);
    if(myrank == 0) printf("[End] Clustering incremental\n");
};

template <class IT, class NT, class LBL, class DER>
void IncClust(SpParMat<IT, NT, DER> &Mpp, SpParMat<IT, NT, DER> &Mpn, SpParMat<IT, NT, DER> &Mnp, SpParMat<IT, NT, DER> &Mnn, FullyDistVec<IT, LBL> &pVtxLbl, FullyDistVec<IT, LBL> &nVtxLbl, 
        FullyDistVec<IT, LBL> &aVtxLbl, FullyDistVec<IT, IT> &clustAsn, SpParMat<IT, NT, DER> &Mstar, int version, HipMCLParam &param){
    IncClustV1(Mpp, Mpn, Mnp, Mnn, pVtxLbl, nVtxLbl, aVtxLbl, clustAsn, Mstar, param);
};

