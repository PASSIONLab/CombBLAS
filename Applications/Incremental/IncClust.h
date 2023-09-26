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

double mcl_symbolictime;
double mcl_Abcasttime;
double mcl_Bbcasttime;
double mcl_localspgemmtime;
double mcl_multiwaymergetime;
double mcl_kselecttime;
double mcl_prunecolumntime;

/* Variables specific for timing communication avoiding setting in detail*/
double mcl3d_conversiontime;
double mcl3d_symbolictime;
double mcl3d_Abcasttime;
double mcl3d_Bbcasttime;
double mcl3d_SUMMAtime;
double mcl3d_localspgemmtime;
double mcl3d_SUMMAmergetime;
double mcl3d_reductiontime;
double mcl3d_3dmergetime;
double mcl3d_kselecttime;

// for compilation (TODO: fix this dependency)
double cblas_alltoalltime;
double cblas_allgathertime;
double cblas_localspmvtime;
double cblas_mergeconttime;
double cblas_transvectime;

int64_t mcl_memory;
double tIO;

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
    double selectivePruneThreshold;
    string incMatFname;
    bool shuffleVertexOrder;
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
    param.maxIter = std::numeric_limits<int>::max(); // Arbitrary large number means no limit on number of iterations 
    param.summaryIter = -1; // Not defined by default 
    param.summaryThresholdNNZ = -1.0; // Not defined by default
    param.normalizedAssign = false; // Turn off normalized assign by default
    param.selectivePruneThreshold = -1.0; // Turn off selective prunning by default
    param.shuffleVertexOrder = false; // Turn off vertex shuffle by default
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

// Given an adjacency matrix, and cluster assignment vector, removes inter cluster edges
template <class IT, class NT, class DER>
void RemoveInterClusterEdges(SpParMat<IT, NT, DER>& M, FullyDistVec<IT, IT>& C){
    FullyDistVec<IT, NT> Ctemp(C); // To convert value types from IT to NT. Because DimApply and PruneColumn requires that.
    SpParMat<IT, NT, DER> Mask(M);
    Mask.DimApply(Row, Ctemp, [](NT mv, NT vv){return vv;});
    Mask.PruneColumn(Ctemp, [](NT mv, NT vv){return static_cast<NT>(vv == mv);}, true);

    //Mask.PrintInfo();
    M.SetDifference(Mask);
}

/*
 * Prune nz from A depending on provided mask, vertex flag and provided threshold
 * Dont prune a nz if -
 * either (1) the flags corresponding to row and column of the nz are different
 * or (2) mask has an nz at the same location
 * or (3) nz value is higher than the threshold provided in the parameter
 * */
template <typename IT, typename NT, typename DER, typename FLAGTYPE> 
void SelectivePrune (SpParMat<IT,NT,DER> & A, SpParMat<IT,NT,DER> & Mask, FullyDistVec<IT,FLAGTYPE>& isOld, HipMCLParam & param){
    //SpParHelper::Print("[SelectivePrune]\n");

    // Make a copy. This copy would keep the information which nz to be pruned
    SpParMat<IT, NT, DER> Ap(A);
    //Ap.PrintInfo();
    //Mask.PrintInfo();
    //SpParHelper::Print("===\n");

    // IMPORTANT: Apply criteria(2) at first
    Ap.SetDifference(Mask);
    //Ap.PrintInfo();
    //SpParHelper::Print("===\n");

    // IMPORTANT: Apply criteria(3) next
    Ap.Prune(std::bind2nd(std::greater_equal<NT>(), param.selectivePruneThreshold), true); // Remove nz where value stays above threshold. Those values would never be pruned.
    //Ap.PrintInfo();
    //SpParHelper::Print("===\n");

    // Apply criteria (1)
    FullyDistVec<IT, NT> isOldTemp(isOld); // To convert value types from IT to NT. Because DimApply and PruneColumn requires that.
    Ap.DimApply(Row, isOldTemp, [](NT mv, NT vv){return vv;}); // Store row flag at the nz locations
    Ap.PruneColumn(isOldTemp, [](NT mv, NT vv){return static_cast<NT>(vv != mv);}, true); // Remove nz where row flag and column flag does not match
    //Ap.PrintInfo();
    //SpParHelper::Print("===\n");

    // Prune everything from A where a corresponding entry exists in Ap
    A.SetDifference(Ap);
    //A.PrintInfo();
    //SpParHelper::Print("[SelectivePrune]\n");

    return;
}

template <typename IT, typename NT, typename DER>
void HipMCL(SpParMat<IT,NT,DER> & A, HipMCLParam & param, FullyDistVec<IT, IT> & clustAsn, SpParMat<IT, NT, DER> & Asummary)
{
    int nprocs, myrank, nthreads = 1;
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);


#ifdef TIMING
    mcl_Abcasttime = 0;
    mcl_Bbcasttime = 0;
    mcl_localspgemmtime = 0;
    mcl_multiwaymergetime = 0;
    mcl_kselecttime = 0;
    mcl_prunecolumntime = 0;
#endif

    double newBalance; 
    //newBalance = A.LoadImbalance();
    //if(myrank == 0) printf("LoadImbalance: %lf\n", newBalance);

    if(param.remove_isolated)
        RemoveIsolated(A, param);

    //newBalance = A.LoadImbalance();
    //if(myrank == 0) printf("LoadImbalance: %lf\n", newBalance);
    
    if(param.randpermute)
        RandPermute(A, param);

    //newBalance = A.LoadImbalance();
    //if(myrank == 0) printf("LoadImbalance: %lf\n", newBalance);

    // Adjust self loops
    AdjustLoops(A);

    //newBalance = A.LoadImbalance();
    //if(myrank == 0) printf("LoadImbalance: %lf\n", newBalance);
    
    // Make stochastic
    MakeColStochastic(A);

    //newBalance = A.LoadImbalance();
    //if(myrank == 0) printf("LoadImbalance: %lf\n", newBalance);
    
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

    //newBalance = A.LoadImbalance();
    //if(myrank == 0) printf("LoadImbalance: %lf\n", newBalance);

    if(param.show)
    {
        A.PrintInfo();
    }
    
    NT chaos = Chaos(A);
    newBalance = A.LoadImbalance();
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

    bool summarySaved = false; // Flag indicating if a summary is saved or not
    bool stopIter = false; // If true MCL iteration will stop
    IT nnzStart = A.getnnz();

    Asummary.FreeMemory();
    Asummary = A; // Keeping the initial MCL state as fallback summary

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

        if(param.summaryThresholdNNZ >= 0.0){
            // If summaryThresholdNNZ is defined (+ve percentage value represents defined)
            if(param.summaryIter >= 0){
                // If summaryIter is defined (+ve represents defined)

                // Every iteration will enter here if above two parameters are defined
                // Need to decide appropriate time to take action
                if(summarySaved == false){
                    // If summary is not yet saved
                    if (it < param.summaryIter){
                        // If summaryIter is defined to be a later state
                        if (A.getnnz() < (IT)(param.summaryThresholdNNZ * nnzStart)){
                            // Save summary when nnz drops below the threshold
                            Asummary.FreeMemory();
                            Asummary = A;
                            summarySaved = true; // To prevent overwritting of summary
                            if(param.show){
                                SpParHelper::Print("Summary saved\n");
                            }
                        }
                        else{
                            // Just keep going
                        }
                    }
                    else{
                        // If MCL state moved past summaryIter state
                        // Save the current state as summary
                        Asummary.FreeMemory();
                        Asummary = A;
                        summarySaved = true;
                        if(param.show){
                            SpParHelper::Print("Summary saved\n");
                        }
                    }
                }
                else{
                    // If summary is already saved keep going, no action is necessary
                }
            }
            else{
                // If summaryIter is not defined (-ve represents not defined)
                // Keep going without doing anything fallback summary(initial MCL state) will be used
            }
        }
        else{
            // If summaryThresholdNNZ is not defined (-ve percentage value represents not defined)
            if(param.summaryIter >= 0){
                // If summaryIter is defined (+ve represents defined)

                // Every iteration will enter here depending on parameter definition
                // Need to decide appropriate time to take action
                if(summarySaved == false){
                    // If summary is not yet saved
                    if(it == param.summaryIter){
                        // If current MCL state should be saved as summary
                        Asummary.FreeMemory();
                        Asummary = A;
                        summarySaved = true;
                        if(param.show){
                            SpParHelper::Print("Summary saved\n");
                        }
                    }
                }
                else{
                    // If summary is already saved keep going, no action is necessary
                }
            }
            else{
                // If summaryIter is not defined (-ve represents not defined)
                // Keep going without doing anything fallback summary(initial MCL state) will be used
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
//#ifdef TIMING
    //s2 << "Abcasttime: " << mcl_Abcasttime << endl;
    //s2 << "Bbcasttime: " << mcl_Bbcasttime << endl;
    //s2 << "localspgemmtime: " << mcl_localspgemmtime << endl;
    //s2 << "multiwaymergetime: " << mcl_multiwaymergetime << endl;
    //s2 << "kselecttime: " << mcl_kselecttime << endl;
    //s2 << "prunecolumntime: " << mcl_prunecolumntime << endl;
//#endif
    //s2 << "=================================================\n" << endl ;
    SpParHelper::Print(s2.str());
}

template <typename IT, typename NT, typename DER, typename FLAGTYPE>
void IncrementalMCL(SpParMat<IT,NT,DER> & A, HipMCLParam & param, FullyDistVec<IT, IT> & clustAsn, SpParMat<IT, NT, DER> & Asummary, FullyDistVec<IT, FLAGTYPE>& isOld, SpParMat<IT,NT,DER> & Mask)
{

#ifdef TIMING
    mcl_Abcasttime = 0;
    mcl_Bbcasttime = 0;
    mcl_localspgemmtime = 0;
    mcl_multiwaymergetime = 0;
    mcl_kselecttime = 0;
    mcl_prunecolumntime = 0;
#endif

    if(param.remove_isolated)
        RemoveIsolated(A, param);
    
    if(param.randpermute)
        RandPermute(A, param);

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

    Asummary.FreeMemory();
    Asummary = A;

    // while there is an epsilon improvement
    while( (chaos > EPS) && (stopIter == false) )
    {
        IT nnzBeforeIter = A.getnnz();

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
        IT nnzAfterIter = A.getnnz();
        
        if(param.show)
        {
            A.PrintInfo();
        }
        
        if( (nnzAfterIter > nnzBeforeIter) && (param.selectivePruneThreshold >= 0.0) ){
            SelectivePrune(A, Mask, isOld, param); //in-place
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

        if(param.summaryThresholdNNZ >= 0.0){
            // If summaryThresholdNNZ is defined (+ve percentage value represents defined)
            if(param.summaryIter >= 0){
                // If summaryIter is defined (+ve represents defined)

                // Every iteration will enter here if above two parameters are defined
                // Need to decide appropriate time to take action
                if(summarySaved == false){
                    // If summary is not yet saved
                    if (it < param.summaryIter){
                        // If summaryIter is defined to be a later state
                        if (A.getnnz() < (IT)(param.summaryThresholdNNZ * nnzStart)){
                            // Save summary when nnz drops below the threshold
                            Asummary.FreeMemory();
                            Asummary = A;
                            summarySaved = true; // To prevent overwritting of summary
                            if(param.show){
                                SpParHelper::Print("Summary saved\n");
                            }
                        }
                        else{
                            // Just keep going
                        }
                    }
                    else{
                        // If MCL state moved past summaryIter state
                        // Save the current state as summary
                        Asummary.FreeMemory();
                        Asummary = A;
                        summarySaved = true;
                        if(param.show){
                            SpParHelper::Print("Summary saved\n");
                        }
                    }
                }
                else{
                    // If summary is already saved keep going, no action is necessary
                }
            }
            else{
                // If summaryIter is not defined (-ve represents not defined)
                // Keep going without doing anything fallback summary(initial MCL state) will be used
            }
        }
        else{
            // If summaryThresholdNNZ is not defined (-ve percentage value represents not defined)
            if(param.summaryIter >= 0){
                // If summaryIter is defined (+ve represents defined)

                // Every iteration will enter here depending on parameter definition
                // Need to decide appropriate time to take action
                if(summarySaved == false){
                    // If summary is not yet saved
                    if(it == param.summaryIter){
                        // If current MCL state should be saved as summary
                        Asummary.FreeMemory();
                        Asummary = A;
                        summarySaved = true;
                        if(param.show){
                            SpParHelper::Print("Summary saved\n");
                        }
                    }
                    else{
                        // Keep going
                    }
                }
                else{
                    // If summary is already saved keep going, no action is necessary
                }
            }
            else{
                // If summaryIter is not defined (-ve represents not defined)
                // Keep going without doing anything fallback summary(initial MCL state) will be used
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
//#ifdef TIMING
    //s2 << "Abcasttime: " << mcl_Abcasttime << endl;
    //s2 << "Bbcasttime: " << mcl_Bbcasttime << endl;
    //s2 << "localspgemmtime: " << mcl_localspgemmtime << endl;
    //s2 << "multiwaymergetime: " << mcl_multiwaymergetime << endl;
    //s2 << "kselecttime: " << mcl_kselecttime << endl;
    //s2 << "prunecolumntime: " << mcl_prunecolumntime << endl;
//#endif
    //s2 << "=================================================\n" << endl ;
    SpParHelper::Print(s2.str());
}

/*
 * Prepares incremental matrix given four pieces of information
 * Inputs:
 *  - Mpp: Matrix denoting edges between vertices involved in previous incremental step
 *  - Mpn: Matrix denoting edges between previous vertices and new vertices
 *  - Mnp: Matrix denoting edges between new vertices and previous vertices
 *  - Mnn: Matrix denoting edges between new vertices
 *  - pLbl: Labels of the previous vertices 
 *  - nLbl: Labels of the new vertices
 *  Outputs:
 *  - Minc: Matrix denoting all edges. Possibly with edges permuted randomly
 *  - incLbl: Labels of Mall. But in permuted order
 *  - isOld: Flags denoting whether a vertex is old or new
 *  - permMap: Vector representing permutation
 * */
template <class IT, class NT,  class DER, class LBL, class FLAGTYPE>
void PrepIncMat(SpParMat<IT, NT, DER> &Mpp, SpParMat<IT, NT, DER> &Mpn, SpParMat<IT, NT, DER> &Mnp, SpParMat<IT, NT, DER> &Mnn, 
        FullyDistVec<IT, LBL> &pLbl, FullyDistVec<IT, LBL> &nLbl, 
        SpParMat<IT, NT, DER> &Minc, FullyDistVec<IT, LBL> &incLbl, FullyDistVec<IT, FLAGTYPE> &isOld, FullyDistVec<IT, IT> &permMap, HipMCLParam & param){
    int nprocs, myrank, nthreads = 1;
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
#ifdef THREADED
#pragma omp parallel
    {
        nthreads = omp_get_num_threads();
    }
#endif
    auto commGrid = pLbl.getcommgrid();

    double t0, t1, t2, t3;
    
    //t0 = MPI_Wtime();
    
    /*
     *  Prepare a position mapping of where each old and new vertex will go in the incremental matrix
     *  Shuffle them in such a way so that if the rows/columns of the pieces of the incremental matrix is positioned 
     *  in this way, the incremental matrix would effectively be randomly permuted (for better load balance)
     * */
    // Prepare positions as if position of new rows comes right after all the old nodes
    FullyDistVec<IT, IT> pMap( commGrid );
    pMap.iota(pLbl.TotalLength(), 0); // Intialize with consecutive numbers
    FullyDistVec<IT, FLAGTYPE> pFlags(commGrid, pLbl.TotalLength(), 1); // True is to indicate previous vertices
    FullyDistVec<IT, IT> nMap( commGrid );
    nMap.iota(nLbl.TotalLength(), pLbl.TotalLength()); // Initialize with consecutive numbers
    FullyDistVec<IT, FLAGTYPE> nFlags(commGrid, nLbl.TotalLength(), 0); // False is to indicate new vertices
    
    // All prev and new vectors are assumed to be of same length as the respective vertex, label and mapping vector
    IT pLocLen = pLbl.LocArrSize();     
    IT nLocLen = nLbl.LocArrSize();
    IT incLocLen = incLbl.LocArrSize();

    // If the parameter is set to shuffle this mapping
    if(param.shuffleVertexOrder){
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

                IT pPos = pMap.GetLocalElement(idxPrev);
                IT nPos = nMap.GetLocalElement(idxNew);
                FLAGTYPE pFlag = pFlags.GetLocalElement(idxPrev);
                FLAGTYPE nFlag = nFlags.GetLocalElement(idxNew);
                pMap.SetLocalElement(idxPrev, nPos);
                nMap.SetLocalElement(idxNew, pPos);
                pFlags.SetLocalElement(idxPrev, nFlag);
                nFlags.SetLocalElement(idxNew, pFlag);
            }
        }
        
        // Global permutation after local shuffle will result in true shuffle
        // MTH: Can be done in a single all-to-all?
        pMap.RandPerm(31415929535);
        nMap.RandPerm(31415929535);
        pFlags.RandPerm(31415929535);
        nFlags.RandPerm(31415929535);
    }
    /*
     * Preparation of position mapping done
     * */
    
    //pMap.DebugPrint();

    /*
     * Combine the two pieces(old, new) of each vector(label, mapping, old-new flag) to get one piece of each
     * Combining two mapping vectors would be simple concatenation - this vector, called permutation vector, would be used to create permutation matrix
     * Combining label and old-new flag vector would require customization following the permutation vector - 
     *      - the flag vector would be used in selective prunning during markov expansion if necessary
     *      - the label vector would give the label of the matrix rows
     * */
    // Combining label and old-new flag vector using custom logic
    const std::vector<IT> pMapLoc = pMap.GetLocVec();
    const std::vector<LBL> pLblLoc = pLbl.GetLocVec();
    const std::vector<IT> nMapLoc = nMap.GetLocVec();
    const std::vector<LBL> nLblLoc = nLbl.GetLocVec();
    const std::vector<FLAGTYPE> pFlagsLoc = pFlags.GetLocVec();
    const std::vector<FLAGTYPE> nFlagsLoc = nFlags.GetLocVec();
    
    std::vector<int> sendcnt(nprocs, 0); // Must be array of `int` for MPI requirements
    std::vector<int> sdispls(nprocs+1);
    std::vector<int> recvcnt(nprocs, 0);
    std::vector<int> rdispls(nprocs+1);

    for (IT i = 0; i < pLocLen; i++){
        IT rLocIdx; // Index of the local array in the receiver side
        int owner = incLbl.Owner(pMapLoc[i], rLocIdx);
        sendcnt[owner] = sendcnt[owner] + 1;
    }
    for (IT i = 0; i < nLocLen; i++){
        IT rLocIdx; // Index of the local array in the receiver side
        int owner = incLbl.Owner(nMapLoc[i], rLocIdx);
        sendcnt[owner] = sendcnt[owner] + 1;
    }

    MPI_Alltoall(sendcnt.data(), 1, MPI_INT, recvcnt.data(), 1, MPI_INT, commGrid->GetWorld());

    sdispls[0] = 0;
    rdispls[0] = 0;
    std::partial_sum(sendcnt.begin(), sendcnt.end(), sdispls.begin()+1);
    std::partial_sum(recvcnt.begin(), recvcnt.end(), rdispls.begin()+1);

    int totsend = sdispls[sdispls.size()-1]; // Can be safely assumed to be int because MPI forces the array to be of int
    int totrecv = rdispls[rdispls.size()-1];

    std::vector< std::tuple<IT, LBL, FLAGTYPE, IT> > sendTuples(totsend);
    std::vector< std::tuple<IT, LBL, FLAGTYPE, IT> > recvTuples(totrecv);
    std::vector<int> sidx(sdispls); // Copy sdispls array to use for preparing sendTuples

    for (IT i = 0; i < pLocLen; i++){
        IT rLocIdx; // Index of the local array in the receiver side
        int owner = incLbl.Owner(pMapLoc[i], rLocIdx);
        sendTuples[sidx[owner]] = std::make_tuple(rLocIdx, pLblLoc[i], pFlagsLoc[i], pMap.LengthUntil()+i);
        sidx[owner]++;
    }
    for (IT i = 0; i < nLocLen; i++){
        IT rLocIdx; // Index of the local array in the receiver side
        int owner = incLbl.Owner(nMapLoc[i], rLocIdx);
        sendTuples[sidx[owner]] = std::make_tuple(rLocIdx, nLblLoc[i], nFlagsLoc[i], pMap.TotalLength() + nMap.LengthUntil()+i);
        sidx[owner]++;
    }

    MPI_Datatype MPI_Custom;
    MPI_Type_contiguous(sizeof(std::tuple<IT,LBL,FLAGTYPE,IT>), MPI_CHAR, &MPI_Custom);
    MPI_Type_commit(&MPI_Custom);
    MPI_Alltoallv(sendTuples.data(), sendcnt.data(), sdispls.data(), MPI_Custom, recvTuples.data(), recvcnt.data(), rdispls.data(), MPI_Custom, commGrid->GetWorld());

    //FullyDistVec<IT, FLAGTYPE> isOld( commGrid, incLbl.TotalLength(), 1); // No significance of the initval

    for(int i = 0; i < totrecv; i++){
        IT rLocIdx = std::get<0>(recvTuples[i]);
        LBL rLocLbl = std::get<1>(recvTuples[i]);
        FLAGTYPE rLocFlag = std::get<2>(recvTuples[i]);
        IT rLocPermMap = std::get<3>(recvTuples[i]);
        incLbl.SetLocalElement(rLocIdx, rLocLbl);
        isOld.SetLocalElement(rLocIdx, rLocFlag);
        permMap.SetLocalElement(rLocIdx, rLocPermMap);
    }
    // Combining label and old-new flag vector done
    
    //// Combining label vectors by conctenate
    //std::vector<FullyDistVec<IT, LBL>> toConcatenateLbl(2); 
    //toConcatenateLbl[0] = pVtxlbl;
    //toConcatenateLbl[1] = nVtxlbl;
    //aVtxLbl = Concatenate(toConcatenateLbl);
    //// Combining label vectors done
    
    //// Combining position mapping vectors by conctenate
    //std::vector<FullyDistVec<IT, IT>> toConcatenateMap(2); 
    //toConcatenateMap[0] = pMap;
    //toConcatenateMap[1] = nMap;
    //permMap = Concatenate(toConcatenateMap);
    // Combining position mapping vectors done
    /*
     * Combining the two pieces(old, new) of each vector(label, mapping, old-new flag) done
     * */

    //t1 = MPI_Wtime();
    //if(myrank == 0) printf("Time to calculate vertex mapping: %lf\n", t1-t0);

    /*
     * Combine four pieces of matrix to prepare final incremental matrix
     * Possible to do it by doing one custom alltoall
     * Here done by taking four big empty matrices and assigning four pieces on each of them followed by sparse matrix addition
     * */
    {
        t0 = MPI_Wtime();
        Minc = SpParMat<IT,NT,DER>(Mpp.getnrow() + Mnn.getnrow(), 
                     Mpp.getncol() + Mnn.getncol(), 
                     FullyDistVec<IT,IT>(commGrid), 
                     FullyDistVec<IT,IT>(commGrid), 
                     FullyDistVec<IT,IT>(commGrid), true); 
        SpParMat<IT, NT, DER> Minc_Mpn = SpParMat<IT,NT,DER>(Minc);
        SpParMat<IT, NT, DER> Minc_Mnp = SpParMat<IT,NT,DER>(Minc);
        SpParMat<IT, NT, DER> Minc_Mnn = SpParMat<IT,NT,DER>(Minc);
        //SpParMat<IT, NT, DER> Mask = SpParMat<IT,NT,DER>(Mall); // Mask will contain information to reduce recomputation of markov flows
        
        // Assign each piece of incremental matrix to empty matrix
        if (param.normalizedAssign){
            MakeColStochastic(Mpp);
            Mpp.Apply(bind1st(multiplies<NT>(), Mpp.getnrow()));
            //MakeColStochastic(Mpp);

            MakeColStochastic(Mpn);
            Mpn.Apply(bind1st(multiplies<NT>(), Mpn.getnrow()));
            //MakeColStochastic(Mpn);

            MakeColStochastic(Mnp);
            Mnp.Apply(bind1st(multiplies<NT>(), Mnp.getnrow()));
            //MakeColStochastic(Mnp);

            MakeColStochastic(Mnn);
            Mnn.Apply(bind1st(multiplies<NT>(), Mnn.getnrow()));
            //MakeColStochastic(Mnn);
        }
        Minc.SpAsgn(pMap, pMap, Mpp);
        Minc_Mpn.SpAsgn(pMap, nMap, Mpn);
        Minc_Mnp.SpAsgn(nMap, pMap, Mnp);
        Minc_Mnn.SpAsgn(nMap, nMap, Mnn);

        // Sum them up
        //Mask += Mall; Mask += Mall_Mnn;
        Minc += Minc_Mpn; Minc += Minc_Mnp; Minc += Minc_Mnn;
        //MakeColStochastic(Minc);
        t1 = MPI_Wtime();
        //Minc = Mall;

        //Minc = SpParMat<IT,NT,DER>(Mpp.getnrow() + Mnn.getnrow(), 
                     //Mpp.getncol() + Mnn.getncol(), 
                     //FullyDistVec<IT,IT>(commGrid), 
                     //FullyDistVec<IT,IT>(commGrid), 
                     //FullyDistVec<IT,IT>(commGrid), true); 
        //Minc.SpAsgn(pMap, pMap, Mpp);
        //Minc.SpAsgn(pMap, nMap, Mpn);
        //Minc.SpAsgn(nMap, pMap, Mnp);
        //Minc.SpAsgn(nMap, nMap, Mnn);
    } // All intermediate copies get deleted here

    //{
        //FullyDistVec<IT, IT> x( commGrid );
        //x.iota(pMap.TotalLength(), 0); // Intialize with consecutive numbers
        //FullyDistVec<IT, IT> y( commGrid );
        //y.iota(nMap.TotalLength(), pMap.TotalLength()); // Initialize with consecutive numbers

        //SpParMat<IT, NT, DER> MincTemp = SpParMat<IT,NT,DER>(Mpp.getnrow() + Mnn.getnrow(), Mpp.getncol() + Mnn.getncol(), 
                     //FullyDistVec<IT,IT>(commGrid), FullyDistVec<IT,IT>(commGrid), FullyDistVec<IT,IT>(commGrid), true); 

        //MincTemp.SpAsgn(x, x, Mpp);
        //MincTemp.SpAsgn(x, y, Mpn);
        //MincTemp.SpAsgn(y, x, Mnp);
        //MincTemp.SpAsgn(y, y, Mnn);
        //MincTemp(permMap, permMap, true);

        //bool eq = (MincTemp == Minc);
        //if (myrank == 0) cout << eq << endl;
        
        //fprintf(stderr, "[PrepIncMat] myrank: %d\t total nnz: %ld - %ld - %ld | local nnz: %ld - %ld - %ld\n", myrank, Minc.getnnz(), Mall.getnnz(), MincTemp.getnnz(), Minc.seqptr()->getnnz(), Mall.seqptr()->getnnz(), MincTemp.seqptr()->getnnz());
    //}
};

template <class IT, class NT, class LBL, class DER>
void IncClust(SpParMat<IT, NT, DER> &Mpp, SpParMat<IT, NT, DER> &Mpn, SpParMat<IT, NT, DER> &Mnp, SpParMat<IT, NT, DER> &Mnn, FullyDistVec<IT, LBL> &pVtxLbl, FullyDistVec<IT, LBL> &nVtxLbl, 
        FullyDistVec<IT, LBL> &aVtxLbl, FullyDistVec<IT, IT> &clustAsn, FullyDistVec<IT, IT> &aVtxMap, SpParMat<IT, NT, DER> &Mstar, int version, HipMCLParam &param){
    if(version == 1) IncClustV1(Mpp, Mpn, Mnp, Mnn, pVtxLbl, nVtxLbl, aVtxLbl, clustAsn, aVtxMap, Mstar, param);
};

