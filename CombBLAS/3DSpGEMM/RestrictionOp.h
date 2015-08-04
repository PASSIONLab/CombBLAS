//#define DETERMINISTIC
#include "../CombBLAS.h"
#ifdef THREADED
	#ifndef _OPENMP
	#define _OPENMP
	#endif
	#include <omp.h>
#endif

using namespace std;



#ifdef DETERMINISTIC
MTRand GlobalMT(1);
#else
MTRand GlobalMT;	// generate random numbers with Mersenne Twister 
#endif




template <typename T1, typename T2>
struct Select2ndMinSR
{
    static T2 id(){ return T2(); };
    static bool returnedSAID() { return false; }
    static MPI_Op mpi_op() { return MPI_MIN; };
    
    static T2 add(const T2 & arg1, const T2 & arg2)
    {
        return std::min(arg1, arg2);
    }
    
    static T2 multiply(const T1 & arg1, const T2 & arg2)
    {
        return arg2;
    }
    
    static void axpy(const T1 a, const T2 & x, T2 & y)
    {
        y = add(y, multiply(a, x));
    }
};



// second hop MIS (i.e., MIS on A^2)
template <typename ONT, typename IT, typename INT, typename DER>
FullyDistSpVec<IT, ONT> MIS2(SpParMat < IT, INT, DER> A)
{
    IT nvert = A.getncol();
    
    //# the final result set. S[i] exists and is 1 if vertex i is in the MIS
    FullyDistSpVec<IT, ONT> mis ( A.getcommgrid(), nvert);
    
    
    //# the candidate set. initially all vertices are candidates.
    //# If cand[i] exists, then i is a candidate. The value cand[i] is i's random number for this iteration.
    FullyDistVec<IT, double> dcand( A.getcommgrid(), nvert, 1.0);
    FullyDistSpVec<IT, double> cand (dcand);
    FullyDistSpVec<IT, double> min_neighbor_r ( A.getcommgrid(), nvert);
    FullyDistSpVec<IT, double> min_neighbor2_r ( A.getcommgrid(), nvert);
    
    FullyDistSpVec<IT, ONT> new_S_members ( A.getcommgrid(), nvert);
    FullyDistSpVec<IT, ONT> new_S_neighbors ( A.getcommgrid(), nvert);
    FullyDistSpVec<IT, ONT> new_S_neighbors2 ( A.getcommgrid(), nvert);
    
    while (cand.getnnz() > 0)
    {
        
        //# label each vertex in cand with a random value
        cand.Apply([](const double & ignore){return (double) GlobalMT.rand();});
        
        //# find the smallest random value among a vertex's neighbors
        SpMV<Select2ndMinSR<INT, double>>(A, cand, min_neighbor_r, false);
        SpMV<Select2ndMinSR<INT, double>>(A, min_neighbor_r, min_neighbor2_r, false);
        
        
        //# The vertices to be added to S this iteration are those whose random value is
        //# smaller than those of all its neighbors:
        // **** if cand has isolated vertices, they will be included in new_S_members ******
        new_S_members = EWiseApply<ONT>(min_neighbor2_r, cand,
                                        [](double x, double y){return (ONT)1;},
                                        [](double x, double y){return y<=x;}, // equality is for back edges since we are operating on A^2
                                        true, false, 2.0,  2.0, true);
        
        //# new_S_members are no longer candidates, so remove them from cand
        cand = EWiseApply<double>(cand, new_S_members,
                                  [](double x, ONT y){return x;},
                                  [](double x, ONT y){return true;},
                                  false, true, 0.0, (ONT) 0, false);
        
        //# find 2-hop neighbors of new_S_members
        SpMV<Select2ndMinSR<INT, ONT>>(A, new_S_members, new_S_neighbors, false);
        SpMV<Select2ndMinSR<INT, ONT>>(A, new_S_neighbors, new_S_neighbors2, false);
        
        
        //# remove neighbors of new_S_members from cand, because they cannot be part of the MIS anymore
        cand = EWiseApply<double>(cand, new_S_neighbors2,
                                  [](double x, ONT y){return x;},
                                  [](double x, ONT y){return true;},
                                  false, true, 0.0, (ONT) 0, false);
        
        //# add new_S_members to mis
        mis = EWiseApply<ONT>(mis, new_S_members,
                              [](ONT x, ONT y){return x;},
                              [](ONT x, ONT y){return true;},
                              true, true, (ONT) 1, (ONT) 1, true);
    }
    
    return mis;
}

/*
template <typename IT, typename NT>
SpDCCols<IT, NT>* RestrictionOp( CCGrid & CMG, SpDCCols<IT, NT> * localmat)
{
    //if(CMG.layer_grid == 0)
    {
        SpDCCols<IT, bool> *A = new SpDCCols<IT, bool>(*localmat);
        SpParMat < IT, bool, SpDCCols < IT, bool >> B (A, CMG.layerWorld);

	B.RemoveLoops();

        SpParMat < IT, bool, SpDCCols < IT, bool >> BT = B;
        BT.Transpose();
        B += BT;
        
        // ------------ compute MIS-2 ----------------------------
        FullyDistSpVec<IT, IT> mis2 (B.getcommgrid(), B.getncol());
        mis2 = MIS2<IT>(B);
       	mis2.DebugPrint(); 
        // ------------ Obtain restriction matrix from mis2 ----
        FullyDistVec<IT, IT> ri = mis2.FindInds([](IT x){return true;});
        FullyDistVec<IT, IT> ci(B.getcommgrid());
        ci.iota(mis2.getnnz(), (IT)0);
        SpParMat < IT, NT, SpDCCols < IT, NT >> M(B.getnrow(), ci.TotalLength(), ri, ci, (NT)1, false);
        SpParMat < IT, NT, SpDCCols < IT, NT >> R = PSpGEMM<PlusTimesSRing<bool, NT>>(B,M);
        R += M;
        
        return R.seqptr();

    }
    //else
      //  return new SpDCCols<IT,NT>();
}
*/

template <typename ONT, typename IT, typename LIT, typename NT>
SpParMat < IT, ONT, SpDCCols < LIT, ONT >> RestrictionOp( SpParMat < IT, NT, SpDCCols < LIT, NT >> A)
{
    
    SpParMat < IT, bool, SpDCCols < LIT, bool >> B = SpParMat < IT, bool, SpDCCols < LIT, bool >> (A);
    B.RemoveLoops();
    SpParMat < IT, bool, SpDCCols < LIT, bool >> BT = B;
    BT.Transpose();
    B += BT;
    
    // ------------ compute MIS-2 ----------------------------
    FullyDistSpVec<IT, IT> mis2 (B.getcommgrid(), B.getncol());
    mis2 = MIS2<IT>(B);
	mis2.DebugPrint();    
    // ------------ Obtain restriction matric from mis2 ----
    FullyDistVec<IT, IT> ri = mis2.FindInds([](IT x){return true;});
    FullyDistVec<IT, IT> ci(A.getcommgrid());
    ci.iota(mis2.getnnz(), (IT)0);
    SpParMat < IT, ONT, SpDCCols < LIT, ONT >> M(A.getnrow(), ci.TotalLength(), ri, ci, (ONT) 1, false);
    SpParMat < IT, ONT, SpDCCols < LIT, ONT >> R = PSpGEMM<PlusTimesSRing<bool, ONT>>(B,M);
    R += M;

    return R;
    
}

