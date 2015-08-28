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

template <typename T1, typename T2>
struct MIS2verifySR // identical to Select2ndMinSR except for the printout in add()
{
    static T2 id(){ return T2(); };
    static bool returnedSAID() { return false; }
    static MPI_Op mpi_op() { return MPI_MIN; };
    
    static T2 add(const T2 & arg1, const T2 & arg2)
    {
        cout << "This should have never been executed for MIS-2 to be correct" << endl;
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



// second hop MIS (i.e., MIS on A \Union A^2)
template <typename ONT, typename IT, typename INT, typename DER>
FullyDistSpVec<IT, ONT> MIS2(SpParMat < IT, INT, DER> A)
{
    IT nvert = A.getncol();
    
    //# the final result set. S[i] exists and is 1 if vertex i is in the MIS
    FullyDistSpVec<IT, ONT> mis ( A.getcommgrid(), nvert);
    

    //# the candidate set. initially all vertices are candidates.
    //# If cand[i] exists, then i is a candidate. The value cand[i] is i's random number for this iteration.
    FullyDistSpVec<IT, double> cand(A.getcommgrid());
    cand.iota(nvert, 1.0); // any value is fine since we randomize it later
    FullyDistSpVec<IT, double> min_neighbor_r ( A.getcommgrid(), nvert);
    FullyDistSpVec<IT, double> min_neighbor2_r ( A.getcommgrid(), nvert);
    
    
    FullyDistSpVec<IT, ONT> new_S_members ( A.getcommgrid(), nvert);
    FullyDistSpVec<IT, ONT> new_S_neighbors ( A.getcommgrid(), nvert);
    FullyDistSpVec<IT, ONT> new_S_neighbors2 ( A.getcommgrid(), nvert);
    
    
    while (cand.getnnz() > 0)
    {
        
        //# label each vertex in cand with a random value (in what range, [0,1]?)
        cand.Apply([](const double & ignore){return (double) GlobalMT.rand();});
        
        //# find the smallest random value among a vertex's 1 and 2-hop neighbors
        SpMV<Select2ndMinSR<INT, double>>(A, cand, min_neighbor_r, false);
        SpMV<Select2ndMinSR<INT, double>>(A, min_neighbor_r, min_neighbor2_r, false);
        
        FullyDistSpVec<IT, double> min_neighbor_r_union = EWiseApply<double>(min_neighbor2_r, min_neighbor_r,
                                        [](double x, double y){return std::min(x,y);},
                                        [](double x, double y){return true;},   // do_op is totalogy
                                        true, true, 2.0,  2.0, true);   // we allow nulls for both V and W
        
        
        //# The vertices to be added to S this iteration are those whose random value is
        //# smaller than those of all its 1-hop and 2-hop neighbors:
        // **** if cand has isolated vertices, they will be included in new_S_members ******
        new_S_members = EWiseApply<ONT>(min_neighbor_r_union, cand,
                                        [](double x, double y){return (ONT)1;},
                                        [](double x, double y){return y<=x;}, // equality is for back edges since we are operating on A^2
                                        true, false, 2.0,  2.0, true);
        
        //# new_S_members are no longer candidates, so remove them from cand
        cand = EWiseApply<double>(cand, new_S_members,
                                  [](double x, ONT y){return x;},
                                  [](double x, ONT y){return true;},
                                  false, true, 0.0, (ONT) 0, false);
        
        //# find 1-hop and 2-hop neighbors of new_S_members
        SpMV<Select2ndMinSR<INT, ONT>>(A, new_S_members, new_S_neighbors, false);
        SpMV<Select2ndMinSR<INT, ONT>>(A, new_S_neighbors, new_S_neighbors2, false);
        
        FullyDistSpVec<IT, ONT> new_S_neighbors_union = EWiseApply<ONT>(new_S_neighbors, new_S_neighbors2,
                                                                          [](ONT x, ONT y){return x;},  // in case of intersection, doesn't matter which one to propagate
                                                                          [](ONT x, ONT y){return true;},
                                                                          true, true, (ONT) 1, (ONT) 1, true);

        
        //# remove 1-hop and 2-hop neighbors of new_S_members from cand, because they cannot be part of the MIS anymore
        cand = EWiseApply<double>(cand, new_S_neighbors_union,
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



template <typename IT, typename NT>
void RestrictionOp( CCGrid & CMG, SpDCCols<IT, NT> * localmat, SpDCCols<IT, NT> *& R, SpDCCols<IT, NT> *& RT)
{
    if(CMG.layer_grid == 0)
    {
        SpDCCols<IT, bool> *A = new SpDCCols<IT, bool>(*localmat);
        
        SpParMat < IT, bool, SpDCCols < IT, bool >> B (A, CMG.layerWorld);
        
        B.RemoveLoops();
        
        SpParMat < IT, bool, SpDCCols < IT, bool >> BT = B;
        BT.Transpose();
        B += BT;
        
        // ------------ compute MIS-2 ----------------------------
        FullyDistSpVec<IT, IT> mis2 (B.getcommgrid(), B.getncol()); // values of the mis2 vector are just "ones"
        mis2 = MIS2<IT>(B);
       	mis2.PrintInfo("MIS original");
        // ------------ Obtain restriction matrix from mis2 ----
        FullyDistVec<IT,IT> ri = mis2.FindInds([](IT x){return true;});
        
        // find the vertices that are not covered by mis2 AND its one hop neighborhood
        FullyDistSpVec<IT,IT> mis2neigh = SpMV<MIS2verifySR<bool, IT>>(B, mis2, false);
        mis2neigh.PrintInfo("MIS neighbors");
        
        // ABAB: mis2 and mis2neigh should be independent, because B doesn't have any loops.
        FullyDistSpVec<IT,IT> isection = EWiseApply<IT>(mis2neigh, mis2,
                                                        [](IT x, IT y){return x;},
                                                        [](IT x, IT y){return true;},
                                                        false, false, (IT) 1, (IT) 1, true);  /// allowVNulls and allowWNulls are both false
        isection.PrintInfo("intersection of mis2neigh and mis2");
        
        
        // find the union of mis2neigh and mis2 (ABAB: this function to be wrapped & called "SetUnion")
        mis2neigh = EWiseApply<IT>(mis2neigh, mis2,
                              [](IT x, IT y){return x;},
                              [](IT x, IT y){return true;},
                              true, true, (IT) 1, (IT) 1, true);
        mis2neigh.PrintInfo("MIS original+neighbors");


        // FullyDistVec<IT, NT>::FullyDistVec ( shared_ptr<CommGrid> grid, IT globallen, NT initval)
        //       : FullyDist<IT,NT,typename CombBLAS::disable_if< CombBLAS::is_boolean<NT>::value, NT >::type>(grid,globallen)
        
        FullyDistVec<IT, IT> denseones(mis2neigh.getcommgrid(), B.getncol(), 1);    // calls the default constructor... why?
        FullyDistSpVec<IT,IT> spones (denseones);
        
        // subtract the entries of mis2neigh from all vertices (ABAB: this function to be wrapped & called "SetDiff")
        spones = EWiseApply<IT>(spones, mis2neigh,
                                  [](IT x, IT y){return x;},   // binop
                                  [](IT x, IT y){return true;}, // doop
                                  false, true, (IT) 1, (IT) 1, false);  // allowintersect=false (all joint entries are removed)
        
        spones.PrintInfo("Leftovers (singletons)");
        
        
        FullyDistVec<IT, IT> ci(B.getcommgrid());
        ci.iota(mis2.getnnz(), (IT)0);
        SpParMat<IT,NT,SpDCCols<IT,NT>> M(B.getnrow(), ci.TotalLength(), ri, ci, (NT)1, false);
        
        SpParHelper::Print("M matrix... ");
        M.PrintInfo();
        
        SpParMat<IT,NT,SpDCCols<IT,NT>> Rop = PSpGEMM<PlusTimesSRing<bool, NT>>(B,M);
        
        SpParHelper::Print("R (minus M) matrix... ");
        Rop.PrintInfo();
        
        Rop += M;
        
        SpParHelper::Print("R without singletons... ");
        Rop.PrintInfo();
        
        FullyDistVec<IT,IT> rrow(Rop.getcommgrid());
        FullyDistVec<IT,IT> rcol(Rop.getcommgrid());
        FullyDistVec<IT,NT> rval(Rop.getcommgrid());
        Rop.Find(rrow, rcol, rval);
        
        FullyDistVec<IT, IT> extracols(Rop.getcommgrid());
        extracols.iota(spones.getnnz(), ci.TotalLength()); // one column per singleton
        
        // Returns a dense vector of nonzero global indices for which the predicate is satisfied on values
        FullyDistVec<IT,IT> extrarows = spones.FindInds([](IT x){return true;});   // dense leftovers array is the extra rows
        
        // Resize Rop
        SpParMat<IT,NT,SpDCCols<IT,NT>> RopFull1(Rop.getnrow(), Rop.getncol() +extracols.TotalLength(), rrow, rcol, rval, false);

        SpParHelper::Print("RopFull1... ");
        RopFull1.PrintInfo();
        
        
        SpParMat<IT,NT,SpDCCols<IT,NT>> RopT = Rop;
        RopT.Transpose();
        
        R = new SpDCCols<IT,NT>(Rop.seq()); // deep copy
        RT = new SpDCCols<IT,NT>(RopT.seq()); // deep copy
        
    }
}



/*
template <typename IT, typename NT, typename DER>
SpParMat < IT, NT, DER> RestrictionOp( SpParMat < IT, NT, DER> A)
{
    
    SpParMat < IT, bool, SpDCCols < IT, bool >> B = SpParMat < IT, bool, SpDCCols < IT, bool >> (A);
    B.RemoveLoops();
    SpParMat < IT, bool, SpDCCols < IT, bool >> BT = B;
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
    SpParMat < IT, NT, DER> M(A.getnrow(), ci.TotalLength(), ri, ci, (NT) 1, false);
    SpParMat < IT, NT, DER> R = PSpGEMM<PlusTimesSRing<bool, NT>>(B,M);
    R += M;
    
    return R;
    
}
*/




