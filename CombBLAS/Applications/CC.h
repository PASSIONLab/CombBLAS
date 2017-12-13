/****************************************************************/
/* Parallel Combinatorial BLAS Library (for Graph Computations) */
/* version 1.6 -------------------------------------------------*/
/* date: 6/15/2017 ---------------------------------------------*/
/* authors: Ariful Azad, Aydin Buluc  --------------------------*/
/****************************************************************/
/*
 Copyright (c) 2010-2017, The Regents of the University of California
 
 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:
 
 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.
 
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE.
 */


#include <mpi.h>

// These macros should be defined before stdint.h is included
#ifndef __STDC_CONSTANT_MACROS
#define __STDC_CONSTANT_MACROS
#endif
#ifndef __STDC_LIMIT_MACROS
#define __STDC_LIMIT_MACROS
#endif
#include <stdint.h>

#include <sys/time.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <ctime>
#include <cmath>
#include "CombBLAS/CombBLAS.h"

/**
 ** Connected components based on Awerbuch-Shiloach algorithm
 **/

namespace combblas {

template <typename T1, typename T2>
struct Select2ndMinSR
{
    typedef typename promote_trait<T1,T2>::T_promote T_promote;
    static T_promote id(){ return std::numeric_limits<T_promote>::max(); };
    static bool returnedSAID() { return false; }
    static MPI_Op mpi_op() { return MPI_MIN; };
    
    static T_promote add(const T_promote & arg1, const T_promote & arg2)
    {
        return std::min(arg1, arg2);
    }
    
    static T_promote multiply(const T1 & arg1, const T2 & arg2)
    {
        return static_cast<T_promote> (arg2);
    }
    
    static void axpy(const T1 a, const T2 & x, T_promote & y)
    {
        y = add(y, multiply(a, x));
    }
};



template <typename IT, typename NT, typename DER>
FullyDistVec<IT,short> StarCheck(const SpParMat<IT,NT,DER> & A, FullyDistVec<IT, IT> & father)
{
    // FullyDistVec doesn't support "bool" due to crippled vector<bool> semantics
    //TODO: change the value of star entries to grandfathers so it will be IT as well.
    
    FullyDistVec<IT,short> star(A.getcommgrid(), A.getnrow(), 1);    // all initialized to true
    FullyDistVec<IT, IT> grandfather = father(father); // find grandparents
    
  
    
    grandfather.EWiseOut(father,
                         [](IT pv, IT gpv) -> short { return static_cast<short>(pv == gpv); },
                         star); // remove some stars
    
    // nostars
    FullyDistSpVec<IT,short> nonstar = star.Find([](short isStar){return isStar==0;});
      // grandfathers of nonstars
    FullyDistSpVec<IT, IT> nonstarGF = EWiseApply<IT>(nonstar, grandfather,
                                            [](short isStar, IT gf){return gf;},
                                            [](short isStar, IT gf){return true;},
                                            false, static_cast<short>(0));
    // grandfather pointing to a grandchild
    FullyDistSpVec<IT, IT> gfNonstar = nonstarGF.Invert(nonstarGF.TotalLength()); // for duplicates, keep the first one
    
    
   
    
    // star(GF) = 0
    star.EWiseApply(gfNonstar, [](short isStar, IT x){return static_cast<short>(0);},
                    false, static_cast<IT>(0));
    
    // at this point vertices at level 1 (children of the root) can still be stars
    FullyDistVec<IT,short> starFather = star(father);
    star.EWiseApply(starFather, std::multiplies<short>());
    
    /* alternative approach (used in the Matlab code)
     // fathers of nonstars
     FullyDistSpVec<IT, IT> nonstarF = EWiseApply<IT>(nonstar, father,[](short isStar, IT f){return f;}, [](short isStar, IT f){return true;},false, static_cast<short>(0));
     // father pointing to a child
     FullyDistSpVec<IT, IT> fNonstar = nonstarF.Invert(nonstarF.TotalLength());
     // star(F) = 0
     star.EWiseApply(fNonstar, [](short isStar, IT x){return static_cast<short>(0);},false, static_cast<IT>(0));
     star = star(father);
     */
    return star;
}

template <typename IT, typename NT, typename DER>
void ConditionalHook(const SpParMat<IT,NT,DER> & A, FullyDistVec<IT, IT> & father)
{
    FullyDistVec<IT,short> stars = StarCheck(A, father);
    
    FullyDistVec<IT, IT> minNeighborFather ( A.getcommgrid());
    minNeighborFather = SpMV<Select2ndMinSR<NT, IT>>(A, father); // value is the minimum of all neighbors' fathers
    FullyDistSpVec<IT, std::pair<IT, IT>> hooks(A.getcommgrid(), A.getnrow());
    // create entries belonging to stars
    hooks = EWiseApply<std::pair<IT, IT>>(hooks, stars,
                                               [](std::pair<IT, IT> x, short isStar){return std::make_pair(0,0);},
                                               [](std::pair<IT, IT> x, short isStar){return isStar==1;},
                                               true, {0,0});
    
    
    // include father information
    
    hooks = EWiseApply<std::pair<IT, IT>>(hooks, father,
                                               [](std::pair<IT, IT> x, IT f){return std::make_pair(f,0);},
                                               [](std::pair<IT, IT> x, IT f){return true;},
                                               false, {0,0});
    
    //keep entries with father>minNeighborFather and insert minNeighborFather information
    hooks = EWiseApply<std::pair<IT, IT>>(hooks,  minNeighborFather,
                                               [](std::pair<IT, IT> x, IT mnf){return std::make_pair(std::get<0>(x), mnf);},
                                               [](std::pair<IT, IT> x, IT mnf){return std::get<0>(x) > mnf;},
                                               false, {0,0});
    //Invert
    FullyDistSpVec<IT, std::pair<IT, IT>> starhooks= hooks.Invert(hooks.TotalLength(),
                                        [](std::pair<IT, IT> val, IT ind){return std::get<0>(val);},
                                        [](std::pair<IT, IT> val, IT ind){return std::make_pair(ind, std::get<1>(val));},
                                                [](std::pair<IT, IT> val1, std::pair<IT, IT> val2){return val2;} );
    // allowing the last vertex to pick the parent of the root of a star gives the correct output!!
    // [](pair<IT, IT> val1, pair<IT, IT> val2){return val1;} does not give the correct output. why?
    
    
    // drop the index informaiton
    FullyDistSpVec<IT, IT> finalhooks = EWiseApply<IT>(starhooks,  father,
                                                                      [](std::pair<IT, IT> x, IT f){return std::get<1>(x);},
                                                                      [](std::pair<IT, IT> x, IT f){return true;},
                                                                      false, {0,0});
    father.Set(finalhooks);
}

template <typename IT, typename NT, typename DER>
void UnconditionalHook(const SpParMat<IT,NT,DER> & A, FullyDistVec<IT, IT> & father)
{
    FullyDistVec<IT,short> stars = StarCheck(A, father);
    
    FullyDistVec<IT, IT> minNeighborFather ( A.getcommgrid());
    minNeighborFather = SpMV<Select2ndMinSR<NT, IT>>(A, father); // value is the minimum of all neighbors' fathers

    FullyDistSpVec<IT, std::pair<IT, IT>> hooks(A.getcommgrid(), A.getnrow());
    // create entries belonging to stars
    hooks = EWiseApply<std::pair<IT, IT>>(hooks, stars,
                                               [](std::pair<IT, IT> x, short isStar){return std::make_pair(0,0);},
                                               [](std::pair<IT, IT> x, short isStar){return isStar==1;},
                                               true, {0,0});
    
    
    // include father information
    
    hooks = EWiseApply<std::pair<IT, IT>>(hooks, father,
                                               [](std::pair<IT, IT> x, IT f){return std::make_pair(f,0);},
                                               [](std::pair<IT, IT> x, IT f){return true;},
                                               false, {0,0});
    
    //keep entries with father!minNeighborFather and insert minNeighborFather information
    hooks = EWiseApply<std::pair<IT, IT>>(hooks,  minNeighborFather,
                                               [](std::pair<IT, IT> x, IT mnf){return std::make_pair(std::get<0>(x), mnf);},
                                               [](std::pair<IT, IT> x, IT mnf){return std::get<0>(x) != mnf;},
                                               false, {0,0});
    //Invert
    FullyDistSpVec<IT, std::pair<IT, IT>> starhooks= hooks.Invert(hooks.TotalLength(),
                                                                            [](std::pair<IT, IT> val, IT ind){return std::get<0>(val);},
                                                                            [](std::pair<IT, IT> val, IT ind){return std::make_pair(ind, std::get<1>(val));},
                                                                            [](std::pair<IT, IT> val1, std::pair<IT, IT> val2){return val1;} );
    
    
    // drop the index informaiton
    FullyDistSpVec<IT, IT> finalhooks = EWiseApply<IT>(starhooks,  father,
                                                                      [](std::pair<IT, IT> x, IT f){return std::get<1>(x);},
                                                                      [](std::pair<IT, IT> x, IT f){return true;},
                                                                      false, {0,0});
    father.Set(finalhooks);
}

template <typename IT>
void Shortcut(FullyDistVec<IT, IT> & father)
{
    FullyDistVec<IT, IT> grandfather = father(father);
    father = grandfather; // we can do it unconditionally because it is trivially true for stars
}

template <typename IT, typename NT, typename DER>
bool neigborsInSameCC(const SpParMat<IT,NT,DER> & A, FullyDistVec<IT, IT> & cclabel)
{
    FullyDistVec<IT, IT> minNeighborCCLabel ( A.getcommgrid());
    minNeighborCCLabel = SpMV<Select2ndMinSR<NT, IT>>(A, cclabel);
    return minNeighborCCLabel==cclabel;
}


// works only on P=1
template <typename IT, typename NT, typename DER>
void Correctness(const SpParMat<IT,NT,DER> & A, FullyDistVec<IT, IT> & cclabel, IT nCC, FullyDistVec<IT,IT> father)
{
    DER* spSeq = A.seqptr(); // local submatrix
    
    for(auto colit = spSeq->begcol(); colit != spSeq->endcol(); ++colit) // iterate over columns
    {
        IT j = colit.colid(); // local numbering
        for(auto nzit = spSeq->begnz(colit); nzit < spSeq->endnz(colit); ++nzit)
        {
            IT i = nzit.rowid();
            if( cclabel[i] != cclabel[j])
            {
                std::cout << i << " (" << father[i] << ", "<< cclabel[i] << ") & "<< j << "("<< father[j] << ", " << cclabel[j] << ")\n";
            }
        }
        
    }
}

// Input:
// father: father of each vertex. Father is essentilly the root of the star which a vertex belongs to.
//          father of the root is itself
// Output:
// cclabel: connected components are incrementally labeled
// returns the number of connected components
// Example: input = [0, 0, 2, 3, 0, 2], output = (0, 0, 1, 2, 0, 1), return 3
template <typename IT>
IT LabelCC(FullyDistVec<IT, IT> & father, FullyDistVec<IT, IT> & cclabel)
{
    cclabel = father;
    cclabel.ApplyInd([](IT val, IT ind){return val==ind ? -1 : val;});
    FullyDistSpVec<IT, IT> roots (cclabel, bind2nd(std::equal_to<IT>(), -1));
    roots.nziota(0);
    cclabel.Set(roots);
    cclabel = cclabel(father);
    return roots.getnnz();
}

// Compute strongly connected components
// If you need weakly connected components, symmetricize the matrix beforehand
template <typename IT, typename NT, typename DER>
FullyDistVec<IT, IT> CC(SpParMat<IT,NT,DER> & A, IT & nCC)
{
    A.AddLoops(1); // needed for isolated vertices
    FullyDistVec<IT,IT> father(A.getcommgrid());
    father.iota(A.getnrow(), 0);    // father(i)=i initially
    IT nonstars = 1;
    int iteration = 0;
    std::ostringstream outs;
    while (true)
    {
#ifdef TIMING
        double t1 = MPI_Wtime();
#endif
        ConditionalHook(A, father);
        UnconditionalHook(A, father);
        Shortcut(father);
        FullyDistVec<IT,short> stars = StarCheck(A, father);
        nonstars = stars.Reduce(std::plus<IT>(), static_cast<IT>(0), [](short isStar){return static_cast<IT>(isStar==0);});
        if(nonstars==0)
        {
            if(neigborsInSameCC(A, father)) break;
            
        }
        iteration++;
#ifdef CC_TIMING
        double t2 = MPI_Wtime();
        outs.str("");
        outs.clear();
        outs << "Iteration: " << iteration << " Non stars: " << nonstars;
        outs << " Time: " << t2 - t1;
        outs<< endl;
        SpParHelper::Print(outs.str());
#endif
    }
    
    FullyDistVec<IT, IT> cc(father.getcommgrid());
    nCC = LabelCC(father, cc);
    
    // TODO: Print to file
    //PrintCC(cc, nCC);
    //Correctness(A, cc, nCC, father);
    
    return cc;
}


template <typename IT>
void PrintCC(FullyDistVec<IT, IT> CC, IT nCC)
{
    for(IT i=0; i< nCC; i++)
    {
        FullyDistVec<IT, IT> ith = CC.FindInds(bind2nd(std::equal_to<IT>(), i));
        ith.DebugPrint();
    }
}

// Print the size of the first 4 clusters
template <typename IT>
void First4Clust(FullyDistVec<IT, IT>& cc)
{
    FullyDistSpVec<IT, IT> cc1 = cc.Find([](IT label){return label==0;});
    FullyDistSpVec<IT, IT> cc2 = cc.Find([](IT label){return label==1;});
    FullyDistSpVec<IT, IT> cc3 = cc.Find([](IT label){return label==2;});
    FullyDistSpVec<IT, IT> cc4 = cc.Find([](IT label){return label==3;});
    
    std::ostringstream outs;
    outs.str("");
    outs.clear();
    outs << "Size of the first component: " << cc1.getnnz() << std::endl;
    outs << "Size of the second component: " << cc2.getnnz() << std::endl;
    outs << "Size of the third component: " << cc3.getnnz() << std::endl;
    outs << "Size of the fourth component: " << cc4.getnnz() << std::endl;
}


template <typename IT>
void HistCC(FullyDistVec<IT, IT> CC, IT nCC)
{
    FullyDistVec<IT, IT> ccSizes(CC.getcommgrid(), nCC, 0);
    for(IT i=0; i< nCC; i++)
    {
        FullyDistSpVec<IT, IT> ith = CC.Find(bind2nd(std::equal_to<IT>(), i));
        ccSizes.SetElement(i, ith.getnnz());
    }
    
    IT largestCCSise = ccSizes.Reduce(maximum<IT>(), static_cast<IT>(0));
    
    
    const IT * locCCSizes = ccSizes.GetLocArr();
    int numBins = 200;
    std::vector<IT> localHist(numBins,0);
    for(IT i=0; i< ccSizes.LocArrSize(); i++)
    {
        IT bin = (locCCSizes[i]*(numBins-1))/largestCCSise;
        localHist[bin]++;
    }
    
    std::vector<IT> globalHist(numBins,0);
    MPI_Comm world = CC.getcommgrid()->GetWorld();
    MPI_Reduce(localHist.data(), globalHist.data(), numBins, MPIType<IT>(), MPI_SUM, 0, world);
    
    
    int myrank;
    MPI_Comm_rank(world,&myrank);
    if(myrank==0)
    {
        std::cout << "The largest component size: " << largestCCSise  << std::endl;
        std::ofstream output;
        output.open("hist.txt", std::ios_base::app );
        std::copy(globalHist.begin(), globalHist.end(), std::ostream_iterator<IT> (output, " "));
        output << std::endl;
        output.close();
    }

    
    //ccSizes.PrintToFile("histCC.txt");
}

}

