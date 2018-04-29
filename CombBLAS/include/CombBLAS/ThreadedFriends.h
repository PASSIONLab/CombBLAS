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


#ifndef _THREADED_FRIENDS_H_
#define _THREADED_FRIENDS_H_

#include <iostream>
#include "SpMat.h"	// Best to include the base class first
#include "SpHelper.h"
#include "StackEntry.h"
#include "Isect.h"
#include "Deleter.h"
#include "SpImpl.h"
#include "SpParHelper.h"
#include "Compare.h"
#include "CombBLAS.h"
#include "PreAllocatedSPA.h"

namespace combblas {

template <class IU, class NU>	
class SpTuples;

template <class IU, class NU>	
class SpDCCols;

template <class IU, class NU>	
class Dcsc;


// multithreaded HeapSpGEMM
template <typename SR, typename NTO, typename IT, typename NT1, typename NT2>
SpTuples<IT, NTO> * LocalSpGEMM
(const SpDCCols<IT, NT1> & A,
 const SpDCCols<IT, NT2> & B,
 bool clearA, bool clearB)
{
    IT mdim = A.getnrow();
    IT ndim = B.getncol();
    IT nnzA = A.getnnz();
    if(A.isZero() || B.isZero())
    {
        return new SpTuples<IT, NTO>(0, mdim, ndim);
    }
    
    Dcsc<IT,NT1>* Adcsc = A.GetDCSC();
    Dcsc<IT,NT2>* Bdcsc = B.GetDCSC();
    IT nA = A.getncol();
    IT cnzmax = Adcsc->nz + Bdcsc->nz;	// estimate on the size of resulting matrix C
    float cf  = static_cast<float>(nA+1) / static_cast<float>(Adcsc->nzc);
    IT csize = static_cast<IT>(ceil(cf));   // chunk size
    IT * aux;
    Adcsc->ConstructAux(nA, aux);
    
    int numThreads = 1;	// default case
#ifdef THREADED
#pragma omp parallel
    {
        numThreads = omp_get_num_threads();
    }
#endif
    
    IT* colnnzC = estimateNNZ(A, B);
    IT* colptrC = prefixsum<IT>(colnnzC, Bdcsc->nzc, numThreads);
    delete [] colnnzC;
    IT nnzc = colptrC[Bdcsc->nzc];
    std::tuple<IT,IT,NTO> * tuplesC = static_cast<std::tuple<IT,IT,NTO> *> (::operator new (sizeof(std::tuple<IT,IT,NTO>[nnzc])));
    
    // thread private space for heap and colinds
    std::vector<std::vector< std::pair<IT,IT>>> colindsVec(numThreads);
    std::vector<std::vector<HeapEntry<IT,NT1>>> globalheapVec(numThreads);
    
    for(int i=0; i<numThreads; i++) //inital allocation per thread, may be an overestimate, but does not require more memoty than inputs
    {
        colindsVec[i].resize(nnzA/numThreads);
        globalheapVec[i].resize(nnzA/numThreads);
    }
    
    
#pragma omp parallel for
    for(int i=0; i < Bdcsc->nzc; ++i)
    {
        IT nnzcolB = Bdcsc->cp[i+1] - Bdcsc->cp[i]; //nnz in the current column of B
        int myThread = omp_get_thread_num();
        if(colindsVec[myThread].size() < nnzcolB) //resize thread private vectors if needed
        {
            colindsVec[myThread].resize(nnzcolB);
            globalheapVec[myThread].resize(nnzcolB);
        }
        
        
        // colinds.first vector keeps indices to A.cp, i.e. it dereferences "colnums" vector (above),
        // colinds.second vector keeps the end indices (i.e. it gives the index to the last valid element of A.cpnack)
        Adcsc->FillColInds(Bdcsc->ir + Bdcsc->cp[i], nnzcolB, colindsVec[myThread], aux, csize);
        std::pair<IT,IT> * colinds = colindsVec[myThread].data();
        HeapEntry<IT,NT1> * wset = globalheapVec[myThread].data();
        IT hsize = 0;
        
        
        for(IT j = 0; (unsigned)j < nnzcolB; ++j)		// create the initial heap
        {
            if(colinds[j].first != colinds[j].second)	// current != end
            {
                wset[hsize++] = HeapEntry< IT,NT1 > (Adcsc->ir[colinds[j].first], j, Adcsc->numx[colinds[j].first]);
            }
        }
        std:make_heap(wset, wset+hsize);
        
        IT curptr = colptrC[i];
        while(hsize > 0)
        {
            std::pop_heap(wset, wset + hsize);         // result is stored in wset[hsize-1]
            IT locb = wset[hsize-1].runr;	// relative location of the nonzero in B's current column
            
            NTO mrhs = SR::multiply(wset[hsize-1].num, Bdcsc->numx[Bdcsc->cp[i]+locb]);
            if (!SR::returnedSAID())
            {
                if( (curptr > colptrC[i]) && std::get<0>(tuplesC[curptr-1]) == wset[hsize-1].key)
                {
                  std::get<2>(tuplesC[curptr-1]) = SR::add(std::get<2>(tuplesC[curptr-1]), mrhs);
                }
                else
                {
                    tuplesC[curptr++]= std::make_tuple(wset[hsize-1].key, Bdcsc->jc[i], mrhs) ;
                }
                
            }
            
            if( (++(colinds[locb].first)) != colinds[locb].second)	// current != end
            {
                // runr stays the same !
                wset[hsize-1].key = Adcsc->ir[colinds[locb].first];
                wset[hsize-1].num = Adcsc->numx[colinds[locb].first];
                std::push_heap(wset, wset+hsize);
            }
            else
            {
                --hsize;
            }
        }
    }
    
    if(clearA)
        delete const_cast<SpDCCols<IT, NT1> *>(&A);
    if(clearB)
        delete const_cast<SpDCCols<IT, NT2> *>(&B);
    
    delete [] colptrC;
    delete [] aux;
    
    SpTuples<IT, NTO>* spTuplesC = new SpTuples<IT, NTO> (nnzc, mdim, ndim, tuplesC, true);
    return spTuplesC;
    
}

}

#endif
