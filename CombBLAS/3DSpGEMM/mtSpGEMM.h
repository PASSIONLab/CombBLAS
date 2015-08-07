#ifndef _mtSpGEMM_h
#define _mtSpGEMM_h

#include "../CombBLAS.h"


template <typename T>
T* prefixsum(T* in, int size, int nthreads)
{
    vector<T> tsum(nthreads+1);
    tsum[0] = 0;
    T* out = new T[size+1];
    out[0] = 0;
    T* psum = &out[1];
    
#pragma omp parallel
    {
        int ithread = omp_get_thread_num();
        T sum = 0;
#pragma omp for schedule(static)
        for (int i=0; i<size; i++)
        {
            sum += in[i];
            psum[i] = sum;
        }
        
        tsum[ithread+1] = sum;
#pragma omp barrier
        T offset = 0;
        for(int i=0; i<(ithread+1); i++)
        {
            offset += tsum[i];
        }
#pragma omp for schedule(static)
        for (int i=0; i<size; i++)
        {
            psum[i] += offset;
        }
    
    }
    return out;
}




// multithreaded
template <typename SR, typename NTO, typename IT, typename NT1, typename NT2>
SpTuples<IT, NTO> * LocalSpGEMM
(const SpDCCols<IT, NT1> & A,
 const SpDCCols<IT, NT2> & B,
 bool clearA, bool clearB)
{
    IT mdim = A.getnrow();
    IT ndim = B.getncol();
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
    
   
    
    
    // *************** Creating global space to store result, used by all threads *********************
    
    IT* maxnnzc = new IT[Bdcsc->nzc]; // maximum number of nnz in each column of C
    IT flops = 0; // total flops (multiplication) needed to generate C
#pragma omp parallel
    {
        IT tflops=0; //thread private flops
#pragma omp for
        for(int i=0; i < Bdcsc->nzc; ++i)
        {
            IT locmax = 0;
            IT nnzcol = Bdcsc->cp[i+1] - Bdcsc->cp[i];
            //vector< pair<IT,IT> > colinds(nnzcol);
            //Adcsc->FillColInds(Bdcsc->ir + Bdcsc->cp[i], nnzcol, colinds, aux, csize);
            bool found;
            IT* curptr = Bdcsc->ir + Bdcsc->cp[i];
            
            for(IT j = 0; j < nnzcol; ++j)
            {
                IT pos = Adcsc->AuxIndex(curptr[j], found, aux, csize);
                if(found)
                {
                    locmax = locmax + (Adcsc->cp[pos+1] - Adcsc->cp[pos]);
                }
                //locmax = locmax + (colinds[j].second - colinds[j].first);
            }
            maxnnzc[i] = locmax;
            tflops += locmax;
        }
#pragma omp critical
        {
            flops += tflops;
        }
    }
    
    
    int numThreads;
#pragma omp parallel
    {
        numThreads = omp_get_num_threads();
    }
    
    IT colPerThread [numThreads + 1]; // thread i will process columns from colPerThread[i] to colPerThread[i+1]-1
    colPerThread[0] = 0;
    
    
    IT* colStart = prefixsum<IT>(maxnnzc, Bdcsc->nzc, numThreads);
#pragma omp parallel for
    for(int i=1; i< numThreads; i++)
    {
        IT cur_col = i * (flops/numThreads);
        IT* it = std::lower_bound (colStart, colStart+Bdcsc->nzc+1, cur_col);
        colPerThread[i] = it - colStart;
        if(colPerThread[i]>Bdcsc->nzc) colPerThread[i]=Bdcsc->nzc;
    }
    colPerThread[numThreads] = Bdcsc->nzc;
    
    
    IT size = colStart[Bdcsc->nzc-1] + maxnnzc[Bdcsc->nzc-1];
    tuple<IT,IT,NTO> * tuplesC = static_cast<tuple<IT,IT,NTO> *> (::operator new (sizeof(tuple<IT,IT,NTO>[size])));
    
    delete [] maxnnzc;
    // ************************ End Creating global space *************************************
    
    // *************** Creating global heap space to be used by all threads *********************
    IT threadHeapSize[numThreads];
#pragma omp parallel
    {
        int thisThread = omp_get_thread_num();
        IT localmax = 0;
        for(int i=colPerThread[thisThread]; i < colPerThread[thisThread+1]; ++i)
        {
            IT colnnz = Bdcsc->cp[i+1]-Bdcsc->cp[i];
            if(colnnz > localmax) localmax = colnnz;
        }
        threadHeapSize[thisThread] = localmax;
    }
    
    IT threadHeapStart[numThreads+1];
    threadHeapStart[0] = 0;
    for(int i=0; i<numThreads; i++)
    threadHeapStart[i+1] = threadHeapStart[i] + threadHeapSize[i];
    HeapEntry<IT,NT1> * globalheap = new HeapEntry<IT,NT1>[threadHeapStart[numThreads]];
    //HeapEntry<IT,NT1> * colinds1 = new HeapEntry<IT,NT1>[threadHeapStart[numThreads]];
    
    // ************************ End Creating global heap space *************************************
    IT* colEnd = new IT[Bdcsc->nzc]; //end index in the global array for storing ith column of C

#pragma omp parallel
    {
        int thisThread = omp_get_thread_num();
        vector< pair<IT,IT> > colinds(threadHeapSize[thisThread]);  //
        HeapEntry<IT,NT1> * wset = globalheap + threadHeapStart[thisThread]; // thread private heap space
        
        for(int i=colPerThread[thisThread]; i < colPerThread[thisThread+1]; ++i)
        {
            
            
            IT nnzcol = Bdcsc->cp[i+1] - Bdcsc->cp[i];
            colEnd[i] = colStart[i];
            
            // colinds.first vector keeps indices to A.cp, i.e. it dereferences "colnums" vector (above),
            // colinds.second vector keeps the end indices (i.e. it gives the index to the last valid element of A.cpnack)
            Adcsc->FillColInds(Bdcsc->ir + Bdcsc->cp[i], nnzcol, colinds, aux, csize); // can be done multithreaded
            IT hsize = 0;
            
            for(IT j = 0; (unsigned)j < nnzcol; ++j)		// create the initial heap
            {
                if(colinds[j].first != colinds[j].second)	// current != end
                {
                    wset[hsize++] = HeapEntry< IT,NT1 > (Adcsc->ir[colinds[j].first], j, Adcsc->numx[colinds[j].first]);
                }
            }
            make_heap(wset, wset+hsize);
            
            
            while(hsize > 0)
            {
                pop_heap(wset, wset + hsize);         // result is stored in wset[hsize-1]
                IT locb = wset[hsize-1].runr;	// relative location of the nonzero in B's current column
                
                NTO mrhs = SR::multiply(wset[hsize-1].num, Bdcsc->numx[Bdcsc->cp[i]+locb]);
                if (!SR::returnedSAID())
                {
                    if( (colEnd[i] > colStart[i]) && get<0>(tuplesC[colEnd[i]-1]) == wset[hsize-1].key)
                    {
                        get<2>(tuplesC[colEnd[i]-1]) = SR::add(get<2>(tuplesC[colEnd[i]-1]), mrhs);
                    }
                    else
                    {
                        tuplesC[colEnd[i]]= make_tuple(wset[hsize-1].key, Bdcsc->jc[i], mrhs) ;
                        colEnd[i] ++;
                    }
                    
                }
                
                if( (++(colinds[locb].first)) != colinds[locb].second)	// current != end
                {
                    // runr stays the same !
                    wset[hsize-1].key = Adcsc->ir[colinds[locb].first];
                    wset[hsize-1].num = Adcsc->numx[colinds[locb].first];
                    push_heap(wset, wset+hsize);
                }
                else
                {
                    --hsize;
                }
            }
        }
        
    }
    delete [] aux;
    delete [] globalheap;
    
    
    vector<IT> nnzcol(Bdcsc->nzc);
#pragma omp parallel for
    for(IT i=0; i< Bdcsc->nzc; ++i)
    {
        nnzcol[i] = colEnd[i]-colStart[i];
    }
    
    IT* colptrC = prefixsum<IT>(nnzcol.data(), Bdcsc->nzc, numThreads); //parallel
    
    IT nnzc = colptrC[Bdcsc->nzc];
    tuple<IT,IT,NTO> * tuplesOut = static_cast<tuple<IT,IT,NTO> *> (::operator new (sizeof(tuple<IT,IT,NTO>[nnzc])));
    
#pragma omp parallel for
    for(IT i=0; i< Bdcsc->nzc; ++i)
    {
        copy(&tuplesC[colStart[i]], &tuplesC[colEnd[i]], tuplesOut + colptrC[i]);
    }

    if(clearA)
        delete const_cast<SpDCCols<IT, NT1> *>(&A);
    if(clearB)
        delete const_cast<SpDCCols<IT, NT2> *>(&B);
    
    ::operator delete(tuplesC);
    delete [] colStart;
    delete [] colEnd;
    delete [] colptrC;
    
    SpTuples<IT, NTO>* spTuplesC = new SpTuples<IT, NTO> (nnzc, mdim, ndim, tuplesOut, true);
    return spTuplesC;
}


#endif
