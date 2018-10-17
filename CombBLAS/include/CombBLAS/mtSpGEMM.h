#ifndef _mtSpGEMM_h
#define _mtSpGEMM_h

#include "CombBLAS.h"

namespace combblas {
/*
 Multithreaded prefix sum
 Inputs:
    in: an input array
    size: the length of the input array "in"
    nthreads: number of threads used to compute the prefix sum
 
 Output:
    return an array of size "size+1"
    the memory of the output array is allocated internallay
 
 Example:
 
    in = [2, 1, 3, 5]
    out = [0, 2, 3, 6, 11]
 */
template <typename T>
T* prefixsum(T* in, int size, int nthreads)
{
    std::vector<T> tsum(nthreads+1);
    tsum[0] = 0;
    T* out = new T[size+1];
    out[0] = 0;
    T* psum = &out[1];
#ifdef THREADED
#pragma omp parallel
#endif
    {
		int ithread = 0;
	#ifdef THREADED
        ithread = omp_get_thread_num();
	#endif

        T sum = 0;
#ifdef THREADED
#pragma omp for schedule(static)
#endif
        for (int i=0; i<size; i++)
        {
            sum += in[i];
            psum[i] = sum;
        }
        
        tsum[ithread+1] = sum;
#ifdef THREADED
#pragma omp barrier
#endif
        T offset = 0;
        for(int i=0; i<(ithread+1); i++)
        {
            offset += tsum[i];
        }
		
#ifdef THREADED
#pragma omp for schedule(static)
#endif
        for (int i=0; i<size; i++)
        {
            psum[i] += offset;
        }
    
    }
    return out;
}


template <typename IT, typename NT>
bool sort_less(const std::pair<IT, NT> &left, const std::pair<IT, NT> &right)
{
    return left.first < right.first;
}



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
    float cf  = static_cast<float>(nA+1) / static_cast<float>(Adcsc->nzc);
    IT csize = static_cast<IT>(ceil(cf));   // chunk size
    IT * aux;
    Adcsc->ConstructAux(nA, aux);

	
    int numThreads = 1;
#ifdef THREADED
#pragma omp parallel
    {
        numThreads = omp_get_num_threads();
    }
#endif
   
    std::cout << "numThreads: " << numThreads << std::endl;

    IT* flopC = estimateFLOP(A, B);
    IT* flopptr = prefixsum<IT>(flopC, Bdcsc->nzc, numThreads);
    IT flop = flopptr[Bdcsc->nzc];
    std::cout << "FLOP of A * B is " << flop << std::endl;
    delete [] flopC;

    IT* colnnzC = estimateNNZ(A, B);
    IT* colptrC = prefixsum<IT>(colnnzC, Bdcsc->nzc, numThreads);
    delete [] colnnzC;
    IT nnzc = colptrC[Bdcsc->nzc];
    std::cout << "NNZ of A * B is " << nnzc << std::endl;

    std::cout << "Compression ratio is " << (float)flop / nnzc << std::endl;

    std::tuple<IT,IT,NTO> * tuplesC = static_cast<std::tuple<IT,IT,NTO> *> (::operator new (sizeof(std::tuple<IT,IT,NTO>[nnzc])));
	
    if (1)  // Heap SpGEMM
    {
        // thread private space for heap and colinds
        std::vector<std::vector< std::pair<IT,IT>>> colindsVec(numThreads);
        std::vector<std::vector<HeapEntry<IT,NT1>>> globalheapVec(numThreads);
    
        for(int i=0; i<numThreads; i++) //inital allocation per thread, may be an overestimate, but does not require more memoty than inputs
        {
            colindsVec[i].resize(nnzA/numThreads);
            globalheapVec[i].resize(nnzA/numThreads);
        }


#ifdef THREADED
#pragma omp parallel for
#endif
        for(int i=0; i < Bdcsc->nzc; ++i)
        {
            size_t nnzcolB = Bdcsc->cp[i+1] - Bdcsc->cp[i]; //nnz in the current column of B
            int myThread = 0;
#ifdef THREADED
            myThread = omp_get_thread_num();
#endif
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
            std::make_heap(wset, wset+hsize);
        
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
    }
    
    else // Hash SpGEMM
    {
        const IT minHashTableSize = 16;
        const IT hashScale = 107;

        // thread private space for hash tables and colinds
        std::vector<std::vector< std::pair<IT,IT>>> colindsVec(numThreads);
        std::vector<std::vector< std::pair<IT,NT1>>> globalHashVec(numThreads);

        for(int i=0; i<numThreads; i++) //inital allocation per thread, for colindsVec, may be an overestimate, but does not require more memoty than inputs. For globalHashVec, this is just minimum.
        {
            colindsVec[i].resize(nnzA/numThreads);
            globalHashVec[i].resize(minHashTableSize);
        }

#ifdef THREADED
#pragma omp parallel for
#endif
        for(IT i=0; i < Bdcsc->nzc; ++i)
        {
            int myThread = 0;
#ifdef THREADED
            myThread = omp_get_thread_num();
#endif
            size_t nnzcolB = Bdcsc->cp[i+1] - Bdcsc->cp[i]; //nnz in the current column of B
            size_t nnzcolC = colptrC[i+1] - colptrC[i]; //nnz in the current column of C (=Output)
            if(colindsVec[myThread].size() < nnzcolB) //resize thread private vectors if needed
            {
                colindsVec[myThread].resize(nnzcolB);
            }

            IT ht_size = minHashTableSize;
            while(ht_size < nnzcolC) //ht_size is set as 2^n
            {
                ht_size <<= 1;
            }
            if(globalHashVec[myThread].size() < ht_size)
            {
                globalHashVec[myThread].resize(ht_size);
            }

            // colinds.first vector keeps indices to A.cp, i.e. it dereferences "colnums" vector (above),
            // colinds.second vector keeps the end indices (i.e. it gives the index to the last valid element of A.cpnack)
            Adcsc->FillColInds(Bdcsc->ir + Bdcsc->cp[i], nnzcolB, colindsVec[myThread], aux, csize);
            std::pair<IT,IT> * colinds = colindsVec[myThread].data();
            
            // Initialize hash tables
            for(IT j=0; (unsigned)j < ht_size; ++j)
            {
                globalHashVec[myThread][j].first = -1;
            }
            
            // Multiply and add on Hash table
            for (IT j=0; (unsigned)j < nnzcolB; ++j)
            {
                IT t_bcol = Bdcsc->ir[Bdcsc->cp[i] + j];
                NT2 t_bval = Bdcsc->numx[Bdcsc->cp[i] + j];
                for (IT k = colinds[j].first; (unsigned)k < colinds[j].second; ++k)
                {
                    NTO mrhs = SR::multiply(Adcsc->numx[k], t_bval);
                    IT key = Adcsc->ir[k];
                    IT hash = (key*hashScale) & (ht_size-1);
                    while (1) //hash probing
                    {
                        if (globalHashVec[myThread][hash].first == key) //key is found in hash table
                        {
                            globalHashVec[myThread][hash].second = SR::add(mrhs, globalHashVec[myThread][hash].second);
                            break;
                        }
                        else if (globalHashVec[myThread][hash].first == -1) //key is not registered yet
                        {
                            globalHashVec[myThread][hash].first = key;
                            globalHashVec[myThread][hash].second = mrhs;
                            break;
                        }
                        else //key is not found
                        {
                            hash = (hash+1) & (ht_size-1);
                        }
                    }
                }
            }
            // gather non-zero elements from hash table, and then sort them by row indices
            IT index = 0;
            for (IT j=0; j < ht_size; ++j)
            {
                if (globalHashVec[myThread][j].first != -1)
                {
                    globalHashVec[myThread][index++] = globalHashVec[myThread][j];
                }
            }
            std::sort(globalHashVec[myThread].begin(), globalHashVec[myThread].begin() + index, sort_less<IT, NT1>);

            IT curptr = colptrC[i];
            for (IT j=0; j < index; ++j)
            {
                tuplesC[curptr++]= std::make_tuple(globalHashVec[myThread][j].first, Bdcsc->jc[i], globalHashVec[myThread][j].second);
            }
        }
    }

    if(clearA)
        delete const_cast<SpDCCols<IT, NT1> *>(&A);
    if(clearB)
        delete const_cast<SpDCCols<IT, NT2> *>(&B);
    
    delete [] colptrC;
    delete [] flopptr;
    delete [] aux;
    
    SpTuples<IT, NTO>* spTuplesC = new SpTuples<IT, NTO> (nnzc, mdim, ndim, tuplesC, true, true);
    return spTuplesC;
}

// estimate space for result of SpGEMM
template <typename IT, typename NT1, typename NT2>
IT* estimateNNZ(const SpDCCols<IT, NT1> & A,const SpDCCols<IT, NT2> & B)
{
    IT nnzA = A.getnnz();
    if(A.isZero() || B.isZero())
    {
        return NULL;
    }
    
    Dcsc<IT,NT1>* Adcsc = A.GetDCSC();
    Dcsc<IT,NT2>* Bdcsc = B.GetDCSC();
    
    float cf  = static_cast<float>(A.getncol()+1) / static_cast<float>(Adcsc->nzc);
    IT csize = static_cast<IT>(ceil(cf));   // chunk size
    IT * aux;
    Adcsc->ConstructAux(A.getncol(), aux);
	
	
    int numThreads = 1;
#ifdef THREADED
#pragma omp parallel
    {
        numThreads = omp_get_num_threads();
    }
#endif
    

    IT* colnnzC = new IT[Bdcsc->nzc]; // nnz in every nonempty column of C
	
#ifdef THREADED
#pragma omp parallel for
#endif
    for(IT i=0; i< Bdcsc->nzc; ++i)
    {
        colnnzC[i] = 0;
    }
    
    // thread private space for heap and colinds
    std::vector<std::vector< std::pair<IT,IT>>> colindsVec(numThreads);
    std::vector<std::vector<std::pair<IT,IT>>> globalheapVec(numThreads);

	
    for(int i=0; i<numThreads; i++) //inital allocation per thread, may be an overestimate, but does not require more memoty than inputs
    {
        colindsVec[i].resize(nnzA/numThreads);
        globalheapVec[i].resize(nnzA/numThreads);
    }

#ifdef THREADED
#pragma omp parallel for
#endif
    for(int i=0; i < Bdcsc->nzc; ++i)
    {
        size_t nnzcolB = Bdcsc->cp[i+1] - Bdcsc->cp[i]; //nnz in the current column of B
		int myThread = 0;
#ifdef THREADED
        myThread = omp_get_thread_num();
#endif
        if(colindsVec[myThread].size() < nnzcolB) //resize thread private vectors if needed
        {
            colindsVec[myThread].resize(nnzcolB);
            globalheapVec[myThread].resize(nnzcolB);
        }
		
        // colinds.first vector keeps indices to A.cp, i.e. it dereferences "colnums" vector (above),
        // colinds.second vector keeps the end indices (i.e. it gives the index to the last valid element of A.cpnack)
        Adcsc->FillColInds(Bdcsc->ir + Bdcsc->cp[i], nnzcolB, colindsVec[myThread], aux, csize);
        std::pair<IT,IT> * colinds = colindsVec[myThread].data();
        std::pair<IT,IT> * curheap = globalheapVec[myThread].data();
        IT hsize = 0;
        
        // create the initial heap
        for(IT j = 0; (unsigned)j < nnzcolB; ++j)
        {
            if(colinds[j].first != colinds[j].second)
            {
                curheap[hsize++] = std::make_pair(Adcsc->ir[colinds[j].first], j);
            }
        }
        std::make_heap(curheap, curheap+hsize, std::greater<std::pair<IT,IT>>());
        
        IT prevRow=-1; // previously popped row from heap
		
        while(hsize > 0)
        {
          std::pop_heap(curheap, curheap + hsize, std::greater<std::pair<IT,IT>>()); // result is stored in wset[hsize-1]
            IT locb = curheap[hsize-1].second;
            
            if( curheap[hsize-1].first != prevRow)
            {
                prevRow = curheap[hsize-1].first;
                colnnzC[i] ++;
            }
            
            if( (++(colinds[locb].first)) != colinds[locb].second)	// current != end
            {
                curheap[hsize-1].first = Adcsc->ir[colinds[locb].first];
                std::push_heap(curheap, curheap+hsize, std::greater<std::pair<IT,IT>>());
            }
            else
            {
                --hsize;
            }
        }
    }
    
    delete [] aux;
    return colnnzC;
}

// estimate space for result of SpGEMM
template <typename IT, typename NT1, typename NT2>
IT* estimateFLOP(const SpDCCols<IT, NT1> & A,const SpDCCols<IT, NT2> & B)
{
    IT nnzA = A.getnnz();
    if(A.isZero() || B.isZero())
    {
        return NULL;
    }
    
    Dcsc<IT,NT1>* Adcsc = A.GetDCSC();
    Dcsc<IT,NT2>* Bdcsc = B.GetDCSC();
    
    float cf  = static_cast<float>(A.getncol()+1) / static_cast<float>(Adcsc->nzc);
    IT csize = static_cast<IT>(ceil(cf));   // chunk size
    IT * aux;
    Adcsc->ConstructAux(A.getncol(), aux);
	
	
    int numThreads = 1;
#ifdef THREADED
#pragma omp parallel
    {
        numThreads = omp_get_num_threads();
    }
#endif
    

    IT* colflopC = new IT[Bdcsc->nzc]; // flop in every nonempty column of C
	
#ifdef THREADED
#pragma omp parallel for
#endif
    for(IT i=0; i< Bdcsc->nzc; ++i)
    {
        colflopC[i] = 0;
    }
    
    


    // thread private space for heap and colinds
    std::vector<std::vector< std::pair<IT,IT>>> colindsVec(numThreads);

	
    for(int i=0; i<numThreads; i++) //inital allocation per thread, may be an overestimate, but does not require more memoty than inputs
    {
        colindsVec[i].resize(nnzA/numThreads);
    }

#ifdef THREADED
#pragma omp parallel for
#endif
    for(int i=0; i < Bdcsc->nzc; ++i)
    {
        size_t nnzcolB = Bdcsc->cp[i+1] - Bdcsc->cp[i]; //nnz in the current column of B
		int myThread = 0;
#ifdef THREADED
        myThread = omp_get_thread_num();
#endif
        if(colindsVec[myThread].size() < nnzcolB) //resize thread private vectors if needed
        {
            colindsVec[myThread].resize(nnzcolB);
        }
		
        // colinds.first vector keeps indices to A.cp, i.e. it dereferences "colnums" vector (above),
        // colinds.second vector keeps the end indices (i.e. it gives the index to the last valid element of A.cpnack)
        Adcsc->FillColInds(Bdcsc->ir + Bdcsc->cp[i], nnzcolB, colindsVec[myThread], aux, csize);
        for (IT j = 0; (unsigned)j < nnzcolB; ++j) {
            colflopC[i] += colindsVec[myThread][j].second - colindsVec[myThread][j].first;
        }
    }
    
    delete [] aux;
    return colflopC;
}

}

#endif
