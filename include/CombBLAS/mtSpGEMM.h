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
   
    IT* colnnzC = estimateNNZ(A, B, aux,false);	// don't free aux	
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

    size_t Bnzc = (size_t) Bdcsc->nzc;
#ifdef THREADED
#pragma omp parallel for
#endif
    for(size_t i=0; i < Bnzc; ++i)
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
        
        
        for(size_t j = 0; j < nnzcolB; ++j)		// create the initial heap
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

    if(clearA)
        delete const_cast<SpDCCols<IT, NT1> *>(&A);
    if(clearB)
        delete const_cast<SpDCCols<IT, NT2> *>(&B);
    
    delete [] colptrC;
    delete [] aux;
    
    SpTuples<IT, NTO>* spTuplesC = new SpTuples<IT, NTO> (nnzc, mdim, ndim, tuplesC, true, true);
    return spTuplesC;
    
}



template <typename IT, typename NT>
bool sort_less(const std::pair<IT, NT> &left, const std::pair<IT, NT> &right)
{
    return left.first < right.first;
}

// Hybrid approach of multithreaded HeapSpGEMM and HashSpGEMM
template <typename SR, typename NTO, typename IT, typename NT1, typename NT2>
SpTuples<IT, NTO> * LocalHybridSpGEMM
(const SpDCCols<IT, NT1> & A,
 const SpDCCols<IT, NT2> & B,
 bool clearA, bool clearB, IT * aux = nullptr)
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
    //IT * aux;
    bool deleteAux = false;
    if(aux==nullptr)
    {
	deleteAux = true;
    	Adcsc->ConstructAux(nA, aux);
    }
	
    int numThreads = 1;
#ifdef THREADED
#pragma omp parallel
    {
        numThreads = omp_get_num_threads();
    }
#endif
   
    // std::cout << "numThreads: " << numThreads << std::endl;

    IT* flopC =  estimateFLOP(A, B, aux);
    //IT* flopptr = prefixsum<IT>(flopC, Bdcsc->nzc, numThreads);
    //IT flop = flopptr[Bdcsc->nzc];
    // std::cout << "FLOP of A * B is " << flop << std::endl;


    IT* colnnzC = estimateNNZ_Hash(A, B, flopC, aux);
    IT* flopptr = prefixsum<IT>(flopC, Bdcsc->nzc, numThreads);
    IT flop = flopptr[Bdcsc->nzc];
    IT* colptrC = prefixsum<IT>(colnnzC, Bdcsc->nzc, numThreads);
    delete [] colnnzC;
    delete [] flopC;
    IT nnzc = colptrC[Bdcsc->nzc];
    //double compression_ratio = (double)flop / nnzc;


    // std::cout << "NNZ of A * B is " << nnzc << std::endl;
    // std::cout << "Compression ratio is " << compression_ratio << std::endl;

    std::tuple<IT,IT,NTO> * tuplesC = static_cast<std::tuple<IT,IT,NTO> *> (::operator new (sizeof(std::tuple<IT,IT,NTO>[nnzc])));
   //std::tuple<IT,IT,NTO> * tuplesC = new std::tuple<IT,IT,NTO>[nnzc];
       
    // thread private space for heap and colinds
    std::vector<std::vector< std::pair<IT,IT>>> colindsVec(numThreads);
   
     std::vector<std::vector< std::pair<IT,NTO>>> globalHashVecAll(numThreads); 
     std::vector<std::vector< HeapEntry<IT,NT1>>> globalHeapVecAll(numThreads);
    /*
    for(int i=0; i<numThreads; i++) //inital allocation per thread, may be an overestimate, but does not require more memoty than inputs
    {
        colindsVec[i].resize(nnzA/numThreads);
    }*/

    // IT hashSelected = 0;


#ifdef THREADED
#pragma omp parallel for
#endif
    for(size_t i=0; i < Bdcsc->nzc; ++i)
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
        std::pair<IT,IT> * colinds = colindsVec[myThread].data();

        double cr = static_cast<double>(flopptr[i+1] - flopptr[i]) / (colptrC[i+1] - colptrC[i]);
        if (cr < 2.0) // Heap Algorithm
        {
	    if(globalHeapVecAll[myThread].size() < nnzcolB)
	    	globalHeapVecAll[myThread].resize(nnzcolB);	    
            //std::vector<HeapEntry<IT,NT1>> globalheapVec(nnzcolB);
            //HeapEntry<IT, NT1> * wset = globalheapVec.data();
            HeapEntry<IT, NT1> * wset = globalHeapVecAll[myThread].data();

            IT hsize = 0;
        
            for(size_t j = 0; j < nnzcolB; ++j)		// create the initial heap
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
        } // Finish Heap
        
        else // Hash Algorithm
        {
	// #pragma omp atomic
	// 	hashSelected++;
            const IT minHashTableSize = 16;
            const IT hashScale = 107;
            size_t nnzcolC = colptrC[i+1] - colptrC[i]; //nnz in the current column of C (=Output)

            size_t ht_size = minHashTableSize;
            while(ht_size < nnzcolC) //ht_size is set as 2^n
            {
                ht_size <<= 1;
            }
            
	    
	    if(globalHashVecAll[myThread].size() < ht_size)
                globalHashVecAll[myThread].resize(ht_size);
            //std::vector<HeapEntry<IT,NT1>> globalheapVec(nnzcolB);
            //HeapEntry<IT, NT1> * wset = globalheapVec.data();
            //HeapEntry<IT, NT1> * wset = globalheapVecAll[myThread].data();

	    //std::vector< std::pair<IT,NTO>> globalHashVec(ht_size);
	    std::pair<IT,NTO>* globalHashVec =  globalHashVecAll[myThread].data();
            // colinds.first vector keeps indices to A.cp, i.e. it dereferences "colnums" vector (above),
            // colinds.second vector keeps the end indices (i.e. it gives the index to the last valid element of A.cpnack)
            
            // Initialize hash tables
            for(size_t j=0; j < ht_size; ++j)
            {
                globalHashVec[j].first = -1;
            }
            
            // Multiply and add on Hash table
            for (size_t j=0; j < nnzcolB; ++j)
            {
                IT t_bcol = Bdcsc->ir[Bdcsc->cp[i] + j];
                NT2 t_bval = Bdcsc->numx[Bdcsc->cp[i] + j];
                for (IT k = colinds[j].first; k < colinds[j].second; ++k)
                {
                    NTO mrhs = SR::multiply(Adcsc->numx[k], t_bval);
                    IT key = Adcsc->ir[k];
                    IT hash = (key*hashScale) & (ht_size-1);
                    while (1) //hash probing
                    {
                        if (globalHashVec[hash].first == key) //key is found in hash table
                        {
                            globalHashVec[hash].second = SR::add(mrhs, globalHashVec[hash].second);
                            break;
                        }
                        else if (globalHashVec[hash].first == -1) //key is not registered yet
                        {
                            globalHashVec[hash].first = key;
                            globalHashVec[hash].second = mrhs;
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
            size_t index = 0;
            for (size_t j=0; j < ht_size; ++j)
            {
                if (globalHashVec[j].first != -1)
                {
                    globalHashVec[index++] = globalHashVec[j];
                }
            }
            //std::sort(globalHashVec.begin(), globalHashVec.begin() + index, sort_less<IT, NTO>);
	    std::sort(globalHashVecAll[myThread].begin(), globalHashVecAll[myThread].begin() + index, sort_less<IT, NTO>);
            IT curptr = colptrC[i];
            for (size_t j=0; j < index; ++j)
            {
                tuplesC[curptr++]= std::make_tuple(globalHashVec[j].first, Bdcsc->jc[i], globalHashVec[j].second);
            }
        }
    }
    
    if(clearA)
        delete const_cast<SpDCCols<IT, NT1> *>(&A);
    if(clearB)
        delete const_cast<SpDCCols<IT, NT2> *>(&B);
    
    delete [] colptrC;
    delete [] flopptr;
    if(deleteAux)
    	delete [] aux;
    
    SpTuples<IT, NTO>* spTuplesC = new SpTuples<IT, NTO> (nnzc, mdim, ndim, tuplesC, true, true);


    // std::cout << "localspgemminfo," << flop << "," << nnzc << "," << compression_ratio << "," << t1-t0 << std::endl;
    // std::cout << hashSelected << ", " << Bdcsc->nzc << ", " << (float)hashSelected / Bdcsc->nzc << std::endl;

    return spTuplesC;
}

    // Hybrid approach of multithreaded HeapSpGEMM and HashSpGEMM
    template <typename SR, typename NTO, typename IT, typename NT1, typename NT2>
    SpTuples<IT, NTO> * LocalSpGEMMHash
    (const SpDCCols<IT, NT1> & A,
     const SpDCCols<IT, NT2> & B,
     bool clearA, bool clearB, bool sort=true)
    {

        double t0=MPI_Wtime();

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

        // std::cout << "numThreads: " << numThreads << std::endl;

        IT* flopC = estimateFLOP(A, B);
        IT* flopptr = prefixsum<IT>(flopC, Bdcsc->nzc, numThreads);
        IT flop = flopptr[Bdcsc->nzc];
        // std::cout << "FLOP of A * B is " << flop << std::endl;

        IT* colnnzC = estimateNNZ_Hash(A, B, flopC);
        IT* colptrC = prefixsum<IT>(colnnzC, Bdcsc->nzc, numThreads);
        delete [] colnnzC;
        delete [] flopC;
        IT nnzc = colptrC[Bdcsc->nzc];
        double compression_ratio = (double)flop / nnzc;

        // std::cout << "NNZ of A * B is " << nnzc << std::endl;
        // std::cout << "Compression ratio is " << compression_ratio << std::endl;

        // std::tuple<IT,IT,NTO> * tuplesC = static_cast<std::tuple<IT,IT,NTO> *> (::operator new (sizeof(std::tuple<IT,IT,NTO>[nnzc])));
        std::tuple<IT,IT,NTO> * tuplesC = new std::tuple<IT,IT,NTO>[nnzc];

        // thread private space for heap and colinds
        std::vector<std::vector< std::pair<IT,IT>>> colindsVec(numThreads);

        for(int i=0; i<numThreads; i++) //inital allocation per thread, may be an overestimate, but does not require more memoty than inputs
        {
            colindsVec[i].resize(nnzA/numThreads);
        }

        // IT hashSelected = 0;

#ifdef THREADED
#pragma omp parallel for
#endif
        for(size_t i=0; i < Bdcsc->nzc; ++i)
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
            std::pair<IT,IT> * colinds = colindsVec[myThread].data();




            // #pragma omp atomic
            //     hashSelected++;
            const IT minHashTableSize = 16;
            const IT hashScale = 107;
            size_t nnzcolC = colptrC[i+1] - colptrC[i]; //nnz in the current column of C (=Output)

            size_t ht_size = minHashTableSize;
            while(ht_size < nnzcolC) //ht_size is set as 2^n
            {
                ht_size <<= 1;
            }
            std::vector< std::pair<IT,NTO>> globalHashVec(ht_size);

            // colinds.first vector keeps indices to A.cp, i.e. it dereferences "colnums" vector (above),
            // colinds.second vector keeps the end indices (i.e. it gives the index to the last valid element of A.cpnack)

            // Initialize hash tables
            for(size_t j=0; j < ht_size; ++j)
            {
                globalHashVec[j].first = -1;
            }

            // Multiply and add on Hash table
            for (size_t j=0; j < nnzcolB; ++j)
            {
                IT t_bcol = Bdcsc->ir[Bdcsc->cp[i] + j];
                NT2 t_bval = Bdcsc->numx[Bdcsc->cp[i] + j];
                for (IT k = colinds[j].first; k < colinds[j].second; ++k)
                {
                    NTO mrhs = SR::multiply(Adcsc->numx[k], t_bval);
                    IT key = Adcsc->ir[k];
                    IT hash = (key*hashScale) & (ht_size-1);
                    while (1) //hash probing
                    {
                        if (globalHashVec[hash].first == key) //key is found in hash table
                        {
                            globalHashVec[hash].second = SR::add(mrhs, globalHashVec[hash].second);
                            break;
                        }
                        else if (globalHashVec[hash].first == -1) //key is not registered yet
                        {
                            globalHashVec[hash].first = key;
                            globalHashVec[hash].second = mrhs;
                            break;
                        }
                        else //key is not found
                        {
                            hash = (hash+1) & (ht_size-1);
                        }
                    }
                }
            }
            
	    if(sort)
            {
            // gather non-zero elements from hash table, and then sort them by row indices
            size_t index = 0;
            for (size_t j=0; j < ht_size; ++j)
            {   
                if (globalHashVec[j].first != -1)
                {   
                    globalHashVec[index++] = globalHashVec[j];
                }
            }
            std::sort(globalHashVec.begin(), globalHashVec.begin() + index, sort_less<IT, NTO>);

            IT curptr = colptrC[i];
            for (size_t j=0; j < index; ++j)
            {
                tuplesC[curptr++]= std::make_tuple(globalHashVec[j].first, Bdcsc->jc[i], globalHashVec[j].second);
            }
            }
	    else
            {
		IT curptr = colptrC[i];
                for (size_t j=0; j < ht_size; ++j)
                {
                    if (globalHashVec[j].first != -1)
                    {
                        tuplesC[curptr++]= std::make_tuple(globalHashVec[j].first, Bdcsc->jc[i], globalHashVec[j].second);
                    }
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

        SpTuples<IT, NTO>* spTuplesC = new SpTuples<IT, NTO> (nnzc, mdim, ndim, tuplesC, true, false);

        double t1=MPI_Wtime();

        // std::cout << "localspgemminfo," << flop << "," << nnzc << "," << compression_ratio << "," << t1-t0 << std::endl;
        // std::cout << hashSelected << ", " << Bdcsc->nzc << ", " << (float)hashSelected / Bdcsc->nzc << std::endl;

        return spTuplesC;
    }

/*
 *  Estimates total flops necessary to multiply A and B
 *  Then returns the number
 * */
template <typename SR, typename IT, typename NT1, typename NT2>
IT EstimateLocalFLOP
(const SpDCCols<IT, NT1> & A,
 const SpDCCols<IT, NT2> & B,
 bool clearA, bool clearB)
{
    Dcsc<IT,NT1>* Adcsc = A.GetDCSC();
    Dcsc<IT,NT2>* Bdcsc = B.GetDCSC();
    int numThreads = 1;
#ifdef THREADED
#pragma omp parallel
    {
        numThreads = omp_get_num_threads();
    }
#endif
    IT* flopC = estimateFLOP(A, B);
    IT* flopptr = prefixsum<IT>(flopC, Bdcsc->nzc, numThreads);
    IT flop = flopptr[Bdcsc->nzc];
    delete [] flopC;

    if(clearA)
        delete const_cast<SpDCCols<IT, NT1> *>(&A);
    if(clearB)
        delete const_cast<SpDCCols<IT, NT2> *>(&B);
    
    delete [] flopptr;
    return flop;
}

// estimate space for result of SpGEMM
template <typename IT, typename NT1, typename NT2>
IT* estimateNNZ(const SpDCCols<IT, NT1> & A,const SpDCCols<IT, NT2> & B, IT * aux = nullptr, bool freeaux = true)
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
    if(aux == nullptr)
    {
	    Adcsc->ConstructAux(A.getncol(), aux);
    }
	
	
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
    
    if (freeaux) delete [] aux;
    return colnnzC;
}


// estimate space for result of SpGEMM with Hash
template <typename IT, typename NT1, typename NT2>
IT* estimateNNZ_Hash(const SpDCCols<IT, NT1> & A,const SpDCCols<IT, NT2> & B, IT *flopC, IT * aux=nullptr)
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
    bool deleteAux = false;
    if(aux==nullptr)
    {
	deleteAux = true; 
    	Adcsc->ConstructAux(A.getncol(), aux);
    }	
	
    int numThreads = 1;
#ifdef THREADED
#pragma omp parallel
    {
        numThreads = omp_get_num_threads();
    }
#endif
    

    IT* colnnzC = new IT[Bdcsc->nzc]; // nnz in every nonempty column of C


	
    /*
#ifdef THREADED
#pragma omp parallel for
#endif
    for(IT i=0; i< Bdcsc->nzc; ++i)
    {
        colnnzC[i] = 0;
    }
    */
    // thread private space for heap and colinds
    std::vector<std::vector< std::pair<IT,IT>>> colindsVec(numThreads);
    std::vector<std::vector< IT>> globalHashVecAll(numThreads);
    /*
    for(int i=0; i<numThreads; i++) //inital allocation per thread, may be an overestimate, but does not require more memoty than inputs
    {
        colindsVec[i].resize(nnzA/numThreads);
    }*/

#ifdef THREADED
#pragma omp parallel for
#endif
    for(int i=0; i < Bdcsc->nzc; ++i)
    {
	colnnzC[i] = 0;    
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
        std::pair<IT,IT> * colinds = colindsVec[myThread].data();

        // Hash
        const IT minHashTableSize = 16;
        const IT hashScale = 107;
	
        // Initialize hash tables
        IT ht_size = minHashTableSize;
        while(ht_size < flopC[i]) //ht_size is set as 2^n
        {
            ht_size <<= 1;
        }

	if(globalHashVecAll[myThread].size() < ht_size) //resize thread private vectors if needed
        {
            globalHashVecAll[myThread].resize(ht_size);
        }

        IT* globalHashVec = globalHashVecAll[myThread].data();

        for(IT j=0; (unsigned)j < ht_size; ++j)
        {
            globalHashVec[j] = -1;
        }
            
        for (IT j=0; (unsigned)j < nnzcolB; ++j)
        {
            IT t_bcol = Bdcsc->ir[Bdcsc->cp[i] + j];
            for (IT k = colinds[j].first; (unsigned)k < colinds[j].second; ++k)
            {
                IT key = Adcsc->ir[k];
                IT hash = (key*hashScale) & (ht_size-1);
                while (1) //hash probing
                {
                    if (globalHashVec[hash] == key) //key is found in hash table
                    {
                        break;
                    }
                    else if (globalHashVec[hash] == -1) //key is not registered yet
                    {
                        globalHashVec[hash] = key;
                        colnnzC[i] ++;
                        break;
                    }
                    else //key is not found
                    {
                        hash = (hash+1) & (ht_size-1);
                    }
                }
            }
        }
    }
    
    if(deleteAux)
    	delete [] aux;
    return colnnzC;
}

// sampling-based nnz estimation (within SUMMA)
template <typename IT, typename NT1, typename NT2>
int64_t
estimateNNZ_sampling(
    const SpDCCols<IT, NT1> &A,
	const SpDCCols<IT, NT2> &B,
	int						 nrounds = 5
	)
{
	IT nnzA = A.getnnz();
    if (A.isZero() || B.isZero())
        return 0;

	Dcsc<IT,NT1>    *Adcsc	 = A.GetDCSC();
    Dcsc<IT,NT2>    *Bdcsc	 = B.GetDCSC();
	float			 lambda	 = 1.0f;
	float			 usedmem = 0.0f;
	IT				 m		 = A.getnrow();
	IT				 p		 = A.getncol();
	float			*samples_init, *samples_mid, *samples_final;
	float			*colest;

	// samples
	samples_init = (float *) malloc(m * nrounds * sizeof(*samples_init));
	samples_mid	 = (float *) malloc(p * nrounds * sizeof(*samples_mid));

	int nthds = 1;
	#ifdef THREADED
	#pragma omp parallel
	#endif
	{
		nthds = omp_get_num_threads();
	}	

	#ifdef THREADED
	#pragma omp parallel
	#endif
	{
		std::default_random_engine gen;
		std::exponential_distribution<float> exp_dist(lambda);

		#ifdef THREADED
		#pragma omp parallel for
		#endif
		for (IT i = 0; i < m * nrounds; ++i)
			samples_init[i] = exp_dist(gen);
	}

	#ifdef THREADED
	#pragma omp parallel for
	#endif
	for (IT i = 0; i < p * nrounds; ++i)
		samples_mid[i] = std::numeric_limits<float>::max();

	#ifdef THREADED
	#pragma omp parallel for
	#endif
	for (IT i = 0; i < Adcsc->nzc; ++i)
	{
		IT	col		= Adcsc->jc[i];
		IT	beg_mid = col * nrounds;
		for (IT j = Adcsc->cp[i]; j < Adcsc->cp[i + 1]; ++j)
		{
			IT	row		 = Adcsc->ir[j];
			IT	beg_init = row * nrounds;
			for (int k = 0; k < nrounds; ++k)
			{
				if (samples_init[beg_init + k] < samples_mid[beg_mid + k])
					samples_mid[beg_mid + k] = samples_init[beg_init + k];
			}
		}
	}

	free(samples_init);

	samples_final = (float *) malloc(B.getnzc() * nrounds *
									 sizeof(*samples_final));
	colest		  = (float *) malloc(B.getnzc() * sizeof(*colest));

	float nnzest = 0.0f;
	
	#ifdef THREADED
	#pragma omp parallel for reduction (+:nnzest)
	#endif
	for (IT i = 0; i < Bdcsc->nzc; ++i)
	{
		int tid = 0;
		#ifdef THREADED
        tid = omp_get_thread_num();
		#endif

		IT beg_final = i * nrounds;
		for (IT k = beg_final; k < beg_final + nrounds; ++k)
			samples_final[k] = std::numeric_limits<float>::max();
		
		for (IT j = Bdcsc->cp[i]; j < Bdcsc->cp[i + 1]; ++j)
		{
			IT	row		= Bdcsc->ir[j];
			IT	beg_mid = row * nrounds;
			for (int k = 0; k < nrounds; ++k)
			{
				if (samples_mid[beg_mid + k] < samples_final[beg_final + k])
					samples_final[beg_final + k] = samples_mid[beg_mid + k];
			}
		}

		colest[i] = 0.0f;
		for (IT k = beg_final; k < beg_final + nrounds; ++k)
			colest[i] += samples_final[k];
		colest[i] = static_cast<float>(nrounds - 1) / colest[i];

		nnzest += colest[i];
	}

	free(samples_mid);
	free(samples_final);
	free(colest);
	
	return static_cast<int64_t>(nnzest);
}

// estimate the number of floating point operations of SpGEMM
template <typename IT, typename NT1, typename NT2>
IT* estimateFLOP(const SpDCCols<IT, NT1> & A,const SpDCCols<IT, NT2> & B, IT * aux = nullptr)
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
    //IT * aux;
    bool deleteAux = false;
    if(aux==nullptr)
    {
        deleteAux = true;
        Adcsc->ConstructAux(A.getncol(), aux);
    }
	
	
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

    /*	
    for(int i=0; i<numThreads; i++) //inital allocation per thread, may be an overestimate, but does not require more memoty than inputs
    {
        colindsVec[i].resize(nnzA/numThreads);
    }*/

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
    if(deleteAux)
    	delete [] aux;
    return colflopC;
}




////////////////////////////////////////////////////////////////////////////////
//////////////////////////// CSC-based local SpGEMM	////////////////////////////
////////////////////////////////////////////////////////////////////////////////
template <typename SR,
		  typename NTO,
		  typename IT,
		  typename NT1,
		  typename NT2>
SpTuples <IT, NTO> *
LocalHybridSpGEMM (const SpCCols<IT, NT1>	&A,
				   const SpCCols<IT, NT2>	&B,
				   bool						 clearA,
				   bool						 clearB
				   )
{
	double t0 = MPI_Wtime();

	IT mdim = A.getnrow();
    IT ndim = B.getncol();
    IT nnzA = A.getnnz();

	if(A.isZero() || B.isZero())
		return new SpTuples<IT, NTO>(0, mdim, ndim);

	Csc<IT, NT1> *Acsc = A.GetCSC();
	Csc<IT, NT2> *Bcsc = B.GetCSC();

	int numThreads = 1;
	#ifdef THREADED
	#pragma omp parallel
	{
		numThreads = omp_get_num_threads();
	}
	#endif

	IT	*flopC	 = estimateFLOP(A, B);
	IT	*flopptr = prefixsum<IT>(flopC, Bcsc->n, numThreads);
	IT	 flop	 = flopptr[Bcsc->n];
	
	IT *colnnzC = estimateNNZ_Hash(A, B, flopC);
    IT *colptrC = prefixsum<IT>(colnnzC, Bcsc->n, numThreads);
    delete [] colnnzC;
    delete [] flopC;
	IT		nnzc			  = colptrC[Bcsc->n];
	double	compression_ratio = (double)flop / nnzc;
	
	std::tuple<IT, IT, NTO> *tuplesC = static_cast<std::tuple<IT, IT, NTO> *>
		(::operator new (sizeof(std::tuple<IT, IT, NTO>[nnzc])));

	#ifdef THREADED
	#pragma omp parallel for
	#endif
	for (size_t i = 0; i < Bcsc->n; ++i)
	{
		size_t nnzcolB = Bcsc->jc[i + 1] - Bcsc->jc[i];
		double cr = static_cast<double>
			(flopptr[i+1] - flopptr[i]) / (colptrC[i+1] - colptrC[i]);
		if (cr < 2.0) // Heap Algorithm
		{
			std::vector<IT> cnt(nnzcolB);
			std::vector<HeapEntry<IT, NT1>> globalheapVec(nnzcolB);
            HeapEntry<IT, NT1> *wset = globalheapVec.data();
			IT hsize = 0;
			for (IT j = Bcsc->jc[i]; j < Bcsc->jc[i + 1]; ++j)
			{
				IT ca = Bcsc->ir[j];
				cnt[j - Bcsc->jc[i]] = Acsc->jc[ca];
				if (Acsc->jc[ca] != Acsc->jc[ca + 1]) // col not empty
					wset[hsize++] = HeapEntry<IT, NT1>
						(Acsc->ir[Acsc->jc[ca]], j, Acsc->num[Acsc->jc[ca]]);
			}

			std::make_heap(wset, wset + hsize);

			IT curptr = colptrC[i];
			while (hsize > 0)
			{
				std::pop_heap(wset, wset + hsize);
				IT	locb = wset[hsize - 1].runr;
				NTO mrhs = SR::multiply(wset[hsize - 1].num, Bcsc->num[locb]);
				if (!SR::returnedSAID())
				{
					if ((curptr > colptrC[i]) &&
						std::get<0>(tuplesC[curptr - 1]) == wset[hsize - 1].key)
						std::get<2>(tuplesC[curptr - 1]) =
							SR::add(std::get<2>(tuplesC[curptr - 1]), mrhs);
					else
						tuplesC[curptr++] = std::make_tuple(wset[hsize - 1].key,
															i, mrhs) ;
				}

				IT	locb_offset = locb - Bcsc->jc[i];
				IT	ca			= Bcsc->ir[locb];
				++(cnt[locb_offset]);
				if (cnt[locb_offset] != Acsc->jc[ca + 1])
				{
					wset[hsize - 1].key = Acsc->ir[cnt[locb_offset]];
					wset[hsize - 1].num = Acsc->num[cnt[locb_offset]];
					std::push_heap(wset, wset + hsize);
				}
				else
					--hsize;
			}
		} // Finish heap
		else					// Hash Algorithm
		{
			// Set up hash table
			const IT	minHashTableSize = 16;
            const IT	hashScale		 = 107;
            size_t		nnzcolC			 = colptrC[i+1] - colptrC[i];
			size_t		ht_size			 = minHashTableSize;

			while (ht_size < nnzcolC)
                ht_size <<= 1;
			std::vector< std::pair<IT, NTO>> T(ht_size);

			for (size_t j = 0; j < ht_size; ++j)
                T[j].first = std::numeric_limits<IT>::max();

			// multiplication
			for (IT j = Bcsc->jc[i]; j < Bcsc->jc[i + 1]; ++j)
			{
				IT	t_bcol = Bcsc->ir[j];
				NT2 t_bval = Bcsc->num[j];
				for (IT k = Acsc->jc[t_bcol]; k < Acsc->jc[t_bcol + 1]; ++k)
				{
					NTO mrhs = SR::multiply(Acsc->num[k], t_bval);
					IT	key	 = Acsc->ir[k];
					IT	hv	 = (key * hashScale) & (ht_size - 1);

				repeat:
					if (T[hv].first == key)
						T[hv].second = SR::add(mrhs, T[hv].second);
					else if (T[hv].first == std::numeric_limits<IT>::max())
					{
						T[hv].first	 = key;
						T[hv].second = mrhs;
					}
					else
					{
						hv = (hv + 1) & (ht_size - 1);
						goto repeat;
					}
				}
			}

			size_t index = 0;
			for (size_t j = 0; j < ht_size; ++j)
			{
				if (T[j].first != std::numeric_limits<IT>::max())
					T[index++] = T[j];
			}

			std::sort(T.begin(), T.begin() + index, sort_less<IT, NTO>);

			IT curptr = colptrC[i];
			for (size_t j = 0; j < index; ++j)
                tuplesC[curptr++] = std::make_tuple(T[j].first, i, T[j].second);
		}
	}

	if (clearA)
		delete const_cast<SpCCols<IT, NT1> *>(&A);
	if (clearB)
		delete const_cast<SpCCols<IT, NT2> *>(&B);

	delete [] colptrC;
    delete [] flopptr;

	SpTuples<IT, NTO> *spTuplesC =
		new SpTuples<IT, NTO> (nnzc, mdim, ndim, tuplesC, true, true);
	
	double t1 = MPI_Wtime();

	return spTuplesC;
}



template <typename IT,
		  typename NT1,
		  typename NT2>
IT *
estimateFLOP (const SpCCols<IT, NT1> &A,
			  const SpCCols<IT, NT2> &B
			  )
{
	IT nnzA = A.getnnz();
	if (A.isZero() || B.isZero())
        return NULL;

	Csc<IT, NT1> *Acsc = A.GetCSC();
    Csc<IT, NT2> *Bcsc = B.GetCSC();

	int numThreads = 1;
	#ifdef THREADED
	#pragma omp parallel
    {
        numThreads = omp_get_num_threads();
    }
	#endif

	IT *colflopC = new IT[Bcsc->n];

	#ifdef THREADED
	#pragma omp parallel for
	#endif
    for (IT i = 0; i < Bcsc->n; ++i)
        colflopC[i] = 0;

	#ifdef THREADED
	#pragma omp parallel for
	#endif
	for (IT i = 0; i < Bcsc->n; ++i)
	{
		for (IT j = Bcsc->jc[i]; j < Bcsc->jc[i+1]; ++j)
			colflopC[i] += Acsc->jc[Bcsc->ir[j]+1] - Acsc->jc[Bcsc->ir[j]];
	}

	return colflopC;
}



template <typename IT,
		  typename NT1,
		  typename NT2>
IT *
estimateNNZ_Hash (const SpCCols<IT, NT1>	&A,
				  const SpCCols<IT, NT2>	&B,
				  const IT					*flopC
				  )
{
	IT nnzA = A.getnnz();
    if (A.isZero() || B.isZero())
        return NULL;
    
    Csc<IT, NT1> *Acsc = A.GetCSC();
    Csc<IT, NT2> *Bcsc = B.GetCSC();

	int numThreads = 1;
	#ifdef THREADED
	#pragma omp parallel
    {
        numThreads = omp_get_num_threads();
    }
	#endif

	IT *colnnzC = new IT[Bcsc->n];

	#ifdef THREADED
	#pragma omp parallel for
	#endif
    for (IT i = 0; i < Bcsc->n; ++i)
        colnnzC[i] = 0;
		
	#ifdef THREADED
	#pragma omp parallel for
	#endif
	for (IT i = 0; i < Bcsc->n; ++i)
	{	
		// init hash table
		const IT	minHashTableSize = 16;
        const IT	hashScale		 = 107;
        IT			ht_size			 = minHashTableSize;
		while (ht_size < flopC[i]) // make size of hash table a power of 2
            ht_size <<= 1;

		// IT can be unsigned
		std::vector<IT> T(ht_size);
		for (IT j = 0; (unsigned)j < ht_size; ++j)
            T[j] = std::numeric_limits<IT>::max();

		for (IT j = Bcsc->jc[i]; j < Bcsc->jc[i + 1]; ++j)
		{
			for (IT k = Acsc->jc[Bcsc->ir[j]]; k < Acsc->jc[Bcsc->ir[j]+1]; ++k)
			{
				IT	key	= Acsc->ir[k];
				IT	hv	= (key * hashScale) & (ht_size - 1);

				while (1)
				{
					if (T[hv] == key)
						break;
					else if (T[hv] == std::numeric_limits<IT>::max())
					{
						T[hv] = key;
						++(colnnzC[i]);
						break;
					}
					else
						hv = (hv + 1) & (ht_size - 1);
				}
			}
		}
	}

	return colnnzC;
}
			  
	
}

#endif
