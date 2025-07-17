#ifndef _MULTIWAY_MERGE_H_
#define _MULTIWAY_MERGE_H_

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
T* prefixSum(T* in, int size, int nthreads)
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


/***************************************************************************
 * Find indices of column splitters in a list of tuple in parallel.
 * Inputs:
 *      tuples: an array of SpTuples each tuple is (rowid, colid, val)
 *      nsplits: number of splits requested
 *  Output:
 *      splitters: An array of size (nsplits+1) storing the starts and ends of split tuples.
 *      different type used for output since we might need int or IT
 ***************************************************************************/

template <typename RT, typename IT, typename NT>
std::vector<RT> findColSplitters(SpTuples<IT,NT> * & spTuples, int nsplits)
{
    std::vector<RT> splitters(nsplits+1);
    splitters[0] = static_cast<RT>(0);
    ColLexiCompare<IT,NT> comp;
#ifdef THREADED
#pragma omp parallel for
#endif
    for(int i=1; i< nsplits; i++)
    {
        IT cur_col = i * (spTuples->getncol()/nsplits);
        std::tuple<IT,IT,NT> search_tuple(0, cur_col, NT());
        std::tuple<IT,IT,NT>* it = std::lower_bound (spTuples->tuples, spTuples->tuples + spTuples->getnnz(), search_tuple, comp);
        splitters[i] = (RT) (it - spTuples->tuples);
    }
    splitters[nsplits] = spTuples->getnnz();
    
    return splitters;
}

    // Find ColSplitters using finger search
    // Run by one threrad
    template <typename RT, typename IT, typename NT>
    std::vector<RT> findColSplittersFinger(SpTuples<IT,NT> * & spTuples, int nsplits)
    {
        std::vector<RT> splitters(nsplits+1);
        splitters[0] = static_cast<RT>(0);
        ColLexiCompare<IT,NT> comp;

        std::tuple<IT,IT,NT>* start = spTuples->tuples;
        std::tuple<IT,IT,NT>* end = spTuples->tuples + spTuples->getnnz();
        for(int i=1; i< nsplits; i++)
        {
            IT cur_col = i * (spTuples->getncol()/nsplits);
            std::tuple<IT,IT,NT> search_tuple(0, cur_col, NT());
            std::tuple<IT,IT,NT>* it = std::lower_bound (start, end, search_tuple, comp);
            splitters[i] = (RT) (it - spTuples->tuples);
            //start = it;
        }
        splitters[nsplits] = spTuples->getnnz();
        
        return splitters;
    }

// Symbolic serial merge : only estimates nnz
template<class IT, class NT>
IT SerialMergeNNZ( const std::vector<SpTuples<IT,NT> *> & ArrSpTups)
{
    int nlists =  ArrSpTups.size();
    ColLexiCompare<IT,int> heapcomp;
    std::vector<std::tuple<IT, IT, int>> heap(nlists);
    std::vector<IT> curptr(nlists, static_cast<IT>(0));
    IT hsize = 0;
    for(int i=0; i< nlists; ++i)
    {
        if(ArrSpTups[i]->getnnz()>0)
        {
            heap[hsize++] = std::make_tuple(std::get<0>(ArrSpTups[i]->tuples[0]), std::get<1>(ArrSpTups[i]->tuples[0]), i);
        }
        
    }
    std::make_heap(heap.data(), heap.data()+hsize,
        [&heapcomp](const auto& a, const auto& b) {
        return !heapcomp(a, b);
    });
    
    std::tuple<IT, IT, NT> curTuple;
    IT estnnz = 0;
    while(hsize > 0)
    {
        std::pop_heap(heap.data(), heap.data() + hsize,
            [&heapcomp](const auto& a, const auto& b) {
          return !heapcomp(a, b);
      });   // result is stored in heap[hsize-1]
        int source = std::get<2>(heap[hsize-1]);
        if( (estnnz ==0) || (std::get<0>(curTuple) != std::get<0>(heap[hsize-1])) || (std::get<1>(curTuple) != std::get<1>(heap[hsize-1])))
        {
            curTuple = ArrSpTups[source]->tuples[curptr[source]];
            estnnz++;
        }
        curptr[source]++;
        if(curptr[source] != ArrSpTups[source]->getnnz())	// That array has not been depleted
        {
            heap[hsize-1] = std::make_tuple(std::get<0>(ArrSpTups[source]->tuples[curptr[source]]),
                                       std::get<1>(ArrSpTups[source]->tuples[curptr[source]]), source);
            std::push_heap(heap.data(), heap.data()+hsize,
                [&heapcomp](const auto& a, const auto& b) {
                return !heapcomp(a, b);
    });
        }
        else
        {
            --hsize;
        }
    }
    return estnnz;
}


/*
 "Internal function" called by MultiwayMerge inside threaded region.
 The merged list is stored in a preallocated buffer ntuples
 Never called from outside.
 Assumption1: the input lists are already column sorted
 Assumption2: at least two lists are passed to this function
 Assumption3: the input and output lists are to be deleted by caller
 */

template<class SR, class IT, class NT>
void SerialMerge( const std::vector<SpTuples<IT,NT> *> & ArrSpTups, std::tuple<IT, IT, NT> * ntuples)
{
    int nlists =  ArrSpTups.size();
    ColLexiCompare<IT,int> heapcomp;
    std::vector<std::tuple<IT, IT, int>> heap(nlists); // if performance issue, create this outside of threaded region
    std::vector<IT> curptr(nlists, static_cast<IT>(0));
    IT estnnz = 0;
    IT hsize = 0;
    for(int i=0; i< nlists; ++i)
    {
        if(ArrSpTups[i]->getnnz()>0)
        {
            estnnz += ArrSpTups[i]->getnnz();
            heap[hsize++] = std::make_tuple(std::get<0>(ArrSpTups[i]->tuples[0]), std::get<1>(ArrSpTups[i]->tuples[0]), i);
        }
        
    }
    std::make_heap(heap.data(), heap.data()+hsize,
        [&heapcomp](const auto& a, const auto& b) {
        return !heapcomp(a, b);
    });
    IT cnz = 0;
    
    while(hsize > 0)
    {
        std::pop_heap(heap.data(), heap.data() + hsize,
            [&heapcomp](const auto& a, const auto& b) {
          return !heapcomp(a, b);
      });   // result is stored in heap[hsize-1]
        int source = std::get<2>(heap[hsize-1]);
        
        if( (cnz != 0) &&
           ((std::get<0>(ntuples[cnz-1]) == std::get<0>(heap[hsize-1])) && (std::get<1>(ntuples[cnz-1]) == std::get<1>(heap[hsize-1]))) )
        {
            std::get<2>(ntuples[cnz-1])  = SR::add(std::get<2>(ntuples[cnz-1]), ArrSpTups[source]->numvalue(curptr[source]++));
        }
        else
        {
            ntuples[cnz++] = ArrSpTups[source]->tuples[curptr[source]++];
        }
        
        if(curptr[source] != ArrSpTups[source]->getnnz())	// That array has not been depleted
        {
            heap[hsize-1] = std::make_tuple(std::get<0>(ArrSpTups[source]->tuples[curptr[source]]),
                                       std::get<1>(ArrSpTups[source]->tuples[curptr[source]]), source);
            std::push_heap(heap.data(), heap.data()+hsize, [&heapcomp](const auto& a, const auto& b) {
        return !heapcomp(a, b);
    });
        }
        else
        {
            --hsize;
        }
    }
}




// Symbolic serial merge : only estimates nnz
    template<class IT, class NT>
    IT* SerialMergeNNZHash( const std::vector<SpTuples<IT,NT> *> & ArrSpTups, IT& totnnz, IT& maxnnzPerCol, IT startCol, IT endCol)
    {
        
        int nlists =  ArrSpTups.size();
        IT ncols = endCol - startCol; // in this split
        std::vector<IT> curptr(nlists, static_cast<IT>(0));
        const IT minHashTableSize = 16;
        const IT hashScale = 107;
        std::vector<NT> globalHashVec(minHashTableSize);
        
        
        
        IT* colnnzC = new IT[ncols](); // nnz in every column of C
        maxnnzPerCol = 0;
        totnnz = 0;
        
        for(IT col = 0; col<ncols; col++)
        {
            IT globalCol = col + startCol;
            
            // symbolic flop
            size_t nnzcol = 0;
            for(int i=0; i<nlists; i++)
            {
                IT curidx = curptr[i];
                while((ArrSpTups[i]->getnnz()>curidx) && (ArrSpTups[i]->colindex(curidx++) == globalCol))
                {
                    nnzcol++;
                }
            }
            size_t ht_size = minHashTableSize;
            while(ht_size < nnzcol) //ht_size is set as 2^n
            {
                ht_size <<= 1;
            }
            if(globalHashVec.size() < ht_size)
                globalHashVec.resize(ht_size);
            
            for(size_t j=0; j < ht_size; ++j)
            {
                globalHashVec[j] = -1;
            }
            for(int i=0; i<nlists; i++)
            {
                //IT curcol = std::get<1>(ArrSpTups[i]->tuples[curptr[i]]);
                while((ArrSpTups[i]->getnnz()>curptr[i]) && (ArrSpTups[i]->colindex(curptr[i]) == globalCol))
                {
                    IT key = ArrSpTups[i]->rowindex(curptr[i]);
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
                            colnnzC[col] ++;
                            break;
                        }
                        else //key is not found
                        {
                            hash = (hash+1) & (ht_size-1);
                        }
                    }
                    curptr[i]++;
                }
            }
            totnnz += colnnzC[col];
            if(colnnzC[col] > maxnnzPerCol) maxnnzPerCol = colnnzC[col];
        }
        return colnnzC;
    }


    
    
    // Serially merge a split along the column
    // startCol and endCol denote the start and end of the current split
    // maxcolnnz: maximum nnz in a merged column (from symbolic)
    template<class SR, class IT, class NT>
    void SerialMergeHash( const std::vector<SpTuples<IT,NT> *> & ArrSpTups, std::tuple<IT, IT, NT> * ntuples, IT* colnnz, IT maxcolnnz, IT startCol, IT endCol, bool sorted)
    {
        int nlists =  ArrSpTups.size();
        IT ncols = endCol - startCol; // in this split
        IT outptr = 0;
        std::vector<IT> curptr(nlists, static_cast<IT>(0));
        
        const IT minHashTableSize = 16;
        const IT hashScale = 107;
        //std::vector< std::pair<IT,NT>> globalHashVec(std::max(minHashTableSize, maxcolnnz*2));
        std::vector< std::pair<uint32_t,NT>> globalHashVec(std::max(minHashTableSize, maxcolnnz*2));
        
        for(IT col = 0; col<ncols; col++)
        {
            IT globalCol = col + startCol;
            size_t ht_size = minHashTableSize;
            while(ht_size < colnnz[col]) //ht_size is set as 2^n
            {
                ht_size <<= 1;
            }
            for(size_t j=0; j < ht_size; ++j)
            {
                globalHashVec[j].first = -1;
            }
            for(int i=0; i<nlists; i++)
            {
                while((ArrSpTups[i]->getnnz()>curptr[i]) && (ArrSpTups[i]->colindex(curptr[i]) == globalCol))
                {
                    IT key = ArrSpTups[i]->rowindex(curptr[i]);
                    IT hash = (key*hashScale) & (ht_size-1);
                    
                    while (1) //hash probing
                    {
                        NT curval = ArrSpTups[i]->numvalue(curptr[i]);
                        if (globalHashVec[hash].first == key) //key is found in hash table
                        {
                            globalHashVec[hash].second = SR::add(curval, globalHashVec[hash].second);
                            break;
                        }
                        else if (globalHashVec[hash].first == -1) //key is not registered yet
                        {
                            globalHashVec[hash].first = key;
                            globalHashVec[hash].second = curval;
                            break;
                        }
                        else //key is not found
                        {
                            hash = (hash+1) & (ht_size-1);
                        }
                    }
                    curptr[i]++;
                }
            }
            
            if(sorted)
            {
                size_t index = 0;
                for (size_t j=0; j < ht_size; ++j)
                {
                    if (globalHashVec[j].first != -1)
                    {
                        globalHashVec[index++] = globalHashVec[j];
                    }
                }
                integerSort<NT>(globalHashVec.data(), index);
                //std::sort(globalHashVec.begin(), globalHashVec.begin() + index, sort_less<IT, NT>);
                
                
                for (size_t j=0; j < index; ++j)
                {
                    ntuples[outptr++]= std::make_tuple(globalHashVec[j].first, globalCol, globalHashVec[j].second);
                }
            }
            else
            {
                for (size_t j=0; j < ht_size; ++j)
                {
                    if (globalHashVec[j].first != -1)
                    {
                        ntuples[outptr++]= std::make_tuple(globalHashVec[j].first, globalCol, globalHashVec[j].second);
                    }
                }
            }
        }
    }



// Performs a balanced merge of the array of SpTuples
// Assumes the input parameters are already column sorted
template<class SR, class IT, class NT>
SpTuples<IT, NT>* MultiwayMerge( std::vector<SpTuples<IT,NT> *> & ArrSpTups, IT mdim = 0, IT ndim = 0, bool delarrs = false )
{
    
    int nlists =  ArrSpTups.size();
    if(nlists == 0)
    {
        return new SpTuples<IT,NT>(0, mdim, ndim); //empty mxn SpTuples
    }
    if(nlists == 1)
    {
        if(delarrs) // steal data from input, and don't delete input
        {
            return ArrSpTups[0];
        }
        else // copy input to output
        {
        //    std::tuple<IT, IT, NT>* mergeTups = static_cast<std::tuple<IT, IT, NT>*>
         //   (::operator new (sizeof(std::tuple<IT, IT, NT>[ArrSpTups[0]->getnnz()])));

	    std::tuple<IT, IT, NT>* mergeTups = new std::tuple<IT, IT, NT>[ArrSpTups[0]->getnnz()];
#ifdef THREADED
#pragma omp parallel for
#endif
            for(int i=0; i<ArrSpTups[0]->getnnz(); i++)
                mergeTups[i] = ArrSpTups[0]->tuples[i];
            
            return new SpTuples<IT,NT> (ArrSpTups[0]->getnnz(), mdim, ndim, mergeTups, false);
        }
    }
    
    // ---- check correctness of input dimensions ------
    for(int i=0; i< nlists; ++i)
    {
        if((mdim != ArrSpTups[i]->getnrow()) || ndim != ArrSpTups[i]->getncol())
        {
            std::cerr << "Dimensions of SpTuples do not match on multiwayMerge()" << std::endl;
            return new SpTuples<IT,NT>(0,0,0);
        }
    }
    
    int nthreads = 1;	
#ifdef THREADED
#pragma omp parallel
    {
        nthreads = omp_get_num_threads();
    }
#endif
    int nsplits = 4*nthreads; // oversplit for load balance
    nsplits = std::min(nsplits, (int)ndim); // we cannot split a column
    std::vector< std::vector<IT> > colPtrs;
    for(int i=0; i< nlists; i++)
    {
        colPtrs.push_back(findColSplitters<IT>(ArrSpTups[i], nsplits)); // in parallel
    }

    std::vector<IT> mergedNnzPerSplit(nsplits);
    std::vector<IT> inputNnzPerSplit(nsplits);
// ------ estimate memory requirement after merge in each split ------
#ifdef THREADED
#pragma omp parallel for schedule(dynamic)
#endif
    for(int i=0; i< nsplits; i++) // for each part
    {
        std::vector<SpTuples<IT,NT> *> listSplitTups(nlists);
        IT t = static_cast<IT>(0);
        for(int j=0; j< nlists; ++j)
        {
            IT curnnz= colPtrs[j][i+1] - colPtrs[j][i];
            listSplitTups[j] = new SpTuples<IT, NT> (curnnz, mdim, ndim, ArrSpTups[j]->tuples + colPtrs[j][i], true);
            t += colPtrs[j][i+1] - colPtrs[j][i];
        }
        mergedNnzPerSplit[i] = SerialMergeNNZ(listSplitTups);
        inputNnzPerSplit[i] = t;
    }

    std::vector<IT> mdisp(nsplits+1,0);
    for(int i=0; i<nsplits; ++i)
        mdisp[i+1] = mdisp[i] + mergedNnzPerSplit[i];
    IT mergedNnzAll = mdisp[nsplits];
    
    
#ifdef COMBBLAS_DEBUG
    IT inputNnzAll = std::accumulate(inputNnzPerSplit.begin(), inputNnzPerSplit.end(), static_cast<IT>(0));
    double ratio = inputNnzAll / (double) mergedNnzAll;
    std::ostringstream outs;
    outs << "Multiwaymerge: inputNnz/mergedNnz = " << ratio << std::endl;
    SpParHelper::Print(outs.str());
#endif
    
    
    // ------ allocate memory outside of the parallel region ------
   //std::tuple<IT, IT, NT> * mergeBuf = static_cast<std::tuple<IT, IT, NT>*> (::operator new (sizeof(std::tuple<IT, IT, NT>[mergedNnzAll])));
   std::tuple<IT, IT, NT> * mergeBuf = new std::tuple<IT, IT, NT>[mergedNnzAll]; 
    // ------ perform merge in parallel ------
#ifdef THREADED
#pragma omp parallel for schedule(dynamic)
#endif
    for(int i=0; i< nsplits; i++) // serially merge part by part
    {
        std::vector<SpTuples<IT,NT> *> listSplitTups(nlists);
        for(int j=0; j< nlists; ++j)
        {
            IT curnnz= colPtrs[j][i+1] - colPtrs[j][i];
            listSplitTups[j] = new SpTuples<IT, NT> (curnnz, mdim, ndim, ArrSpTups[j]->tuples + colPtrs[j][i], true);
        }
        SerialMerge<SR>(listSplitTups, mergeBuf + mdisp[i]);
    }
    
    for(int i=0; i< nlists; i++)
    {
        if(delarrs)
            delete ArrSpTups[i]; // May be expensive for large local matrices
    }
    return new SpTuples<IT, NT> (mergedNnzAll, mdim, ndim, mergeBuf, true, false);
}


   
    // --------------------------------------------------------
    // Hash-based multiway merge
    // Columns of the input matrices may or may not be sorted
    //  the hash merging algorithm does not need sorted inputs
    // If sorted=true, columns of the output matrix are sorted
    // --------------------------------------------------------
    template<class SR, class IT, class NT>
    SpTuples<IT, NT>* MultiwayMergeHash( std::vector<SpTuples<IT,NT> *> & ArrSpTups, IT mdim = 0, IT ndim = 0, bool delarrs = false, bool sorted=true )
    {
        int nprocs, myrank;
        MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
        MPI_Comm_rank(MPI_COMM_WORLD,&myrank);

        int nlists =  ArrSpTups.size();
        if(nlists == 0)
        {
            return new SpTuples<IT,NT>(0, mdim, ndim); //empty mxn SpTuples
        }
        if(nlists == 1)
        {
            if(delarrs) // steal data from input, and don't delete input
            {
                return ArrSpTups[0];
            }
            else // copy input to output
            {
                std::tuple<IT, IT, NT>* mergeTups = static_cast<std::tuple<IT, IT, NT>*>
                (::operator new (sizeof(std::tuple<IT, IT, NT>[ArrSpTups[0]->getnnz()])));
#ifdef THREADED
#pragma omp parallel for
#endif
                for(int i=0; i<ArrSpTups[0]->getnnz(); i++)
                    mergeTups[i] = ArrSpTups[0]->tuples[i];
                
                // Caution: ArrSpTups[0] can be either sorted or unsorted
                // By setting sorted=true, we prevented sorting in the SpTuples constructor
                // TODO: we better keep a isSorted flag in SpTuples (also in DCSC/CSC)
                return new SpTuples<IT,NT> (ArrSpTups[0]->getnnz(), mdim, ndim, mergeTups, true, true);
            }
        }
        
        // ---- check correctness of input dimensions ------
        for(int i=0; i< nlists; ++i)
        {
            if((mdim != ArrSpTups[i]->getnrow()) || ndim != ArrSpTups[i]->getncol())
            {
                std::cerr << "Dimensions of SpTuples do not match on multiwayMerge()" << std::endl;
                return new SpTuples<IT,NT>(0,0,0);
            }
        }
        
        int nthreads = 1;
#ifdef THREADED
#pragma omp parallel
        {
            nthreads = omp_get_num_threads();
        }
#endif
        int nsplits = 4*nthreads; // oversplit for load balance
        nsplits = std::min(nsplits, (int)ndim); // we cannot split a column
        std::vector< std::vector<IT> > colPtrs(nlists);
        

#ifdef THREADED
#pragma omp parallel for
#endif
        for(int j=0; j< nlists; j++)
        {
            colPtrs[j]=findColSplittersFinger<IT>(ArrSpTups[j], nsplits);
        }
        
        
        // listSplitTups is just a temporary vector to facilitate serial merging
        // It does not allocate or move any input tuples
        // Hence, sorted and opnew options do not matter when creating SpTuples
        // Ideally we can directly work with std::tuples
        std::vector<std::vector<SpTuples<IT,NT> *>> listSplitTups(nsplits);

        for(int i=0; i< nsplits; ++i) // for each part
        {
            listSplitTups[i].resize(nlists);

            for(int j=0; j< nlists; ++j)
            {
                IT curnnz= colPtrs[j][i+1] - colPtrs[j][i];
                listSplitTups[i][j] = new SpTuples<IT, NT> (curnnz, mdim, ndim, ArrSpTups[j]->tuples + colPtrs[j][i], true);
            }

        }
       
        std::vector<IT> mergedNnzPerSplit(nsplits);
        std::vector<IT> mergedNnzPerSplit1(nsplits);
        std::vector<IT> maxNnzPerColumnSplit(nsplits);
        std::vector<IT*> nnzPerColSplit(nsplits);
                
        // ------ estimate memory requirement after merge in each split ------
#ifdef THREADED
#pragma omp parallel for schedule(dynamic)
#endif
        for(int i=0; i< nsplits; i++) // for each part
        {
            IT startCol = i* (ndim/nsplits);
            IT endCol = (i+1)* (ndim/nsplits);
            if(i == (nsplits-1)) endCol = ndim;
            
            nnzPerColSplit[i] = SerialMergeNNZHash(listSplitTups[i], mergedNnzPerSplit[i], maxNnzPerColumnSplit[i], startCol, endCol);
        }
       
        std::vector<IT> mdisp(nsplits+1,0);
        for(int i=0; i<nsplits; ++i)
            mdisp[i+1] = mdisp[i] + mergedNnzPerSplit[i];
        IT mergedNnzAll = mdisp[nsplits];

        // ------ allocate memory outside of the parallel region ------
        std::tuple<IT, IT, NT> * mergeBuf = static_cast<std::tuple<IT, IT, NT>*> (::operator new (sizeof(std::tuple<IT, IT, NT>[mergedNnzAll])));
        //std::tuple<IT, IT, NT> * mergeBuf = new std::tuple<IT, IT, NT>[mergedNnzAll]; 
  

        
        // ------ perform merge in parallel ------
#ifdef THREADED
#pragma omp parallel for schedule(dynamic)
#endif
        for(int i=0; i< nsplits; i++) // serially merge part by part
        {
            //SerialMerge<SR>(listSplitTups, mergeBuf + mdisp[i]);
            IT startCol = i* (ndim/nsplits);
            IT endCol = (i+1)* (ndim/nsplits);
            if(i == (nsplits-1)) endCol = ndim;
            SerialMergeHash<SR>(listSplitTups[i], mergeBuf + mdisp[i], nnzPerColSplit[i], maxNnzPerColumnSplit[i], startCol, endCol, sorted);
            // last parameter is for sorted
        }
        
        // Delete and free a lot of dynamic allocations
        for(int i=0; i< nsplits; ++i) // for each part
        {
            delete nnzPerColSplit[i];
            for(int j=0; j< nlists; ++j)
            {
                listSplitTups[i][j]->tuples_deleted = true;
                delete listSplitTups[i][j];
            }

        }
        for(int i=0; i< nlists; i++)
        {
            if(delarrs)
                delete ArrSpTups[i]; // May be expensive for large local matrices
        }
        
        // Caution: We allow both sorted and unsorted tuples in SpTuples
        // By setting sorted=true, we prevented sorting in the SpTuples constructor
        // TODO: we better keep a isSorted flag in SpTuples (also in DCSC/CSC)
        return new SpTuples<IT, NT> (mergedNnzAll, mdim, ndim, mergeBuf, true, true);
    }

    // --------------------------------------------------------
    // Hash-based multiway merge
    // Columns of the input matrices may or may not be sorted
    //  the hash merging algorithm does not need sorted inputs
    // If sorted=true, columns of the output matrix are sorted
    // --------------------------------------------------------
    template<class SR, class IT, class NT>
    SpTuples<IT, NT>* MultiwayMergeHashSliding( std::vector<SpTuples<IT,NT> *> & ArrSpTups, IT mdim = 0, IT ndim = 0, bool delarrs = false, bool sorted=true,  IT maxHashTableSize = 16384)
    {
        int nprocs, myrank;
        MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
        MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
        
        int nlists =  ArrSpTups.size();
        if(nlists == 0)
        {
            return new SpTuples<IT,NT>(0, mdim, ndim); //empty mxn SpTuples
        }
        if(nlists == 1)
        {
            if(delarrs) // steal data from input, and don't delete input
            {
                return ArrSpTups[0];
            }
            else // copy input to output
            {
                std::tuple<IT, IT, NT>* mergeTups = static_cast<std::tuple<IT, IT, NT>*>
                (::operator new (sizeof(std::tuple<IT, IT, NT>[ArrSpTups[0]->getnnz()])));
#ifdef THREADED
#pragma omp parallel for
#endif
                for(int i=0; i<ArrSpTups[0]->getnnz(); i++)
                    mergeTups[i] = ArrSpTups[0]->tuples[i];
                
                // Caution: ArrSpTups[0] can be either sorted or unsorted
                // By setting sorted=true, we prevented sorting in the SpTuples constructor
                // TODO: we better keep a isSorted flag in SpTuples (also in DCSC/CSC)
                return new SpTuples<IT,NT> (ArrSpTups[0]->getnnz(), mdim, ndim, mergeTups, true, true);
            }
        }

        // ---- check correctness of input dimensions ------
        for(int i=0; i< nlists; ++i)
        {   
            if((mdim != ArrSpTups[i]->getnrow()) || ndim != ArrSpTups[i]->getncol())
            {
                std::cerr << "Dimensions of SpTuples do not match on MultiwayMergeHashSliding()" << std::endl;
                //std::cerr << mdim << " vs " << ArrSpTups[i]->getnrow() << " | " << ndim << " vs " << ArrSpTups[i]->getncol() << std::endl;
                return new SpTuples<IT,NT>(0,0,0);
            }
        }
        
        int nthreads = 1;
#ifdef THREADED
#pragma omp parallel
        {
            nthreads = omp_get_num_threads();
        }
#endif
        int nsplits = 4*nthreads; // oversplit for load balance
        nsplits = std::min(nsplits, (int)ndim); // we cannot split a column

        const IT minHashTableSize = 16;
        //const IT maxHashTableSize = 8 * 1024; // Moved to parameter
        const IT hashScale = 107;
        
        /*
         * To store column pointers of CSC like data structures
         * */
        IT** colPtrs = static_cast<IT**> (::operator new (sizeof(IT*[nlists]))); 
        for(int l = 0; l < nlists; l++){
            colPtrs[l] = static_cast<IT*> (::operator new (sizeof(IT[ndim+1]))); 
        }
        ColLexiCompare<IT,NT> colCmp;
        RowLexiCompare<IT,NT> rowCmp;

#ifdef THREADED
#pragma omp parallel for
#endif
        for(int s = 0; s < nsplits; s++){
            IT startColSplit = s * (ndim/nsplits);
            IT endColSplit = (s == (nsplits-1) ) ? ndim : (s+1) * (ndim/nsplits);
            for(int l = 0; l < nlists; l++){
                std::tuple<IT, IT, NT> firstTuple(0, startColSplit, NT());
                std::tuple<IT, IT, NT>* first = std::lower_bound(ArrSpTups[l]->tuples, ArrSpTups[l]->tuples+ArrSpTups[l]->getnnz(), firstTuple, colCmp);
                std::tuple<IT, IT, NT> lastTuple(0, endColSplit, NT());
                std::tuple<IT, IT, NT>* last = std::lower_bound(ArrSpTups[l]->tuples, ArrSpTups[l]->tuples+ArrSpTups[l]->getnnz(), lastTuple, colCmp);
                for(IT c = startColSplit; c < endColSplit; c++){
                    if(c == 0) colPtrs[l][c] = 0;
                    else{
                        std::tuple<IT, IT, NT> searchTuple(0, c, NT());
                        std::tuple<IT, IT, NT>* pos = std::lower_bound(first, last, searchTuple, colCmp);
                        colPtrs[l][c] = pos - ArrSpTups[l]->tuples;
                    }
                }
                if(s == nsplits-1) colPtrs[l][ndim] = ArrSpTups[l]->getnnz();
            }
        }

        size_t* flopsPerCol = static_cast<size_t*> (::operator new (sizeof(size_t[ndim]))); 
        IT* nWindowPerColSymbolic = static_cast<IT*> (::operator new (sizeof(IT[ndim])));
#ifdef THREADED
#pragma omp parallel for
#endif
        for(IT c = 0; c < ndim; c++){
            flopsPerCol[c] = 0;
            for(int l = 0; l < nlists; l++){
                flopsPerCol[c] += colPtrs[l][c+1] - colPtrs[l][c];
            }
            nWindowPerColSymbolic[c] = flopsPerCol[c] / maxHashTableSize + 1;
        }
        
        size_t* prefixSumFlopsPerCol = prefixSum<size_t>(flopsPerCol, ndim, nthreads);
        size_t totalFlops = prefixSumFlopsPerCol[ndim];
        size_t flopsPerSplit = totalFlops / nsplits;
        IT* colSplitters = static_cast<size_t*> (::operator new (sizeof(size_t[nsplits+1]))); 
        
        /*
         * For symbolic, split column between threads in such a way so that total flops is
         * balanced between threads
         * */
#ifdef THREADED
#pragma omp parallel for
#endif
        for(int s = 0; s < nsplits; s++){
            size_t searchItem = s * flopsPerSplit;
            size_t* searchResult = std::lower_bound(prefixSumFlopsPerCol, prefixSumFlopsPerCol + ndim + 1, searchItem);
            colSplitters[s] = searchResult - prefixSumFlopsPerCol;
        }
        colSplitters[nsplits] = ndim;
        
        /*
         * Calculate prefix sum of number of windows needed per column.
         * This information will be used to determine the index in the windowsSymbolic array
         * */
        IT* prefixSumWindowSymbolic = prefixSum<IT>(nWindowPerColSymbolic, ndim, nthreads);

        std::pair<IT, IT>* windowsSymbolic = static_cast<std::pair<IT, IT>*> (::operator new (sizeof(std::pair<IT, IT>[prefixSumWindowSymbolic[ndim]])));
        IT* nnzPerCol = static_cast<IT*> (::operator new (sizeof(IT[ndim])));
        IT* nWindowPerCol = static_cast<IT*> (::operator new (sizeof(IT[ndim])));

        /*
         * To keep track of rows being processed in each matrix over sliding window
         * */
        std::pair<IT, IT>** rowIdsRange = static_cast<std::pair<IT, IT>**> (::operator new (sizeof(std::pair<IT, IT>*[nsplits])));
        for(int s = 0; s < nsplits; s++){
            rowIdsRange[s] = static_cast<std::pair<IT, IT>*> (::operator new (sizeof(std::pair<IT, IT>[nlists])));
        }

#ifdef THREADED
#pragma omp parallel
#endif
        {
            std::vector<NT> globalHashVec(minHashTableSize);
            size_t tid = omp_get_thread_num();
#ifdef THREADED
#pragma omp for schedule(dynamic)
#endif
            for(int s = 0; s < nsplits; s++){
                IT startCol = colSplitters[s];
                IT endCol = colSplitters[s+1];

                for(IT c = startCol; c < endCol; c++){
                    nnzPerCol[c] = 0;
                    nWindowPerCol[c] = 1;
                    if(nWindowPerColSymbolic[c] == 1){
                        IT startRow = 0;
                        IT endRow = mdim;
                        size_t wsIdx = prefixSumWindowSymbolic[c];

                        windowsSymbolic[wsIdx].first = 0; // Stores start row id of the window
                        windowsSymbolic[wsIdx].second = 0; // Stores number of merged nonzero in the window

                        size_t htSize = minHashTableSize;
                        while(htSize < flopsPerCol[c]) {
                            //ht_size is set as 2^n
                            htSize <<= 1;
                        }
                        if(globalHashVec.size() < htSize) globalHashVec.resize(htSize);
                        for(size_t j=0; j < htSize; ++j) {
                            globalHashVec[j] = -1;
                        }

                        for(int l = 0; l < nlists; l++){
                            for(IT i = colPtrs[l][c]; i < colPtrs[l][c+1]; i++){
                                IT key = ArrSpTups[l]->rowindex(i);
                                IT hash = (key*hashScale) & (htSize-1);
                                
                                while (1) {
                                    //hash probing
                                    if (globalHashVec[hash] == key) {
                                        //key is found in hash table
                                        break;
                                    }
                                    else if (globalHashVec[hash] == -1) {
                                        //key is not registered yet
                                        globalHashVec[hash] = key; // Register the key
                                        nnzPerCol[c]++; // New nz in the column after merge
                                        windowsSymbolic[wsIdx].second++; // New nz in the window after merge
                                        break;
                                    }
                                    else {
                                        //key is not found
                                        hash = (hash+1) & (htSize-1);
                                    }
                                }
                            }
                        }
                    }
                    else{
                        IT nrowsPerWindow = mdim / nWindowPerColSymbolic[c];
                        IT runningSum = 0;
                        for(size_t w = 0; w < nWindowPerColSymbolic[c]; w++){
                            IT rowStart = w * nrowsPerWindow;
                            IT rowEnd = (w == nWindowPerColSymbolic[c]-1) ? mdim : (w+1) * nrowsPerWindow;
                            size_t wsIdx = prefixSumWindowSymbolic[c] + w;

                            windowsSymbolic[wsIdx].first = rowStart;
                            windowsSymbolic[wsIdx].second = 0;
                            
                            size_t flopsWindow = 0;
                            for(int l = 0; l < nlists; l++){
                                std::tuple<IT, IT, NT>* first = ArrSpTups[l]->tuples + colPtrs[l][c];
                                std::tuple<IT, IT, NT>* last = ArrSpTups[l]->tuples + colPtrs[l][c+1];

                                if(rowStart > 0){
                                    std::tuple<IT, IT, NT> searchTuple(rowStart, 0, NT());
                                    first = std::lower_bound(first, last, searchTuple, rowCmp);
                                }

                                if(rowEnd < mdim){
                                    std::tuple<IT, IT, NT> searchTuple(rowEnd, 0, NT());
                                    last = std::lower_bound(first, last, searchTuple, rowCmp);
                                }

                                rowIdsRange[s][l].first = first - (ArrSpTups[l]->tuples);
                                rowIdsRange[s][l].second = last - (ArrSpTups[l]->tuples);

                                flopsWindow += last - first;
                            }
                            size_t htSize = minHashTableSize;
                            while(htSize < flopsWindow) {
                                //ht_size is set as 2^n
                                htSize <<= 1;
                            }
                            if(globalHashVec.size() < htSize) globalHashVec.resize(htSize);
                            for(size_t j=0; j < htSize; ++j) {
                                globalHashVec[j] = -1;
                            }
                            for(int l = 0; l < nlists; l++){
                                for(IT i = rowIdsRange[s][l].first; i < rowIdsRange[s][l].second; i++){
                                    IT key = ArrSpTups[l]->rowindex(i);
                                    IT hash = (key * hashScale) & (htSize-1);
                                    while (1) {
                                        //hash probing
                                        if (globalHashVec[hash] == key) {
                                            //key is found in hash table
                                            break;
                                        }
                                        else if (globalHashVec[hash] == -1) {
                                            //key is not registered yet
                                            globalHashVec[hash] = key;
                                            nnzPerCol[c]++;
                                            windowsSymbolic[wsIdx].second++;
                                            break;
                                        }
                                        else {
                                            //key is not found
                                            hash = (hash+1) & (htSize-1);
                                        }
                                    }
                                }
                            }

                            if(w == 0){
                                runningSum = windowsSymbolic[wsIdx].second;
                            }
                            else{
                                if(runningSum + windowsSymbolic[wsIdx].second > maxHashTableSize) {
                                    nWindowPerCol[c]++;
                                    runningSum = windowsSymbolic[wsIdx].second;
                                }
                                else{
                                    runningSum += windowsSymbolic[wsIdx].second;
                                }
                            }
                        }
                    }
                }
            }
        }

        /*
         * Now collapse symbolic windows to get windows of actual computation
         * */
        IT* prefixSumWindow = prefixSum<IT>(nWindowPerCol, ndim, nthreads);
        std::pair<IT, IT>* windows = static_cast<std::pair<IT, IT>*> (::operator new (sizeof(std::pair<IT, IT>[prefixSumWindow[ndim]])));

#ifdef THREADED
#pragma omp parallel for schedule(dynamic)
#endif
        for(int s = 0; s < nsplits; s++){
            IT colStart = colSplitters[s];
            IT colEnd = colSplitters[s+1];
            for(IT c = colStart; c < colEnd; c++){
                IT nWindowSymbolic = nWindowPerColSymbolic[c];
                IT wsIdx = prefixSumWindowSymbolic[c];
                IT wcIdx = prefixSumWindow[c];
                windows[wcIdx].first = windowsSymbolic[wsIdx].first;
                windows[wcIdx].second = windowsSymbolic[wsIdx].second;
                // w = 0 is already taken care of. So start from w = 1
                for(IT w = 1; w < nWindowSymbolic; w++){ 
                    wsIdx = prefixSumWindowSymbolic[c] + w;
                    if(windows[wcIdx].second + windowsSymbolic[wsIdx].second > maxHashTableSize){
                        wcIdx++;
                        windows[wcIdx].first = windowsSymbolic[wsIdx].first;
                        windows[wcIdx].second = windowsSymbolic[wsIdx].second;
                    }
                    else{
                        windows[wcIdx].second = windows[wcIdx].second + windowsSymbolic[wsIdx].second;
                    }
                }
            }
        }

        IT* prefixSumNnzPerCol = prefixSum<IT>(nnzPerCol, ndim, nthreads);
        IT totalNnz = prefixSumNnzPerCol[ndim];
        std::tuple<IT, IT, NT> * mergeBuf = static_cast<std::tuple<IT, IT, NT>*> (::operator new (sizeof(std::tuple<IT, IT, NT>[totalNnz])));

#ifdef THREADED
#pragma omp parallel
#endif
        {
            std::vector< std::pair<uint32_t,NT> > globalHashVec(minHashTableSize);
            size_t tid = omp_get_thread_num();
#ifdef THREADED
#pragma omp for schedule(dynamic)
#endif
            for(int s = 0; s < nsplits; s++){
                IT startCol = colSplitters[s];
                IT endCol = colSplitters[s+1];
                for(IT c = startCol; c < endCol; c++){
                    IT nWindow = nWindowPerCol[c];
                    IT outptr = prefixSumNnzPerCol[c];
                    if(nWindow == 1){
                        IT wcIdx = prefixSumWindow[c];
                        IT nnzWindow = windows[wcIdx].second;

                        size_t htSize = minHashTableSize;
                        while(htSize < nnzWindow) 
                        {
                            //htSize is set as 2^n
                            htSize <<= 1;
                        }
                        if(globalHashVec.size() < htSize) globalHashVec.resize(htSize);
                        for(size_t j=0; j < htSize; ++j)
                        {
                            globalHashVec[j].first = -1;
                        }

                        for(int l = 0; l < nlists; l++){
                            for(IT i = colPtrs[l][c]; i < colPtrs[l][c+1]; i++){
                                IT key = ArrSpTups[l]->rowindex(i);
                                IT hash = (key * hashScale) & (htSize-1);
                                while (1) {
                                    //hash probing
                                    if (globalHashVec[hash].first == key) {
                                        //key is found in hash table
                                        // Add to the previos value stored in the position
                                        globalHashVec[hash].second += ArrSpTups[l]->numvalue(i);
                                        break;
                                    }
                                    else if (globalHashVec[hash].first == -1) {
                                        //key is not registered yet
                                        // Register the key and store the value
                                        globalHashVec[hash].first = key;
                                        globalHashVec[hash].second = ArrSpTups[l]->numvalue(i);
                                        break;
                                    }
                                    else {
                                        //key is not found
                                        hash = (hash+1) & (htSize-1);
                                    }
                                }
                            }
                        }
                        if(sorted){
                            size_t index = 0;
                            for (size_t j=0; j < htSize; j++){
                                if (globalHashVec[j].first != -1){
                                    globalHashVec[index] = globalHashVec[j];
                                    index++;
                                }
                            }
                            integerSort<NT>(globalHashVec.data(), index);
                            //std::sort(globalHashVec.begin(), globalHashVec.begin() + index, sort_less<IT, NT>);
                            for(size_t j = 0; j < index; j++){
                                mergeBuf[outptr] = std::tuple<IT, IT, NT>(globalHashVec[j].first, c, globalHashVec[j].second);
                                outptr++;
                            }
                        }
                        else{
                            for (size_t j=0; j < htSize; j++){
                                if (globalHashVec[j].first != -1){
                                    mergeBuf[outptr] = std::tuple<IT, IT, NT>(globalHashVec[j].first, c, globalHashVec[j].second);
                                    outptr++;
                                }
                            }
                        }
                    }
                    else{
                        for (int l = 0; l < nlists; l++){
                            rowIdsRange[s][l].first = colPtrs[l][c];
                            rowIdsRange[s][l].second = colPtrs[l][c+1];
                        }

                        for (size_t w = 0; w < nWindow; w++){
                            IT wcIdx = prefixSumWindow[c] + w;
                            IT startRow = windows[wcIdx].first;
                            IT endRow = (w == nWindow-1) ? mdim : windows[wcIdx+1].first;
                            IT nnzWindow = windows[wcIdx].second;

                            size_t htSize = minHashTableSize;
                            while(htSize < nnzWindow) htSize <<= 1;
                            if(globalHashVec.size() < htSize) globalHashVec.resize(htSize);
                            for(size_t j = 0; j < htSize; j++) globalHashVec[j].first = -1;

                            for(int l = 0; l < nlists; l++){
                                while( rowIdsRange[s][l].first < rowIdsRange[s][l].second ){
                                    IT i = rowIdsRange[s][l].first;
                                    IT key = ArrSpTups[l]->rowindex(i);
                                    if(key >= endRow) break;
                                    IT hash = (key * hashScale) & (htSize-1);
                                    while (1) {
                                        if (globalHashVec[hash].first == key) {
                                            //key is found in hash table
                                            // Add to the previos value stored in the position
                                            globalHashVec[hash].second += ArrSpTups[l]->numvalue(i);
                                            break;
                                        }
                                        else if (globalHashVec[hash].first == -1) {
                                            //key is not registered yet
                                            // Register the key and store the value
                                            globalHashVec[hash].first = key;
                                            globalHashVec[hash].second = ArrSpTups[l]->numvalue(i);
                                            break;
                                        }
                                        else {
                                            //key is not found
                                            hash = (hash+1) & (htSize-1);
                                        }
                                    }
                                    rowIdsRange[s][l].first++;
                                }
                            }
                            if(sorted){
                                size_t index = 0;
                                for (size_t j=0; j < htSize; j++){
                                    if (globalHashVec[j].first != -1){
                                        globalHashVec[index++] = globalHashVec[j];
                                    }
                                }
                                integerSort<NT>(globalHashVec.data(), index);
                                //std::sort(globalHashVec.begin(), globalHashVec.begin() + index, sort_less<IT, NT>);
                                for(size_t j = 0; j < index; j++){
                                    mergeBuf[outptr] = std::tuple<IT, IT, NT>(globalHashVec[j].first, c, globalHashVec[j].second);
                                    outptr++;
                                }
                            }
                            else{
                                for (size_t j=0; j < htSize; j++){
                                    if (globalHashVec[j].first != -1){
                                        mergeBuf[outptr] = std::tuple<IT, IT, NT>(globalHashVec[j].first, c, globalHashVec[j].second);
                                        outptr++;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        // Delete all allocated memories by prefixSum function
        delete [] prefixSumFlopsPerCol;
        delete [] prefixSumNnzPerCol;
        delete [] prefixSumWindowSymbolic;
        delete [] prefixSumWindow;
        
        // Delete rest with operator delete as all memories was allocated with operator new
        ::operator delete(colSplitters);
        for(int s = 0; s < nsplits; s++) ::operator delete(rowIdsRange[s]);
        ::operator delete(rowIdsRange);

        ::operator delete(nWindowPerColSymbolic);
        ::operator delete(windowsSymbolic);
        ::operator delete(nWindowPerCol);
        ::operator delete(windows);
        
        ::operator delete(flopsPerCol);
        ::operator delete(nnzPerCol);

        for(int l = 0; l < nlists; l++) ::operator delete(colPtrs[l]);
        ::operator delete(colPtrs);

        for(int i=0; i< nlists; i++)
        {
            if(delarrs)
                delete ArrSpTups[i]; // May be expensive for large local matrices
        }
        
        // Caution: We allow both sorted and unsorted tuples in SpTuples
        // By setting sorted=true, we prevented sorting in the SpTuples constructor
        // TODO: we better keep a isSorted flag in SpTuples (also in DCSC/CSC)
        return new SpTuples<IT, NT> (totalNnz, mdim, ndim, mergeBuf, true, true);
    }

}

#endif
