#include "CombBLAS.h"

namespace combblas {


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
        std::tuple<IT,IT,NT> search_tuple(0, cur_col, 0);
        std::tuple<IT,IT,NT>* it = std::lower_bound (spTuples->tuples, spTuples->tuples + spTuples->getnnz(), search_tuple, comp);
        splitters[i] = (RT) (it - spTuples->tuples);
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
    std::make_heap(heap.data(), heap.data()+hsize, std::not2(heapcomp));
    
    std::tuple<IT, IT, NT> curTuple;
    IT estnnz = 0;
    while(hsize > 0)
    {
      std::pop_heap(heap.data(), heap.data() + hsize, std::not2(heapcomp));   // result is stored in heap[hsize-1]
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
            std::push_heap(heap.data(), heap.data()+hsize, std::not2(heapcomp));
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
    std::make_heap(heap.data(), heap.data()+hsize, std::not2(heapcomp));
    IT cnz = 0;
    
    while(hsize > 0)
    {
      std::pop_heap(heap.data(), heap.data() + hsize, std::not2(heapcomp));   // result is stored in heap[hsize-1]
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
            std::push_heap(heap.data(), heap.data()+hsize, std::not2(heapcomp));
        }
        else
        {
            --hsize;
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
            std::tuple<IT, IT, NT>* mergeTups = static_cast<std::tuple<IT, IT, NT>*>
            (::operator new (sizeof(std::tuple<IT, IT, NT>[ArrSpTups[0]->getnnz()])));
#ifdef THREADED
#pragma omp parallel for
#endif
            for(int i=0; i<ArrSpTups[0]->getnnz(); i++)
                mergeTups[i] = ArrSpTups[0]->tuples[i];
            
            return new SpTuples<IT,NT> (ArrSpTups[0]->getnnz(), mdim, ndim, mergeTups, true);
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
   std::tuple<IT, IT, NT> * mergeBuf = static_cast<std::tuple<IT, IT, NT>*> (::operator new (sizeof(std::tuple<IT, IT, NT>[mergedNnzAll])));
    
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
    return new SpTuples<IT, NT> (mergedNnzAll, mdim, ndim, mergeBuf, true, true);
}

}
