#include "../CombBLAS.h"


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
vector<RT> findColSplitters(SpTuples<IT,NT> * & spTuples, int nsplits)
{
    vector<RT> splitters(nsplits+1);
    splitters[0] = static_cast<RT>(0);
    ColLexiCompare<IT,NT> comp;
#pragma omp parallel for
    for(int i=1; i< nsplits; i++)
    {
        IT cur_col = i * (spTuples->getncol()/nsplits);
        tuple<IT,IT,NT> search_tuple(0, cur_col, 0);
        tuple<IT,IT,NT>* it = lower_bound (spTuples->tuples, spTuples->tuples + spTuples->getnnz(), search_tuple, comp);
        splitters[i] = (RT) (it - spTuples->tuples);
    }
    splitters[nsplits] = spTuples->getnnz();
    
    return splitters;
}




/*
 "Internal function" called by MultiwayMerge inside threaded region.
 Never called from outside.
 Assumption1: the input lists are already column sorted
 Assumption2: at least two lists are passed to this function
 Assumption3: the input and output lists are to be deleted by caller
 */

template<class SR, class IT, class NT>
SpTuples<IT,NT>* SerialMerge( const vector<SpTuples<IT,NT> *> & ArrSpTups, tuple<IT, IT, NT> * ntuples)
{
    int nlists =  ArrSpTups.size();
    ColLexiCompare<IT,int> heapcomp;
    vector<tuple<IT, IT, int>> heap(nlists); // if performance issue, create this outside of threaded region
    vector<IT> curptr(nlists, static_cast<IT>(0));
    IT estnnz = 0;
    IT hsize = 0;
    for(int i=0; i< nlists; ++i)
    {
        if(ArrSpTups[i]->getnnz()>0)
        {
            estnnz += ArrSpTups[i]->getnnz();
            heap[hsize++] = make_tuple(get<0>(ArrSpTups[i]->tuples[0]), get<1>(ArrSpTups[i]->tuples[0]), i);
        }
        
    }
    make_heap(heap.data(), heap.data()+hsize, not2(heapcomp));
    IT cnz = 0;
    
    while(hsize > 0)
    {
        pop_heap(heap.data(), heap.data() + hsize, not2(heapcomp));   // result is stored in heap[hsize-1]
        int source = get<2>(heap[hsize-1]);
        
        if( (cnz != 0) &&
           ((get<0>(ntuples[cnz-1]) == get<0>(heap[hsize-1])) && (get<1>(ntuples[cnz-1]) == get<1>(heap[hsize-1]))) )
        {
            get<2>(ntuples[cnz-1])  = SR::add(get<2>(ntuples[cnz-1]), ArrSpTups[source]->numvalue(curptr[source]++));
        }
        else
        {
            ntuples[cnz++] = ArrSpTups[source]->tuples[curptr[source]++];
        }
        
        if(curptr[source] != ArrSpTups[source]->getnnz())	// That array has not been depleted
        {
            heap[hsize-1] = make_tuple(get<0>(ArrSpTups[source]->tuples[curptr[source]]),
                                       get<1>(ArrSpTups[source]->tuples[curptr[source]]), source);
            push_heap(heap.data(), heap.data()+hsize, not2(heapcomp));
        }
        else
        {
            --hsize;
        }
    }
    return new SpTuples<IT,NT> (cnz, ArrSpTups[0]->getnrow(), ArrSpTups[0]->getncol(), ntuples, true);

}



// Performs a balanced merge of the array of SpTuples
// Assumes the input parameters are already column sorted
template<class SR, class IT, class NT>
SpTuples<IT, NT>* MultiwayMerge( vector<SpTuples<IT,NT> *> & ArrSpTups, IT mdim = 0, IT ndim = 0, bool delarrs = false )
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
            tuple<IT, IT, NT>* mergeTups = static_cast<tuple<IT, IT, NT>*>
                    (::operator new (sizeof(tuple<IT, IT, NT>[ArrSpTups[0]->getnnz()])));
            #pragma omp parallel for
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
            cerr << "Dimensions of SpTuples do not match on multiwayMerge()" << endl;
            return new SpTuples<IT,NT>(0,0,0);
        }
    }

    int nthreads;
#pragma omp parallel
    {
        nthreads = omp_get_num_threads();
    }
    int nsplits = 4*nthreads; // oversplit for load balance
    nsplits = min(nsplits, (int)ndim); // we cannot split a column
    vector< vector<IT> > colPtrs;
    for(int i=0; i< nlists; i++)
    {
        colPtrs.push_back(findColSplitters<IT>(ArrSpTups[i], nsplits)); // in parallel
    }
    
    // ------ estimate memory requirement after merge in each split ------
    vector<IT> nnzPerSplit(nsplits);
    IT nnzAll = static_cast<IT>(0);
    //#pragma omp parallel for
    for(int i=0; i< nsplits; i++)
    {
        IT t = static_cast<IT>(0);
        for(int j=0; j< nlists; ++j)
            t += colPtrs[j][i+1] - colPtrs[j][i];
        nnzPerSplit[i] = t;
        nnzAll += t;
    }

    
    
    // ------ allocate memory in a serial region ------
    vector<tuple<IT, IT, NT> *> mergeBuf(nsplits);
    for(int i=0; i< nsplits; i++)
    {
        mergeBuf[i] = static_cast<tuple<IT, IT, NT>*> (::operator new (sizeof(tuple<IT, IT, NT>[nnzPerSplit[i]])));
    }


     // ------ perform merge in parallel ------
    vector<SpTuples<IT,NT> *> listMergeTups(nsplits); // use the memory allocated in mergeBuf
#pragma omp parallel for schedule(dynamic)
    for(int i=0; i< nsplits; i++) // serially merge part by part
    {
        vector<SpTuples<IT,NT> *> listSplitTups(nlists);
        for(int j=0; j< nlists; ++j)
        {
            IT curnnz= colPtrs[j][i+1] - colPtrs[j][i];
            listSplitTups[j] = new SpTuples<IT, NT> (curnnz, mdim, ndim, ArrSpTups[j]->tuples + colPtrs[j][i], true);
        }
        listMergeTups[i] = SerialMerge<SR>(listSplitTups, mergeBuf[i]);
    }
    
    
    // ------ concatenate merged tuples processed by threads ------
    vector<IT> tdisp(nsplits+1);
    tdisp[0] = 0;
    for(int i=0; i<nsplits; ++i)
    {
        tdisp[i+1] = tdisp[i] + listMergeTups[i]->getnnz();
    }

    IT mergedListSize = tdisp[nsplits];
    tuple<IT, IT, NT>* shrunkTuples = static_cast<tuple<IT, IT, NT>*> (::operator new (sizeof(tuple<IT, IT, NT>[mergedListSize])));
    
#pragma omp parallel for schedule(dynamic)
    for(int i=0; i< nsplits; i++)
    {
        std::copy(listMergeTups[i]->tuples , listMergeTups[i]->tuples + listMergeTups[i]->getnnz(), shrunkTuples + tdisp[i]);
    }

    
    for(int i=0; i< nsplits; i++)
    {
        //::operator delete(listMergeTups[i]->tuples);
        ::operator delete(mergeBuf[i]);
    }
    
    for(int i=0; i< nlists; i++)
    {
        if(delarrs)
            delete ArrSpTups[i]; // this might be expensive for large local matrices
    }
    return new SpTuples<IT, NT> (mergedListSize, mdim, ndim, shrunkTuples, true);
}
