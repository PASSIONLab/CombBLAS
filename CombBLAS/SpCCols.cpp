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


#include "SpCCols.h"
#include "Deleter.h"
#include <algorithm>
#include <functional>
#include <vector>
#include <climits>
#include <iomanip>
#include <cassert>

using namespace std;

/****************************************************************************/
/********************* PUBLIC CONSTRUCTORS/DESTRUCTORS **********************/
/****************************************************************************/

template <class IT, class NT>
const IT SpCCols<IT,NT>::esscount = static_cast<IT>(3);


template <class IT, class NT>
SpCCols<IT,NT>::SpCCols():csc(NULL), m(0), n(0), nnz(0), splits(0){
}

// Allocate all the space necessary
template <class IT, class NT>
SpCCols<IT,NT>::SpCCols(IT size, IT nRow, IT nCol)
:m(nRow), n(nCol), nnz(size), splits(0)
{
	if(nnz > 0)
		csc = new Csc<IT,NT>(nnz, n);
	else
		csc = NULL;
}

template <class IT, class NT>
SpCCols<IT,NT>::~SpCCols()
{
	if(nnz > 0)
	{
		if(csc != NULL)
		{	
			if(splits > 0)
			{
				for(int i=0; i<splits; ++i)
					delete cscarr[i];
				delete [] cscarr;
			}
			else
			{
				delete csc;
			}
		}
	}
}

// Copy constructor (constructs a new object. i.e. this is NEVER called on an existing object)
// Derived's copy constructor can safely call Base's default constructor as base has no data members 
template <class IT, class NT>
SpCCols<IT,NT>::SpCCols(const SpCCols<IT,NT> & rhs)
: m(rhs.m), n(rhs.n), nnz(rhs.nnz), splits(rhs.splits)
{
	if(splits > 0)
	{
		for(int i=0; i<splits; ++i)
			CopyCsc(rhs.cscarr[i]);
	}
	else
	{
		CopyCsc(rhs.csc);
	}
}

/** 
 * Constructor for converting SpTuples matrix -> SpCCols
 * @param[in] 	rhs if transpose=true, 
 *	\n		then rhs is assumed to be a row sorted SpTuples object 
 *	\n		else rhs is assumed to be a column sorted SpTuples object
 **/
template <class IT, class NT>
SpCCols<IT,NT>::SpCCols(const SpTuples<IT, NT> & rhs, bool transpose)
: m(rhs.m), n(rhs.n), nnz(rhs.nnz), splits(0)
{	 
	
	if(nnz == 0)	// m by n matrix of complete zeros
	{
		if(transpose) swap(m,n);
		csc = NULL;
	} 
	else
	{
		if(transpose)
		{
			swap(m,n);
			csc = new Csc<IT,NT>(nnz,n);    // the swap is already done here
            vector< pair<IT,NT> > tosort (nnz);
            vector<IT> work(n+1, (IT) 0 );	// workspace, zero initialized, first entry stays zero
            for (IT k = 0 ; k < nnz ; ++k)
            {
                IT tmp =  rhs.rowindex(k);
                work [ tmp+1 ]++ ;		// column counts (i.e, w holds the "col difference array")
            }
            if(nnz > 0)
            {
                std::partial_sum(work.begin(), work.end(), work.begin());
                copy(work.begin(), work.end(), csc->jc);
                IT last;
                for (IT k = 0 ; k < nnz ; ++k)
                {
                    tosort[ work[ rhs.rowindex(k) ]++] = make_pair( rhs.colindex(k), rhs.numvalue(k));
                }
                #ifdef _OPENMP
                #pragma omp parallel for
                #endif
                for(int i=0; i< n; ++i)
                {
                    sort(tosort.begin() + csc->jc[i], tosort.begin() + csc->jc[i+1]);
                    
                    IT ind;
                    typename vector<pair<IT,NT> >::iterator itr;	// iterator is a dependent name
                    for(itr = tosort.begin() + csc->jc[i], ind = csc->jc[i]; itr != tosort.begin() + csc->jc[i+1]; ++itr, ++ind)
                    {
                        csc->ir[ind] = itr->first;
                        csc->num[ind] = itr->second;
                    }
                }
            }
	 	}
		else
		{
            csc = new Csc<IT,NT>(nnz,n);    // the swap is already done here
            vector< pair<IT,NT> > tosort (nnz);
            vector<IT> work(n+1, (IT) 0 );	// workspace, zero initialized, first entry stays zero
            for (IT k = 0 ; k < nnz ; ++k)
            {
                IT tmp =  rhs.colindex(k);
                work [ tmp+1 ]++ ;		// column counts (i.e, w holds the "col difference array")
            }
            if(nnz > 0)
            {
                std::partial_sum(work.begin(), work.end(), work.begin());
                copy(work.begin(), work.end(), csc->jc);
                IT last;
                for (IT k = 0 ; k < nnz ; ++k)
                {
                    tosort[ work[ rhs.colindex(k) ]++] = make_pair( rhs.rowindex(k), rhs.numvalue(k));
                }
                #ifdef _OPENMP
                #pragma omp parallel for
                #endif
                for(int i=0; i< n; ++i)
                {
                    sort(tosort.begin() + csc->jc[i], tosort.begin() + csc->jc[i+1]);
                    
                    IT ind;
                    typename vector<pair<IT,NT> >::iterator itr;	// iterator is a dependent name
                    for(itr = tosort.begin() + csc->jc[i], ind = csc->jc[i]; itr != tosort.begin() + csc->jc[i+1]; ++itr, ++ind)
                    {
                        csc->ir[ind] = itr->first;
                        csc->num[ind] = itr->second;
                    }
                }
            }
		}
	}
}

/****************************************************************************/
/************************** PUBLIC OPERATORS ********************************/
/****************************************************************************/

/**
 * The assignment operator operates on an existing object
 * The assignment operator is the only operator that is not inherited.
 * But there is no need to call base's assigment operator as it has no data members
 */
template <class IT, class NT>
SpCCols<IT,NT> & SpCCols<IT,NT>::operator=(const SpCCols<IT,NT> & rhs)
{
    // this pointer stores the address of the class instance
    // check for self assignment using address comparison
    if(this != &rhs)
    {
        if(csc != NULL && nnz > 0)
        {
            delete csc;
        }
        if(rhs.csc != NULL)
        {
            csc = new Csc<IT,NT>(*(rhs.csc));
            nnz = rhs.nnz;
        }
        else
        {
            csc = NULL;
            nnz = 0;
        }
        
        m = rhs.m; 
        n = rhs.n;
        splits = rhs.splits;
    }
    return *this;
}


template <class IT, class NT>
void SpCCols<IT,NT>::RowSplit(int numsplits)
{
    splits = numsplits;
    IT perpiece = m / splits;
    vector<IT> nnzs(splits, 0);
    vector < vector < tuple<IT,IT,NT> > > colrowpairs(splits);
    vector< vector<IT> > colcnts(splits);
    for(int i=0; i< splits; ++i)
        colcnts[i].resize(n, 0);
    
    if(nnz > 0 && csc != NULL)
    {
        for(IT i=0; i< csc->n; ++i)
        {
            for(IT j = csc->jc[i]; j< csc->jc[i+1]; ++j)
            {
                IT rowid = csc->ir[j];  // colid=i
                IT owner = min(rowid / perpiece, static_cast<IT>(splits-1));
                colrowpairs[owner].push_back(make_tuple(i, rowid - owner*perpiece, csc->num[i]));
                
                ++(colcnts[owner][i]);
                ++(nnzs[owner]);
            }
        }
    }
    delete csc;	// claim memory
    cscarr = new Csc<IT,NT>*[splits];
    
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for(int i=0; i< splits; ++i)    // i iterates over splits
    {
        cscarr[i] = new Csc<IT,NT>(nnzs[i],n);
        sort(colrowpairs[i].begin(), colrowpairs[i].end());	// sort w.r.t. columns first and rows second
        cscarr[i]->jc[0]  = 0;
        std::partial_sum(colcnts[i].begin(), colcnts[i].end(), cscarr[i]->jc+1);
        std::copy(cscarr[i]->jc, cscarr[i]->jc+n, colcnts[i].begin());   // reuse the colcnts as "current column pointers"
        
        
        for(IT k=0; k<nnzs[i]; ++k) // k iterates over all nonzeros
        {
            IT cindex = get<0>(colrowpairs[i][k]);
            IT rindex = get<1>(colrowpairs[i][k]);
            NT value = get<2>(colrowpairs[i][k]);
            
            IT curcptr = (colcnts[i][cindex])++;   // fetch the pointer and post increment
            cscarr[i]->ir[curcptr] = rindex;
            cscarr[i]->num[curcptr] = value;
        }
    }
}


template<class IT, class NT>
void SpCCols<IT,NT>::PrintInfo() const
{
    cout << "m: " << m ;
    cout << ", n: " << n ;
    cout << ", nnz: "<< nnz ;
    
    if(splits > 0)
    {
        cout << ", local splits: " << splits << endl;
#ifdef _OPENMP
        if(omp_get_thread_num() == 0)
        {
            SubPrintInfo(cscarr[0]);
        }
#endif
    }
    else
    {
        cout << endl;
        SubPrintInfo(csc);
    }
}




/****************************************************************************/
/************************* PRIVATE MEMBER FUNCTIONS *************************/
/****************************************************************************/


template <class IT, class NT>
void SpCCols<IT,NT>::SubPrintInfo(Csc<IT,NT> * mycsc) const
{
#ifdef _OPENMP
    cout << "Printing for thread " << omp_get_thread_num() << endl;
#endif
    if(m < PRINT_LIMIT && n < PRINT_LIMIT)	// small enough to print
    {
        NT ** A = SpHelper::allocate2D<NT>(m,n);
        for(IT i=0; i< m; ++i)
            for(IT j=0; j<n; ++j)
                A[i][j] = NT();
        if(mycsc != NULL)
        {
            for(IT i=0; i< n; ++i)
            {
                for(IT j = mycsc->jc[i]; j< mycsc->jc[i+1]; ++j)
                {
                    IT rowid = mycsc->ir[j];
                    A[rowid][i] = mycsc->num[j];
                }
            }
        }
        for(IT i=0; i< m; ++i)
        {
            for(IT j=0; j<n; ++j)
            {
                cout << setiosflags(ios::fixed) << setprecision(2) << A[i][j];
                cout << " ";
            }
            cout << endl;
        }
        SpHelper::deallocate2D(A,m);
    }
}


template <class IT, class NT>
inline void SpCCols<IT,NT>::CopyCsc(Csc<IT,NT> * source)
{
    // source csc will be NULL if number of nonzeros is zero
    if(source != NULL)
        csc = new Csc<IT,NT>(*source);
    else
        csc = NULL;
}


