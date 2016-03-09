/****************************************************************/
/* Parallel Combinatorial BLAS Library (for Graph Computations) */
/* version 1.5 -------------------------------------------------*/
/* date: 10/09/2015 ---------------------------------------------*/
/* authors: Ariful Azad, Aydin Buluc, Adam Lugowski ------------*/
/****************************************************************/
/*
 Copyright (c) 2010-2015, The Regents of the University of California
 
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
                copy(work, work+n+1, csc->jc);
                IT last;
                for (IT k = 0 ; k < nnz ; ++k)
                {
                    tosort[ work[ rhs.rowindex(k) ]++] = make_pair( rhs.colindex(k), rhs.numvalue(k));
                }
                #pragma omp parallel for
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
                copy(work, work+n+1, csc->jc);
                IT last;
                for (IT k = 0 ; k < nnz ; ++k)
                {
                    tosort[ work[ rhs.colindex(k) ]++] = make_pair( rhs.rowindex(k), rhs.numvalue(k));
                }
                #pragma omp parallel for
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
/************************* PRIVATE MEMBER FUNCTIONS *************************/
/****************************************************************************/

template <class IT, class NT>
inline void SpCCols<IT,NT>::CopyCsc(Csc<IT,NT> * source)
{
    // source csc will be NULL if number of nonzeros is zero
    if(source != NULL)
        csc = new Csc<IT,NT>(*source);
    else
        csc = NULL;
}


