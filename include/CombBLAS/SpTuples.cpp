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


#include "SpTuples.h"
#include "SpParHelper.h"
#include <iomanip>

namespace combblas {

template <class IT,class NT>
SpTuples<IT,NT>::SpTuples(int64_t size, IT nRow, IT nCol)
:m(nRow), n(nCol), nnz(size)
{
	if(nnz > 0)
	{
		tuples  = new std::tuple<IT, IT, NT>[nnz];
	}
	else
	{
		tuples = NULL;
	}
    isOperatorNew = false;
}

template <class IT,class NT>
SpTuples<IT,NT>::SpTuples (int64_t size, IT nRow, IT nCol, std::tuple<IT, IT, NT> * mytuples, bool sorted, bool isOpNew)
:tuples(mytuples), m(nRow), n(nCol), nnz(size), isOperatorNew(isOpNew)
{
    if(!sorted)
    {
        SortColBased();
    }
    
}

/**
  * Generate a SpTuples object from an edge list
  * @param[in,out] edges: edge list that might contain duplicate edges. freed upon return
  * Semantics differ depending on the object created:
  * NT=bool: duplicates are ignored
  * NT='countable' (such as short,int): duplicated as summed to keep count 	 
 **/  
template <class IT, class NT>
SpTuples<IT,NT>::SpTuples (int64_t maxnnz, IT nRow, IT nCol, std::vector<IT> & edges, bool removeloops):m(nRow), n(nCol)
{
	if(maxnnz > 0)
	{
		tuples  = new std::tuple<IT, IT, NT>[maxnnz];
	}
	for(int64_t i=0; i<maxnnz; ++i)
	{
		rowindex(i) = edges[2*i+0];
		colindex(i) = edges[2*i+1];
		numvalue(i) = (NT) 1;
	}
	std::vector<IT>().swap(edges);	// free memory for edges

	nnz = maxnnz;	// for now (to sort)
	SortColBased();

	int64_t cnz = 0;
	int64_t dup = 0;  int64_t self = 0;
	nnz = 0; 
	while(cnz < maxnnz)
	{
		int64_t j=cnz+1;
		while(j < maxnnz && rowindex(cnz) == rowindex(j) && colindex(cnz) == colindex(j)) 
		{
			numvalue(cnz) +=  numvalue(j);	
			numvalue(j++) = 0;	// mark for deletion
			++dup;
		}
		if(removeloops && rowindex(cnz) == colindex(cnz))
		{
			numvalue(cnz) = 0;
			--nnz;
			++self;
		}
		++nnz;
		cnz = j;
	}

	std::tuple<IT, IT, NT> * ntuples = new std::tuple<IT,IT,NT>[nnz];
	int64_t j = 0;
	for(int64_t i=0; i<maxnnz; ++i)
	{
		if(numvalue(i) != 0)
		{
			ntuples[j++] = tuples[i];
		}
	}
	assert(j == nnz);

    delete [] tuples;
	tuples = ntuples;
    isOperatorNew = false;
}


/**
  * Generate a SpTuples object from StackEntry array, then delete that array
  * @param[in] multstack {value-key pairs where keys are pair<col_ind, row_ind> sorted lexicographically} 
  * \remark Since input is column sorted, the tuples are automatically generated in that way too
 **/  
template <class IT, class NT>
SpTuples<IT,NT>::SpTuples (int64_t size, IT nRow, IT nCol, StackEntry<NT, std::pair<IT,IT> > * & multstack)
:m(nRow), n(nCol), nnz(size)
{
    isOperatorNew = false;
	if(nnz > 0)
	{
		tuples  = new std::tuple<IT, IT, NT>[nnz];
	}
	for(int64_t i=0; i<nnz; ++i)
	{
		colindex(i) = multstack[i].key.first;
		rowindex(i) = multstack[i].key.second;
		numvalue(i) = multstack[i].value;
    }
	delete [] multstack;
}


template <class IT,class NT>
SpTuples<IT,NT>::~SpTuples()
{
    // This tuples_deleted member is a temporary patch to avoid memory leak from MemEfficietnSpGEMM3D
	if((nnz > 0) && (tuples_deleted != true))
	{   
        if(isOperatorNew)
            ::operator delete(tuples);
        else
            delete [] tuples;
	}
}

/**
  * Hint1: copy constructor (constructs a new object. i.e. this is NEVER called on an existing object)
  * Hint2: Base's default constructor is called under the covers 
  *	  Normally Base's copy constructor should be invoked but it doesn't matter here as Base has no data members
  */
template <class IT,class NT>
SpTuples<IT,NT>::SpTuples(const SpTuples<IT,NT> & rhs): m(rhs.m), n(rhs.n), nnz(rhs.nnz)
{
	tuples  = new std::tuple<IT, IT, NT>[nnz];
    isOperatorNew = false;
	for(IT i=0; i< nnz; ++i)
	{
		tuples[i] = rhs.tuples[i];
	}
}

//! Constructor for converting SpDCCols matrix -> SpTuples 
template <class IT,class NT>
SpTuples<IT,NT>::SpTuples (const SpDCCols<IT,NT> & rhs):  m(rhs.m), n(rhs.n), nnz(rhs.nnz)
{
	if(nnz > 0)
	{
		FillTuples(rhs.dcsc);
	}
    isOperatorNew = false;
}



template <class IT, class NT>
SpTuples<IT, NT>::SpTuples (const SpCCols<IT, NT> &rhs) :
	m(rhs.m), n(rhs.n), nnz(rhs.nnz)
{
	isOperatorNew = false;
	if (nnz > 0)
	{
		tuples = new std::tuple<IT, IT, NT>[nnz];
		Csc<IT, NT> *csc = rhs.csc;
		IT k = 0;
		for (IT i = 0; i < csc->n; ++i)
		{
			for (IT j = csc->jc[i]; j < csc->jc[i + 1]; ++j)
			{
				colindex(k)	  = i;
				rowindex(k)	  = csc->ir[j];
				numvalue(k++) = csc->num[j];
			}
		}
	} 
}



template <class IT,class NT>
inline void SpTuples<IT,NT>::FillTuples (Dcsc<IT,NT> * mydcsc)
{
	tuples  = new std::tuple<IT, IT, NT>[nnz];
	IT k = 0;
	for(IT i = 0; i< mydcsc->nzc; ++i)
	{
		for(IT j = mydcsc->cp[i]; j< mydcsc->cp[i+1]; ++j)
		{
			colindex(k) = mydcsc->jc[i];
			rowindex(k) = mydcsc->ir[j];
			numvalue(k++) = mydcsc->numx[j];
		}
	}
}


// Hint1: The assignment operator (operates on an existing object)
// Hint2: The assignment operator is the only operator that is not inherited.
//		  Make sure that base class data are also updated during assignment
template <class IT,class NT>
SpTuples<IT,NT> & SpTuples<IT,NT>::operator=(const SpTuples<IT,NT> & rhs)
{
	if(this != &rhs)	// "this" pointer stores the address of the class instance
	{
		if(nnz > 0)
		{
			// make empty
            if(isOperatorNew)
                ::operator delete(tuples);
            else
                delete [] tuples;
		}
		m = rhs.m;
		n = rhs.n;
		nnz = rhs.nnz;
        isOperatorNew = false;

		if(nnz> 0)
		{
			tuples  = new std::tuple<IT, IT, NT>[nnz];
			for(IT i=0; i< nnz; ++i)
			{
				tuples[i] = rhs.tuples[i];
			}
		}
	}
	return *this;
}

/**
 * \pre {The object is either column-sorted or row-sorted, either way the identical entries will be consecutive}
 **/
template <class IT,class NT>
template <typename BINFUNC>
void SpTuples<IT,NT>::RemoveDuplicates(BINFUNC BinOp)
{
	if(nnz > 0)
	{
		std::vector< std::tuple<IT, IT, NT> > summed;
		summed.push_back(tuples[0]);
	
		for(IT i=1; i< nnz; ++i)
        	{
                if((joker::get<0>(summed.back()) == joker::get<0>(tuples[i])) && (joker::get<1>(summed.back()) == joker::get<1>(tuples[i])))
			{
				joker::get<2>(summed.back()) = BinOp(joker::get<2>(summed.back()), joker::get<2>(tuples[i]));
			}
			else
			{
				summed.push_back(tuples[i]);
				
			}
                }
        if(isOperatorNew)
            ::operator delete(tuples);
        else
            delete [] tuples;
		tuples  = new std::tuple<IT, IT, NT>[summed.size()];
        isOperatorNew = false;
    std::copy(summed.begin(), summed.end(), tuples);
		nnz =  summed.size();
	}
}


//! Loads a triplet matrix from infile
//! \remarks Assumes matlab type indexing for the input (i.e. indices start from 1)
template <class IT,class NT>
std::ifstream& SpTuples<IT,NT>::getstream (std::ifstream& infile)
{
	std::cout << "Getting... SpTuples" << std::endl;
	IT cnz = 0;
	if (infile.is_open())
	{
		while ( (!infile.eof()) && cnz < nnz)
		{
			infile >> rowindex(cnz) >> colindex(cnz) >>  numvalue(cnz);	// row-col-value
			
			rowindex(cnz) --;
			colindex(cnz) --;
			
			if((rowindex(cnz) > m) || (colindex(cnz)  > n))
			{
				std::cerr << "supplied matrix indices are beyond specified boundaries, aborting..." << std::endl;
			}
			++cnz;
		}
		assert(nnz == cnz);
	}
	else
	{
		std::cerr << "input file is not open!" << std::endl;
	}
	return infile;
}

//! Output to a triplets file
//! \remarks Uses matlab type indexing for the output (i.e. indices start from 1)
template <class IT,class NT>
std::ofstream& SpTuples<IT,NT>::putstream(std::ofstream& outfile) const
{
	outfile << m <<"\t"<< n <<"\t"<< nnz<<std::endl;
	for (IT i = 0; i < nnz; ++i)
	{
		outfile << rowindex(i)+1  <<"\t"<< colindex(i)+1 <<"\t"
			<< numvalue(i) << std::endl;
	}
	return outfile;
}

template <class IT,class NT>
void SpTuples<IT,NT>::PrintInfo()
{
	std::cout << "This is a SpTuples class" << std::endl;

	std::cout << "m: " << m ;
	std::cout << ", n: " << n ;
	std::cout << ", nnz: "<< nnz << std::endl;

	for(IT i=0; i< nnz; ++i)
	{
		if(rowindex(i) < 0 || colindex(i) < 0)
		{
			std::cout << "Negative index at " << i << std::endl;
			return;
		}
		else if(rowindex(i) >= m || colindex(i) >= n)
		{
			std::cout << "Index " << i << " too big with values (" << rowindex(i) << ","<< colindex(i) << ")" << std::endl;
		}
	}

	if(m < 8 && n < 8)	// small enough to print
	{
		NT ** A = SpHelper::allocate2D<NT>(m,n);
		for(IT i=0; i< m; ++i)
			for(IT j=0; j<n; ++j)
				A[i][j] = 0.0;
		
		for(IT i=0; i< nnz; ++i)
		{
			A[rowindex(i)][colindex(i)] = numvalue(i);			
		} 
		for(IT i=0; i< m; ++i)
		{
                        for(IT j=0; j<n; ++j)
			{
                                std::cout << std::setiosflags(std::ios::fixed) << std::setprecision(2) << A[i][j];
				std::cout << " ";
			}
			std::cout << std::endl;
		}
		SpHelper::deallocate2D(A,m);
	}
}

}
