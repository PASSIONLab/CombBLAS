/****************************************************************/
/* Sequential and Parallel Sparse Matrix Multiplication Library */
/* version 2.3 --------------------------------------------------/
/* date: 01/18/2009 ---------------------------------------------/
/* author: Aydin Buluc (aydin@cs.ucsb.edu) ----------------------/
/****************************************************************/

#include "SpTuples.h"
#include <iomanip>

template <class IT, class NT>
const IT SpTuples<IT,NT>::zero = static_cast<IT>(0);


template <class IT,class NT>
SpTuples<IT,NT>::SpTuples(IT size, IT nRow, IT nCol)
:m(nRow), n(nCol), nnz(size)
{
	if(nnz > 0)
	{
		tuples  = new tuple<IT, IT, NT>[nnz];
	}
	else
	{
		tuples = NULL;
	}
}

template <class IT,class NT>
SpTuples<IT,NT>::SpTuples (IT size, IT nRow, IT nCol, tuple<IT, IT, NT> * mytuples)
:m(nRow), n(nCol), nnz(size), tuples(mytuples)
{}

/**
  * Generate a SpTuples object from StackEntry array, then delete that array
  * @param[StackEntry] multstack {value-key pairs where keys are pair<col_ind, row_ind> sorted lexicographically} 
 **/  
template <class IT, class NT>
SpTuples<IT,NT>::SpTuples (IT size, IT nRow, IT nCol, StackEntry<NT, pair<IT,IT> > * & multstack)
:m(nRow), n(nCol), nnz(size)
{
	if(nnz > 0)
	{
		tuples  = new tuple<IT, IT, NT>[nnz];
	}
	for(IT i=0; i<nnz; ++i)
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
	if(nnz > 0)
	{
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
	tuples  = new tuple<IT, IT, NT>[nnz];
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
}

template <class IT,class NT>
inline void SpTuples<IT,NT>::FillTuples (Dcsc<IT,NT> * mydcsc)
{
	tuples  = new tuple<IT, IT, NT>[nnz];

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
			delete [] tuples;
		}
		m = rhs.m;
		n = rhs.n;
		nnz = rhs.nnz;

		if(nnz> 0)
		{
			tuples  = new tuple<IT, IT, NT>[nnz];

			for(IT i=0; i< nnz; ++i)
			{
				tuples[i] = rhs.tuples[i];
			}
		}
	}
	return *this;
}


//! Loads a triplet matrix from infile
//! \remarks Assumes matlab type indexing for the input (i.e. indices start from 1)
template <class IT,class NT>
ifstream& SpTuples<IT,NT>::get (ifstream& infile)
{
	cout << "Getting... SpTuples" << endl;
	IT cnz = SpTuples<IT,NT>::zero;
	if (infile.is_open())
	{
		while ( (!infile.eof()) && cnz < nnz)
		{
			infile >> rowindex(cnz) >> colindex(cnz) >>  numvalue(cnz);	// row-col-value
			
			rowindex(cnz) --;
			colindex(cnz) --;
			
			if((rowindex(cnz) > m) || (colindex(cnz)  > n))
			{
				cerr << "supplied matrix indices are beyond specified boundaries, aborting..." << endl;
			}
			++cnz;
		}
		assert(nnz == cnz);
	}
	else
	{
		cerr << "input file is not open!" << endl;
	}
	return infile;
}

//! Output to a triplets file
//! \remarks Uses matlab type indexing for the output (i.e. indices start from 1)
template <class IT,class NT>
ofstream& SpTuples<IT,NT>::put(ofstream& outfile) const
{
	outfile << m <<"\t"<< n <<"\t"<< nnz<<endl;
	for (IT i = 0; i < nnz; ++i)
	{
		outfile << rowindex(i)+1  <<"\t"<< colindex(i)+1 <<"\t"
			<< numvalue(i) << endl;
	}
	return outfile;
}

template <class IT,class NT>
void SpTuples<IT,NT>::PrintInfo()
{
	cout << "This is a SpTuples class" << endl;

	cout << "m: " << m ;
	cout << ", n: " << n ;
	cout << ", nnz: "<< nnz << endl;

	if(m < 8 && n < 8)	// small enough to print
	{
		NT ** A = SpHelper::allocate2D<NT>(m,n);
		for(IT i=zero; i< m; ++i)
			for(IT j=zero; j<n; ++j)
				A[i][j] = 0.0;
		
		for(IT i=zero; i< nnz; ++i)
		{
			A[rowindex(i)][colindex(i)] = numvalue(i);			
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
