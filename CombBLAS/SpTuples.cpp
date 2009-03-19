/****************************************************************/
/* Sequential and Parallel Sparse Matrix Multiplication Library */
/* version 2.1 --------------------------------------------------/
/* date: 07/11/2007 ---------------------------------------------/
/* main author: Aydin Buluc (aydin@cs.ucsb.edu) -----------------/
/* contributors: Fenglin Liao (fenglin@cs.ucsb.edu) -------------/
/****************************************************************/



#include "SparseTriplets.h"

using namespace boost;
using namespace std;

template <class T>
SparseTriplets<T>::SparseTriplets(ITYPE size, ITYPE nRow, ITYPE nCol)
:SparseMatrix<T, SparseTriplets<T> >(size,nRow,nCol),nz(size)
{
	if(size > 0)
	{
		tuples  = new tuple<ITYPE, ITYPE, T>[nzmax];
	}
}

template <class T>
SparseTriplets<T>::SparseTriplets (ITYPE size, ITYPE nRow, ITYPE nCol, tuple<ITYPE, ITYPE, T> * mytuples)
:SparseMatrix<T, SparseTriplets<T> >(size,nRow,nCol),nz(size), tuples(mytuples)
{}


template <class T>
SparseTriplets<T>::~SparseTriplets()
{
	if(nz > 0)
	{
		delete [] tuples;
	}
}

// Hint1: copy constructor (constructs a new object. i.e. this is NEVER called on an existing object)
// Hint2: Derived's copy constructor must make sure that Base's copy constructor is invoked 
//		  instead of Base's default constructor
template <class T>
SparseTriplets<T>::SparseTriplets(const SparseTriplets<T> & rhs): SparseMatrix<T, SparseTriplets<T> >(rhs), nz(rhs.nz)
{
	tuples  = new boost::tuple<ITYPE, ITYPE, T>[nzmax];
	if(nz > 0)
	{
		for(ITYPE i=0; i< nz; i++)
		{
			tuples[i] = rhs.tuples[i];
		}
	}
}

//! Constructor for converting SparseDColumn matrix -> SparseTriplets 
template <class T>
SparseTriplets<T>::SparseTriplets (const SparseDColumn<T> & rhs): SparseMatrix<T, SparseTriplets<T> >(rhs),  nz(rhs.nzmax)
{
	if(nz > 0)
	{
		FillTuples(rhs.dcsc);
	}
}

template <class T>
inline void SparseTriplets<T>::FillTuples (Dcsc<T> * mydcsc)
{
	tuples  = new boost::tuple<ITYPE, ITYPE, T>[nzmax];

	ITYPE index = 0;
	for(ITYPE i = 0; i< mydcsc->nzc; i++)
	{
		for(ITYPE j = mydcsc->mas[i]; j< mydcsc->mas[i+1]; j++)
		{
			boost::get<1>(tuples[index]) = mydcsc->jc[i];
			boost::get<0>(tuples[index]) = mydcsc->ir[j];
			boost::get<2>(tuples[index]) = mydcsc->numx[j];
			index++;
		}
	}
}
	


// Hint1: The assignment operator (operates on an existing object)
// Hint2: The assignment operator is the only operator that is not inherited.
//		  Make sure that base class data are also updated during assignment
template <class T>
SparseTriplets<T> & SparseTriplets<T>::operator=(const SparseTriplets<T> & rhs)
{
	if(this != &rhs)	// "this" pointer stores the address of the class instance
	{
		if(nzmax > 0)
		{
			// make empty
			delete [] tuples;
		}
		SparseMatrix<T, SparseTriplets<T> >::operator=(rhs);
		nz		= rhs.nz;

		if(nzmax > 0)
		{
			tuples  = new boost::tuple<ITYPE, ITYPE, T>[nzmax];

			for(ITYPE i=0; i< nz; i++)
			{
				tuples[i] = rhs.tuples[i];
			}
		}
	}
	return *this;
}


//! Loads a triplet matrix from infile
//! \remarks Assumes matlab type indexing for the input (i.e. indices start from 1)
template <class T>
ifstream& operator>> (ifstream& infile, SparseTriplets<T> & s)
{
	if (infile.is_open())
	{
		while (! infile.eof() && s.nz < s.nzmax)
		{
			infile >> boost::get<0>(s.tuples[s.nz]) >> boost::get<1>(s.tuples[s.nz]) >> boost::get<2>(s.tuples[s.nz]);	// row-col-value
			
			boost::get<0>(s.tuples[s.nz]) --;
			boost::get<1>(s.tuples[s.nz]) --;
			
			if((boost::get<0>(s.tuples[s.nz]) > s.m) || (boost::get<1>(s.tuples[s.nz]) > s.n))
			{
				cerr << "supplied matrix indices are beyond specified boundaries, aborting..." << endl;
				abort();
			}
			++s.nz;
		}
	}
	return infile;
}

//! Output to a triplets file
//! \remarks Uses matlab type indexing for the output (i.e. indices start from 1)
template <class T>
ofstream& operator<< (ofstream& outfile, const SparseTriplets<T> & s)
{
	ITYPE i = 0;
	outfile << s.m <<"\t"<< s.n <<"\t"<< s.nz<<endl;
	while (i < s.nz)
	{
		outfile << (get<0>(s.tuples[i])+1) <<"\t"<< (get<1>(s.tuples[i])+1) <<"\t"
			<< get<2>(s.tuples[i])<< endl;	// row-col-value
		i++;
	}
	return outfile;
}

template <class T>
const SparseTriplets<T> operator* (const SparseTriplets<T> & r,const SparseTriplets<T> & s )
{
	SparseTriplets<T> C;
	cout<< "Multiplication of SparseTriplets not yet supported"<<endl;
	return C;
}

