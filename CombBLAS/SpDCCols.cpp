/****************************************************************/
/* Sequential and Parallel Sparse Matrix Multiplication Library */
/* version 2.3 --------------------------------------------------/
/* date: 01/18/2009 ---------------------------------------------/
/* author: Aydin Buluc (aydin@cs.ucsb.edu) ----------------------/
/****************************************************************/

#include "SpDCCols.h"
#include "Deleter.h"
#include <algorithm>
#include <functional>
#include <vector>
#include <climits>

/****************************************************************************/
/********************* PUBLIC CONSTRUCTORS/DESTRUCTORS **********************/
/****************************************************************************/

template <class IT, class NT>
SpDCCols<IT,NT>::SpDCCols():dcsc(NULL), m(0), n(0), nnz(0), localpool(NULL) {}


// Allocate all the space necessary
template <class IT, class NT>
SpDCCols<IT,NT>::SpDCCols(IT size, IT nRow, IT nCol, IT nzc)
:m(nRow), n(nCol), nnz(size), localpool(NULL)
{
	if(nnz > 0)
		dcsc = new Dcsc<IT,NT>(nnz, nzc);
	else
		dcsc = NULL; 
}

template <class IT, class NT>
SpDCCols<IT,NT>::~SpDCCols()
{
	if(nnz > 0)
	{
		if(dcsc != NULL) delete dcsc;	// call Dcsc's destructor
	}
}


// Copy constructor (constructs a new object. i.e. this is NEVER called on an existing object)
// Derived's copy constructor can safely call Base's default constructor as base has no data members 
template <class IT, class NT>
SpDCCols<IT,NT>::SpDCCols(const SpDCCols<IT,NT> & rhs)
: m(rhs.m), n(rhs.n), nnz(rhs.nnz), localpool(rhs.localpool)
{
	CopyDcsc(rhs.dcsc);
}

/** 
 * Constructor for converting SpTuples matrix -> SpDCCols (may use a private memory heap)
 * @param[in] 	rhs if transpose=true, 
 *	\n		then rhs is assumed to be a row sorted SpTuples object 
 *	\n		else rhs is assumed to be a column sorted SpTuples object
 * @param[in, out] mpool default parameter value is a null pointer, which means no special memory pool is used
 *	\n	if the parameter is supplied, then memory for MAS, JC, IR, NUMX are served from the given memory pool
 *	\n	also modifies the memory pool so that the used portions are no longer listed as free
 */
template <class IT, class NT>
SpDCCols<IT,NT>::SpDCCols(const SpTuples<IT,NT> & rhs, bool transpose, MemoryPool * mpool)
: m(rhs.m), n(rhs.n), nnz(rhs.nnz), localpool(mpool)
{	 
	if(nnz == 0)	// m by n matrix of complete zeros
	{
		dcsc = NULL;	
	} 
	else
	{
		if(transpose)
		{
			swap(m,n);
			IT localnzc = 1;
			for(IT i=1; i< rhs.getnnz(); ++i)
			{
				if(rhs.rowindex(i) != rhs.rowindex(i-1))
				{
					++localnzc;
	 			}
	 		}

			if(localpool == NULL)	// no special memory pool used
			{
				dcsc = new Dcsc<IT,NT>(rhs.getnnz(),localnzc);	
			}
			else
			{
				dcsc = new Dcsc<IT,NT>(rhs.getnnz(), localnzc, localpool);
	 		}		

			dcsc->jc[zero]  = rhs.rowindex(zero); 
			dcsc->mas[zero] = zero;

			for(IT i=0; i<rhs.getnnz(); ++i)
	 		{
				dcsc->ir[i]  = rhs.colindex(i);		// copy rhs.jc to ir since this transpose=true
				dcsc->numx[i] = rhs.numvalue(i);
			}

			IT jspos  = 1;		
			for(IT i=1; i<rhs.getnnz(); ++i)
			{
				if(rhs.rowindex(i) != dcsc->jc[jspos-1])
				{
					dcsc->jc[jspos] = rhs.rowindex(i);	// copy rhs.ir to jc since this transpose=true
					dcsc->mas[jspos++] = i;
				}
			}		
			dcsc->mas[jspos] = rhs.getnnz();
	 	}
		else
		{
			IT localnzc = 1;
			for(IT i=1; i<rhs.getnnz(); ++i)
			{
				if(rhs.colindex(i) != rhs.colindex(i-1))
				{
					++localnzc;
				}
			}
			if(localpool == NULL)	// no special memory pool used
			{
				dcsc = new Dcsc<IT,NT>(rhs.getnnz(),localnzc);	
			}
			else
			{
				dcsc = new Dcsc<IT,NT>(rhs.getnnz(),localnzc, localpool);
			}

			dcsc->jc[zero]  = rhs.colindex(zero); 
			dcsc->mas[zero] = zero;

			for(IT i=0; i<rhs.getnnz(); ++i)
			{
				dcsc->ir[i]  = rhs.rowindex(i);		// copy rhs.ir to ir since this transpose=false
				dcsc->numx[i] = rhs.numvalues(i);
			}

			IT jspos = 1;		
			for(IT i=1; i<rhs.getnnz(); ++i)
			{
				if(rhs.colindex(i) != dcsc->jc[jspos-1])
				{
					dcsc->jc[jspos] = rhs.colindex(i);	// copy rhs.jc to jc since this transpose=true
					dcsc->mas[jspos++] = i;
				}
			}		
			dcsc->mas[jspos] = rhs.getnnz();
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
SpDCCols<IT,NT> & SpDCCols<IT,NT>::operator=(const SpDCCols<IT,NT> & rhs)
{
	// this pointer stores the address of the class instance
	// check for self assignment using address comparison
	if(this != &rhs)		
	{
		m = rhs.m; 
		n = rhs.n;
		nnz = rhs.nnz; 
		if(dcsc != NULL && dcsc->nz > 0)
		{
			delete dcsc;
		}
		if(rhs.dcsc != NULL)	
		{
			dcsc = new Dcsc<IT,NT>(*(rhs.dcsc));
		}
	}
	return *this;
}

template <class IT, class NT>
SpDCCols<IT,NT> & SpDCCols<IT,NT>::operator+= (const SpDCCols<IT,NT> & rhs)
{
	// this pointer stores the address of the class instance
	// check for self assignment using address comparison
	if(this != &rhs)		
	{
		if(m == rhs.m && n == rhs.n)
		{
			if(rhs.nnz == 0)
			{
				return *this;
			}
			else if(nnz == 0)
			{
				dcsc = new Dcsc<IT,NT>(*(rhs.dcsc));
				nnz = rhs.nnz;
			}
			else
			{
				(*dcsc) += (*(rhs.dcsc));
				nnz = rhs.nnz;
			}		
		}
		else
		{
			cout<< "Not addable !"<<endl;		
		}
	}
	else
	{
		cout<< "Missing feauture (A+A): Use multiply with 2 instead !"<<endl;	
	}
	return *this;
}


/****************************************************************************/
/********************* PUBLIC MEMBER FUNCTIONS ******************************/
/****************************************************************************/

template <class IT, class NT>
void SpDCCols<IT,NT>::CreateImpl(vector<IT> & essentials)
{
	nnz = essentials[0];
	m = essentials[1];
	n = essentials[2];

	if(nnz > 0)
		dcsc = new Dcsc<IT,NT>(nnz,essentials[3]);
	else
		dcsc = NULL; 
}

template <class IT, class NT>
vector<IT> SpDCCols<IT,NT>::GetEssentials() const
{
	vector<IT> essentials(4);
	essentials[0] = nnz;
	essentials[1] = m;
	essentials[2] = n;
	essentials[3] = dcsc->nzc;
	return essentials;
}

template <class IT, class NT>
template <typename NNT>
SpDCCols<IT,NNT> SpDCCols<IT,NT>::ConvertNumericType ()
{
	Dcsc<IT,NNT> * convert = new Dcsc<IT,NNT>(dcsc->ConvertNumericType<NNT>());
	return SpDCCols<IT,NNT>(nnz, m, n, convert);
}

template <class IT, class NT>
Arr<IT,NT> SpDCCols<IT,NT>::GetArrays() const
{
	Arr<IT,NT> arr(3,1);
	arr.indarrs[0] = LocArr<IT,IT>(dcsc->cp, dcsc->nzc+1);
	arr.indarrs[1] = LocArr<IT,IT>(dcsc->jc, dcsc->nzc);
	arr.indarrs[2] = LocArr<IT,IT>(dcsc->ir, dcsc->nz);
	arr.numarrs[0] = LocArr<IT,NT>(dcsc->numx, dcsc->nz);
}

/**
  * O(nnz log(nnz)) time Transpose function
  * \remarks Performs a lexicographical sort
  * \remarks Mutator function (replaces the calling object with its transpose)
  * \remarks respects the memory pool
  */
template <class IT, class NT>
void SpDCCols<IT,NT>::Transpose()
{
	SpTuples<IT,NT> Atuples(*this);
	Atuples.SortRowBased();

	// destruction of (*this) is handled by the assignment operator
	*this = SpDCCols<IT,NT>(Atuples,true, localpool);
}

/**
  * O(nnz log(nnz)) time Transpose function
  * \remarks Performs a lexicographical sort
  * \remarks Const function (doesn't mutate the calling object)
  * \remarks respects the memory pool
  */
template <class IT, class NT>
SpDCCols<IT,NT> SpDCCols<IT,NT>::TransposeConst() const
{
	SpTuples<IT,NT> Atuples(*this);
	Atuples.SortRowBased();

	return SpDCCols<IT,NT>(Atuples,true, localpool);
}

/** 
  * Splits the matrix into two parts, simply by cutting along the columns
  * Simple algorithm that doesn't intend to split perfectly, but it should do a pretty good job
  * Practically destructs the calling object also (frees most of its memory)
  */
template <class IT, class NT>
void SpDCCols<IT,NT>::Split(SpDCCols<IT,NT> & partA, SpDCCols<IT,NT> & partB) 
{
	IT cut = n/2;
	if(cut == zero)
	{
		cout<< "Matrix is too small to be splitted" << endl;
		return;
	}

	Dcsc<IT,NT> *Adcsc, *Bdcsc;
	dcsc->Split(Adcsc, Bdcsc, cut);

	partA = SpDCCols (Adcsc->nz, m, cut, Adcsc);
	partB = SpDCCols (Bdcsc->nz, m, n-cut, Bdcsc);
	
	// handle destruction through assignment operator
	*this = SpDCCols<IT, NT>();		
}

/** 
  * Merges two matrices (cut along the columns) into 1 piece
  * Split method should have been executed on the object beforehand
 **/
template <class IT, class NT>
void SpDCCols<IT,NT>::Merge(SpDCCols<IT,NT> & A, SpDCCols<IT,NT> & B) 
{
	assert(A.m == B.m);

	Dcsc<IT,NT> * Cdcsc = new Dcsc<IT,NT>();
	Cdcsc->Merge(A.dcsc, B.dcsc, A.n);
	
	*this = SpDCCols<IT,NT> (dcsc->nz, A.m, A.n + B.n, Cdcsc);

	A = SpDCCols<IT, NT>();	
	B = SpDCCols<IT, NT>();	
}

/**
 * C += A*B' (Using OuterProduct Algorithm)
 * This version is currently limited to multiplication of matrices with the same precision 
 * (e.g. it can't multiply double-precision matrices with booleans)
 * The multiplication is on the specified semiring (passed as parameter)
 */
template <class IT, class NT>
template <class SR>
int SpDCCols<IT,NT>::PlusEq_AnXBt(const SpDCCols<IT,NT> & A, const SpDCCols<IT,NT> & B, const SR & sring)
{
	if(A.isZero() || B.isZero())
	{
		return -1;	// no need to do anything
	}
	Isect<IT> *isect1, *isect2, *itr1, *itr2, *cols, *rows;
	SpHelper::SpIntersect(A->dcsc, B->dcsc, cols, rows, isect1, isect2, itr1, itr2);
	
	IT kisect = static_cast<IT>(itr1-isect1);		// size of the intersection ((itr1-isect1) == (itr2-isect2))
	if(kisect == zero)
	{
		DeleteAll(isect1, isect2, cols, rows);
		return -1;
	}
	
	StackEntry< NT, pair<IT,IT> > * multstack;
	IT cnz = SpHelper::SpCartesian (A.dcsc, B.dcsc, sring, kisect, isect1, isect2, multstack);  
	DeleteAll(isect1, isect2, cols, rows);

	IT mdim = A.m;	
	IT ndim = B.m;		// since B has already been transposed
	if(isZero())
	{
		dcsc = new Dcsc<IT,NT>(multstack, mdim, ndim, cnz);
	}
	else
	{
		dcsc->AddAndAssign(multstack, mdim, ndim, cnz);
	}

	delete [] multstack;
	return 1;	
}

/**
 * C += A*B (Using ColByCol Algorithm)
 * This version is currently limited to multiplication of matrices with the same precision 
 * (e.g. it can't multiply double-precision matrices with booleans)
 * The multiplication is on the specified semiring (passed as parameter)
 */
template <class IT, class NT>
template <typename SR>
int SpDCCols<IT,NT>::PlusEq_AnXBn(const SpDCCols<IT,NT> & A, const SpDCCols<IT,NT> & B, const SR & sring)
{
	if(A.isZero() || B.isZero())
	{
		return -1;	// no need to do anything
	}
	StackEntry< NT, pair<IT,IT> > * multstack;
	IT cnz = SpHelper::SpColByCol (A.dcsc, B.dcsc, sring, multstack);  
	
	IT mdim = A.m;	
	IT ndim = B.n;
	if(isZero())
	{
		dcsc = new Dcsc<IT,NT>(multstack, mdim, ndim, cnz);
	}
	else
	{
		dcsc->AddAndAssign(multstack, mdim, ndim, cnz);
	}
	delete [] multstack;
	return 1;	
}


template <class IT, class NT>
template <typename SR>
int SpDCCols<IT,NT>::PlusEq_AtXBn(const SpDCCols<IT,NT> & A, const SpDCCols<IT,NT> & B, const SR & sring)
{
	cout << "PlusEq_AtXBn function has not been implemented yet !" << endl;
	return 0;
}

template <class IT, class NT>
template <typename SR>
int SpDCCols<IT,NT>::PlusEq_AtXBt(const SpDCCols<IT,NT> & A, const SpDCCols<IT,NT> & B, const SR & sring)
{
	cout << "PlusEq_AtXBt function has not been implemented yet !" << endl;
	return 0;
}


/** 
 * The almighty indexing polyalgorithm 
 * Calls different subroutines depending the sparseness of ri/ci
 */
template <class IT, class NT>
SpDCCols<IT,NT> SpDCCols<IT,NT>::operator() (const vector<IT> & ri, const vector<IT> & ci) const
{
	IT rsize = ri.size();
	IT csize = ci.size();

	if(rsize == 0 && csize == 0)
	{
		// return an m x n matrix of complete zeros
		// since we don't know whether columns or rows are indexed
		return SpDCCols<IT,NT> (zero, m, n, zero);		
	}
	else if(rsize == 1 && csize == 1)
	{
		cout << "Please use special element-wise indexing instead of vectors of length 1" << endl;
		return ((*this)(ri[0], ci[0]));
	}	
	else if(rsize == 0)
	{
		return ColIndex(ci);
	}
	else if(csize == 0)
	{
		SpDCCols<IT,NT> LeftMatrix(rsize, rsize, this->m, ri, true);
		//return LeftMatrix.template OrdColByCol<PlusTimesSRing>(*this, PlusTimesSRing());
	}
	else
	{
		SpDCCols<IT,NT> LeftMatrix(rsize, rsize, this->m, ri, true);
		SpDCCols<IT,NT> RightMatrix(csize, this->n, csize, ci, false);
		return LeftMatrix.OrdColByCol(OrdColByCol(RightMatrix,PlusTimesSRing()),PlusTimesSRing());
	}
}

template <class IT, class NT>
ofstream & SpDCCols<IT,NT>::put(ofstream & outfile) const 
{
	if(nnz == 0)
	{
		outfile << "Matrix doesn't have any nonzeros" <<endl;
		return outfile;
	}
	SpTuples<IT,NT> tuples(*this); 
	outfile << tuples << endl;
}

template<class IT, class NT>
void SpDCCols<IT,NT>::PrintInfo()
{
	cout << "m: " << m ;
	cout << ", n: " << n ;
	cout << ", nnz: "<< nnz ;
	if(dcsc != NULL)
	{
		cout << ", nzc: "<< dcsc->nzc << endl;
	}
}


/****************************************************************************/
/********************* PRIVATE CONSTRUCTORS/DESTRUCTORS *********************/
/****************************************************************************/

//! Construct SpDCCols from Dcsc
template <class IT, class NT>
SpDCCols<IT,NT>::SpDCCols(IT size, IT nRow, IT nCol, Dcsc<IT,NT> * mydcsc)
:m(nRow), n(nCol), nnz(size), localpool(NULL)
{
	if(size > 0)
		dcsc = mydcsc;
	else
		dcsc = NULL;
}

//! Create a logical matrix from (row/column) indices array, used for indexing only
template <class IT, class NT>
SpDCCols<IT,NT>::SpDCCols (IT size, IT nRow, IT nCol, const vector<IT> & indices, bool isRow)
:m(nRow), n(nCol), nnz(size), localpool(NULL)
{
	if(size > 0)
		dcsc = new Dcsc<IT,NT>(size,indices,isRow);
	else
		dcsc = NULL; 
}


/****************************************************************************/
/************************* PRIVATE MEMBER FUNCTIONS *************************/
/****************************************************************************/

template <class IT, class NT>
inline void SpDCCols<IT,NT>::CopyDcsc(Dcsc<IT,NT> * source)
{
	// source dcsc will be NULL if number of nonzeros is zero 
	if(source != NULL)	
		dcsc = new Dcsc<IT,NT>(*source);
	else
		dcsc = NULL;
}

/**
 * \return An indexed SpDCCols object without using multiplication 
 * \pre ci is sorted and is not completely empty.
 * \remarks it is OK for some indices ci[i] to be empty in the indexed SpDCCols matrix 
 *	[i.e. in the output, nzc does not need to be equal to n]
 */
template <class IT, class NT>
SpDCCols<IT,NT> SpDCCols<IT,NT>::ColIndex(const vector<IT> & ci)
{
	IT csize = ci.size();
	if(nnz == 0)	// nothing to index
	{
		return SpDCCols<IT,NT>(zero, m, csize, zero);	
	}
	else if(ci.empty())
	{
		return SpDCCols<IT,NT>(zero, m,zero, zero);
	}

	// First pass for estimation
	IT estsize = zero;
	IT estnzc = zero;
	for(IT i=0, j=0;  i< dcsc->nzc && j < csize;)
	{
		if((dcsc->jc)[i] < ci[j])
		{
			++i;
		}
		else if ((dcsc->jc)[i] > ci[j])
		{
			++j;
		}
		else
		{
			estsize +=  (dcsc->cp)[i+1] - (dcsc->cp)[i];
			++estnzc;
			++i;
			++j;
		}
	}
	
	SpDCCol<IT,NT> SubA(estsize, m, csize, estnzc);	
	if(estnzc == zero)
	{
		return SubA;		// no need to run the second pass
	}
	SubA->dcsc->cp[0] = 0;
	IT cnzc = 0;
	IT cnz = 0;
	for(IT i=0, j=0;  i < dcsc->nzc && j < csize;)
	{
		if((dcsc->jc)[i] < ci[j])
		{
			++i;
		}
		else if ((dcsc->jc)[i] > ci[j])		// an empty column for the output
		{
			++j;
		}
		else
		{
			IT columncount = (dcsc->cp)[i+1] - (dcsc->cp)[i];
			SubA->dcsc->jc[cnzc++] = j;
			SubA->dcsc->cp[cnzc] = SubA->dcsc->cp[cnzc-1] + columncount;
			copy(dcsc->ir + dcsc->cp[i], dcsc->ir + dcsc->cp[i+1], SubA->dcsc->ir + cnz);
			copy(dcsc->numx + dcsc->cp[i], dcsc->numx + dcsc->cp[i+1], SubA->dcsc->numx + cnz);
			cnz += columncount;
			++i;
			++j;
		}
	}
	return SubA;
}

template <class IT, class NT>
template <typename NTR, typename SR>
SpDCCols< IT, typename promote_trait<NT,NTR>::T_promote > SpDCCols<IT,NT>::OrdOutProdMult(const SpDCCols<IT,NTR> & rhs, const SR & sring) const
{
	typedef typename promote_trait<NT,NTR>::T_promote T_promote;  

	if(isZero() || rhs.isZero())
	{
		return SpDCCols< IT, T_promote > (zero, m, rhs.n, zero);		// return an empty matrix	
	}
	SpDCCols<IT,NTR> Btrans = rhs.TransposeConst();

	Isect<IT> *isect1, *isect2, *itr1, *itr2, *cols, *rows;
	SpHelper::SpIntersect(dcsc, Btrans.dcsc, cols, rows, isect1, isect2, itr1, itr2);
	
	IT kisect = static_cast<IT>(itr1-isect1);		// size of the intersection ((itr1-isect1) == (itr2-isect2))
	if(kisect == zero)
	{
		DeleteAll(isect1, isect2, cols, rows);
		return SpDCCols< IT, T_promote > (zero, m, rhs.n, zero);	
	}
	StackEntry< T_promote, pair<IT,IT> > * multstack;
	IT cnz = SpHelper::SpCartesian (dcsc, Btrans.dcsc, sring, kisect, isect1, isect2, multstack);  
	DeleteAll(isect1, isect2, cols, rows);

	Dcsc< IT, T_promote > * mydcsc = new Dcsc< IT,T_promote >(multstack, m, rhs.n, cnz);
	return SpDCCols< IT, T_promote > (cnz, m, rhs.n, mydcsc);
}


template <class IT, class NT>
template <typename NTR, typename SR>
SpDCCols< IT, typename promote_trait<NT,NTR>::T_promote > SpDCCols<IT,NT>::OrdColByCol(const SpDCCols<IT,NTR> & rhs, const SR & sring) const
{
	typedef typename promote_trait<NT,NTR>::T_promote T_promote;  

	if(isZero() || rhs.isZero())
	{
		return SpDCCols<IT, T_promote> (zero, m, rhs.n, zero);		// return an empty matrix	
	}
	StackEntry< T_promote, pair<IT,IT> > * multstack;
	IT cnz = SpHelper::SpColByCol (dcsc, rhs.dcsc, sring, multstack);  
	
	Dcsc< IT,T_promote > * mydcsc = new Dcsc< IT,T_promote > (multstack, m, rhs.n, cnz);
	return SpDCCols< IT,T_promote > (cnz, m, rhs.n, mydcsc);	
}

/****************************************************************************/
/**************************** FRIEND FUNCTIONS ******************************/
/****************************************************************************/

/**
 * SpTuples(A*B') (Using OuterProduct Algorithm)
 * Returns the tuples for efficient merging later
 * Support mixed precision multiplication
 * The multiplication is on the specified semiring (passed as parameter)
 */
template<class IU, class NU1, class NU2, class SR>
SpTuples<IU, promote_trait<NU1,NU2>::T_promote> Tuples_AnXBt 
					(const SpDCCols<IU, NU1> & A, 
					 const SpDCCols<IU, NU2> & B, 
					 const SR & sring)
{
	IT mdim = A.m;	
	IT ndim = B.m;	// B is already transposed

	if(A.isZero() || B.isZero())
	{
		SpTuples<IU, promote_trait<NU1,NU2>::T_promote>(zero, mdim, ndim);	// just return an empty matrix
	}
	Isect<IT> *isect1, *isect2, *itr1, *itr2, *cols, *rows;
	SpHelper::SpIntersect(A->dcsc, B->dcsc, cols, rows, isect1, isect2, itr1, itr2);
	
	IT kisect = static_cast<IT>(itr1-isect1);		// size of the intersection ((itr1-isect1) == (itr2-isect2))
	if(kisect == zero)
	{
		DeleteAll(isect1, isect2, cols, rows);
		return SpTuples<IU, promote_trait<NU1,NU2>::T_promote>(zero, mdim, ndim);
	}
	
	StackEntry< promote_trait<NT1,NT2>::T_promote, pair<IT,IT> > * multstack;
	IT cnz = SpHelper::SpCartesian (A.dcsc, B.dcsc, sring, kisect, isect1, isect2, multstack);  
	DeleteAll(isect1, isect2, cols, rows);

	return SpTuples<IU, promote_trait<NU1,NU2>::T_promote> (cnz, mdim, ndim, multstack);
}

/**
 * SpTuples(A*B) (Using ColByCol Algorithm)
 * Returns the tuples for efficient merging later
 * Support mixed precision multiplication
 * The multiplication is on the specified semiring (passed as parameter)
 */
template<class IU, class NU1, class NU2, class SR>
SpTuples<IU, promote_trait<NU1,NU2>::T_promote> Tuples_AnXBn 
					(const SpDCCols<IU, NU1> & A, 
					 const SpDCCols<IU, NU2> & B, 
					 const SR & sring)
{
	IT mdim = A.m;	
	IT ndim = B.n;	
	if(A.isZero() || B.isZero())
	{
		SpTuples<IU, promote_trait<NU1,NU2>::T_promote>(zero, mdim, ndim);
	}
	StackEntry< promote_trait<NT1,NT2>::T_promote, pair<IT,IT> > * multstack;
	IT cnz = SpHelper::SpColByCol (A.dcsc, B.dcsc, sring, multstack);  
	
	return SpTuples<IU, promote_trait<NU1,NU2>::T_promote> (cnz, mdim, ndim, multstack);
}


template<class IU, class NU1, class NU2, class SR>
SpTuples<IU, promote_trait<NU1,NU2>::T_promote> Tuples_AtXBt 
					(const SpDCCols<IU, NU1> & A, 
					 const SpDCCols<IU, NU2> & B, 
					 const SR & sring)
{
	IT mdim = A.n;	
	IT ndim = B.m;	
	cout << "Tuples_AtXBt function has not been implemented yet !" << endl;
		
	return SpTuples<IU, promote_trait<NU1,NU2>::T_promote> (zero, mdim, ndim);
}

template<class IU, class NU1, class NU2, class SR>
SpTuples<IU, promote_trait<NU1,NU2>::T_promote> Tuples_AtXBn 
					(const SpDCCols<IU, NU1> & A, 
					 const SpDCCols<IU, NU2> & B, 
					 const SR & sring)
{
	IT mdim = A.n;	
	IT ndim = B.n;	
	cout << "Tuples_AtXBn function has not been implemented yet !" << endl;
		
	return SpTuples<IU, promote_trait<NU1,NU2>::T_promote> (zero, mdim, ndim);
}



