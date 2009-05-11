/****************************************************************/
/* Sequential and Parallel Sparse Matrix Multiplication Library */
/* version 2.3 --------------------------------------------------/
/* date: 01/18/2009 ---------------------------------------------/
/* author: Aydin Buluc (aydin@cs.ucsb.edu) ----------------------/
/****************************************************************/


#include "dcsc.h"
#include <algorithm>
#include <functional>
#include <iostream>
#include <ext/numeric>

using namespace std;

template <class IT, class NT>
Dcsc<IT,NT>::Dcsc ():nz(0), nzc(0),pool(NULL), cp(NULL), jc(NULL), ir(NULL), numx(NULL){}

template <class IT, class NT>
Dcsc<IT,NT>::Dcsc (IT nnz, IT nzcol): nz(nnz),nzc(nzcol),pool(NULL)
{
	assert (nz != 0);
	size_t sit = sizeof(IT);
	
	cp = (IT *) mallocarray ( (nzc+1)*sit ); 
	jc  = (IT *) mallocarray ( nzc*sit ); 	
	ir  = (IT *) mallocarray ( nz*sit ); 
	numx= (NT *) mallocarray ( nz*sizeof(NT) ); 
}

/** 
 * Constructor that is used when the memory for arrays are already allocated by pinning
 */
template <class IT, class NT>
Dcsc<IT,NT>::Dcsc (IT nnz, IT nzcol, MemoryPool * mpool): nz(nnz),nzc(nzcol), pool(mpool)
{
	assert (nz != 0);
	size_t sit = sizeof(IT);
	
	cp = (IT *) mallocarray ( (nzc+1)*sit); 
	jc  = (IT *) mallocarray ( nzc*sit); 	
	ir  = (IT *) mallocarray ( nz*sit); 
	numx= (NT *) mallocarray ( nz*sizeof(NT)); 
}

//! GetIndices helper function for StackEntry arrays
template <class IT, class NT>
inline void Dcsc<IT,NT>::getindices (StackEntry<NT, pair<IT,IT> > * multstack, IT & rindex, IT & cindex, IT & j, IT nnz)
{
	if(j<nnz)
	{
		cindex = multstack[j].key.first;
		rindex = multstack[j].key.second;
	}
	else
	{
		rindex = numeric_limits<IT>::max();
		cindex = numeric_limits<IT>::max();
	}
	++j;
}

template <class IT, class NT>
void * Dcsc<IT,NT>::mallocarray (size_t size)
{
	void * addr;
	if(pool == NULL)
	{
		addr = malloc( size + ALIGN );
	}
	else
	{
		addr = pool->alloc( size + ALIGN );		
	}
	if(addr == NULL)
	{
		cerr<< "Mallocarray() failed" << endl;
		return NULL;
	}

	*((MemoryPool **) addr) = pool;			// Store the pointer to the memory pool (to be able to deallocate later)
	return (void *) (((char *) addr) + ALIGN);	// The returned raw memory excludes the memory pool pointer
}

//! Frees the memory and assigns the pointer to NULL 
template <class IT, class NT>
void Dcsc<IT, NT>::deletearray(void * array, size_t size)
{
	
	if(size != 0 && array != NULL)
	{
		array = (void *) ((char*)array - ALIGN);
		MemoryPool * pool = * ((MemoryPool **) array);
		if(pool == NULL)
			free(array);
		else
			pool->dealloc(array, size + ALIGN);

		array = NULL;
	}
}

	
template <class IT, class NT>
Dcsc<IT,NT> & Dcsc<IT,NT>::AddAndAssign (StackEntry<NT, pair<IT,IT> > * multstack, IT mdim, IT ndim, IT nnz)
{
	if(nnz == 0)	return *this;
		
	IT estnzc = nzc + nnz;
	IT estnz  = nz + nnz;
	Dcsc<IT,NT> temp(estnz, estnzc);

	IT curnzc = 0;		// number of nonzero columns constructed so far
	IT curnz = 0;
	IT i = 0;
	IT j = 0;
	IT rindex, cindex;
	getindices(multstack, rindex, cindex,j,nnz);

	temp.cp[0] = 0;
	while(i< nzc && cindex < numeric_limits<IT>::max())	// i runs over columns of "this",  j runs over all the nonzeros of "multstack"
	{
		if(jc[i] > cindex)
		{
			IT columncount = 0;
			temp.jc[curnzc++] = cindex;
			do
			{
				temp.ir[curnz] 		= rindex;
				temp.numx[curnz++] 	= multstack[j-1].value;

				getindices(multstack, rindex, cindex,j,nnz);
				++columncount;
			}
			while(temp.jc[curnzc-1] == cindex);	// loop until cindex changes

			temp.cp[curnzc] = temp.cp[curnzc-1] + columncount;
		}
		else if(jc[i] < cindex)
		{
			temp.jc[curnzc++] = jc[i++];
			for(IT k = cp[i-1]; k< cp[i]; ++k)
			{
				temp.ir[curnz] 		= ir[k];
				temp.numx[curnz++] 	= numx[k];
			}
			temp.cp[curnzc] = temp.cp[curnzc-1] + (cp[i] - cp[i-1]);
		}
		else	// they are equal, merge the column
		{
			temp.jc[curnzc++] = jc[i];
			ITYPE ii = cp[i];
			ITYPE prevnz = curnz;		
			while (ii < cp[i+1] && cindex == jc[i])	// cindex would be MAX if multstack is deplated
			{
				if (ir[ii] < rindex)
				{
					temp.ir[curnz] = ir[ii];
					temp.numx[curnz++] = numx[ii++];
				}
				else if (ir[ii] > rindex)
				{
					temp.ir[curnz] = rindex;
					temp.numx[curnz++] = multstack[j-1].value;

					getindices(multstack, rindex, cindex,j,nnz);
				}
				else
				{
					temp.ir[curnz] = ir[ii];
					temp.numx[curnz++] = numx[ii++] + multstack[j-1].value;

					getindices(multstack, rindex, cindex,j,nnz);
				}
			}
			while (ii < cp[i+1])
			{
				temp.ir[curnz] = ir[ii];
				temp.numx[curnz++] = numx[ii++];
			}
			while (cindex == jc[i])
			{
				temp.ir[curnz] = rindex;
				temp.numx[curnz++] = multstack[j-1].value;

				getindices(multstack, rindex, cindex,j,nnz);
			}
			temp.cp[curnzc] = temp.cp[curnzc-1] + curnz-prevnz;
			++i;
		}
	}
	while(i< nzc)
	{
		temp.jc[curnzc++] = jc[i++];
		for(IT k = cp[i-1]; k< cp[i]; ++k)
		{
			temp.ir[curnz] 		= ir[k];
			temp.numx[curnz++] 	= numx[k];
		}
		temp.cp[curnzc] = temp.cp[curnzc-1] + (cp[i] - cp[i-1]);
	}
	while(cindex < numeric_limits<IT>::max())
	{
		IT columncount = 0;
		temp.jc[curnzc++] = cindex;
		do
		{
			temp.ir[curnz] 		= rindex;
			temp.numx[curnz++] 	= multstack[j-1].value;

			getindices(multstack, rindex, cindex,j,nnz);
			++columncount;
		}
		while(temp.jc[curnzc-1] == cindex);	// loop until cindex changes

		temp.cp[curnzc] = temp.cp[curnzc-1] + columncount;
	}
	temp.Resize(curnzc, curnz);
	*this = temp;
	return *this;
}


/**
  * Creates DCSC structure from an array of StackEntry's
  * \remark Complexity: O(nnz)
  */
template <class IT, class NT>
Dcsc<IT,NT>::Dcsc (StackEntry<NT, pair<IT,IT> > * multstack, IT mdim, IT ndim, IT nnz): nz(nnz), pool(NULL)
{
	assert(nz != 0 );
	size_t sit = sizeof(IT);
	
	cp = (IT *) mallocarray ( (nz+1)*sit ); 	// to be shrinked
	jc  = (IT *) mallocarray ( nz*sit ); 		// to be shrinked
	ir  = (IT *) mallocarray ( nz*sit ); 
	numx= (NT *) mallocarray ( nz*sizeof(NT) ); 

	IT curnzc = 0;				// number of nonzero columns constructed so far
	IT cindex = multstack[0].key.first;
	IT rindex = multstack[0].key.second;

	ir[0]	= rindex;
	numx[0] = multstack[0].value;
	jc[curnzc] = cindex;
	cp[curnzc] = zero; 
	++curnzc;

	for(IT i=1; i<nz; ++i)
	{
		cindex = multstack[i].key.first;
		rindex = multstack[i].key.second;

		ir[i]	= rindex;
		numx[i] = multstack[i].value;
		if(cindex != jc[curnzc-1])
		{
			jc[curnzc] = cindex;
			cp[curnzc++] = i;
		}
	}
	cp[curnzc] = nz;

	Resize(curnzc, nz);	// only shrink cp & jc arrays
}

/**
  * Create a logical matrix from (row/column) indices array
  * \remark This function should only be used for indexing 
  * \remark For these temporary matrices nz = nzc (which are both equal to nnz)
  */
template <class IT, class NT>
Dcsc<IT,NT>::Dcsc (IT nnz, const vector<IT> & indices, bool isRow): nz(nnz),nzc(nnz),pool(NULL)
{
	assert((nnz != 0) && (indices.size() == nnz));
	size_t sit = sizeof(IT);
	
	cp = (IT *) mallocarray ( (nnz+1)*sit ); 	
	jc  = (IT *) mallocarray ( nnz*sit ); 	
	ir  = (IT *) mallocarray ( nnz*sit ); 
	numx= (NT *) mallocarray ( nnz*sizeof(NT) ); 

	SpHelper::iota(cp, cp+nnz+1, 0);  // insert sequential values {0,1,2,..}
	fill_n(numx, nnz, static_cast<NT>(1));
	
	if(isRow)
	{
		SpHelper::iota(ir, ir+nnz, 0);	
		std::copy (indices.begin(), indices.end(), jc);
	}
	else
	{
		SpHelper::iota(jc, jc+nnz, 0);
		std::copy (indices.begin(), indices.end(), ir);	
	}
}


template <class IT, class NT>
template <typename NNT>
Dcsc<IT,NNT> Dcsc<IT,NT>::ConvertNumericType ()
{
	Dcsc<IT,NNT> convert(nz, nzc);	
	
	for(IT i=0; i< nz; ++i)
	{		
		convert.numx[i] = static_cast<NNT>(convert.rhs.numx[i]);		
		convert.ir[i] = convert.rhs.ir[i];
	}
	memcpy(convert.jc, jc, nzc * sizeof(IT));
	memcpy(convert.cp, cp, (nzc+1) * sizeof(IT));
	return convert;
}

/**
  * Copy constructor that respects the memory pool
  */
template <class IT, class NT>
Dcsc<IT,NT>::Dcsc (const Dcsc<IT,NT> & rhs): nz(rhs.nz), nzc(rhs.nzc), pool(rhs.pool)
{
	size_t sit = sizeof(IT);
	if(nz > 0)
	{
		numx= (NT *) mallocarray ( nz*sizeof(NT) ); 
		ir  = (IT *) mallocarray ( nz*sit ); 
		memcpy(numx, rhs.numx, nz*sizeof(NT));
		memcpy(ir, rhs.ir, nz*sit);
	}
	else
	{
		numx = NULL;
		ir = NULL;
	}
	if(nzc > 0)
	{
		jc  = (IT *) mallocarray ( nzc*sit ); 
		cp = (IT *) mallocarray ( (nzc+1)*sit ); 		
		memcpy(jc, rhs.jc, nzc*sit);
		memcpy(cp, rhs.cp, (nzc+1)*sit);
	}
	else
	{
		jc = NULL;
		cp = NULL;
	}
}

/**
  * Assignment operator (called on an existing object)
  * \attention The memory pool of the lvalue is replaced by the memory pool of rvalue
  * If A = B where B uses pinnedPool and A uses NULL before the operation,
  * then after the operation A now uses pinnedPool too
  */
template <class IT, class NT>
Dcsc<IT,NT> & Dcsc<IT,NT>::operator =(const Dcsc<IT,NT> & rhs)
{
	if(this != &rhs)		
	{
		size_t sit = sizeof(IT);

		// make empty first !
		if(nz > 0)
		{
			deletearray(numx, sizeof(NT) * nz);
			deletearray(ir, sit * nz);
		}
		if(nzc > 0)
		{
			deletearray(jc, sit * nzc);
			deletearray(cp, sit * (nzc+1));
		}
		pool = rhs.pool;
		nz = rhs.nz;
		nzc = rhs.nzc;
		if(nz > 0)
		{
			numx= (NT *) mallocarray ( nz*sizeof(NT) ); 
			ir  = (IT *) mallocarray ( nz*sit ); 
			memcpy(numx, rhs.numx, nz*sizeof(NT));
			memcpy(ir, rhs.ir, nz*sit);	
		}
		else
		{
			numx = NULL;
			ir = NULL;
		}
		if(nzc > 0)
		{
			jc  = (IT *) mallocarray ( nzc*sit ); 
			cp = (IT *) mallocarray ( (nzc+1)*sit ); 		
			memcpy(jc, rhs.jc, nzc*sit);
			memcpy(cp, rhs.cp, (nzc+1)*sit);
		}
		else
		{
			jc = NULL;
			cp = NULL;
		}
	}
	return *this;
}

/**
  * \attention The memory pool of the lvalue is preserved
  * If A += B where B uses pinnedPool and A uses NULL before the operation,
  * then after the operation A still uses NULL memory (old school 'malloc')
  */
template <class IT, class NT>
Dcsc<IT, NT> & Dcsc<IT,NT>::operator+=(const Dcsc<IT,NT> & rhs)	// add and assign operator
{
	IT estnzc = nzc + rhs.nzc;
	IT estnz  = nz + rhs.nz;
	Dcsc<IT,NT> temp(estnz, estnzc);

	IT curnzc = 0;
	IT curnz = 0;
	IT i = 0;
	IT j = 0;
	temp.cp[0] = zero;
	while(i< nzc && j<rhs.nzc)
	{
		if(jc[i] > rhs.jc[j])
		{
			temp.jc[curnzc++] = rhs.jc[j++];
			for(IT k = rhs.cp[j-1]; k< rhs.cp[j]; ++k)
			{
				temp.ir[curnz] 		= rhs.ir[k];
				temp.numx[curnz++] 	= rhs.numx[k];
			}
			temp.cp[curnzc] = temp.cp[curnzc-1] + (rhs.cp[j] - rhs.cp[j-1]);
		}
		else if(jc[i] < rhs.jc[j])
		{
			temp.jc[curnzc++] = jc[i++];
			for(IT k = cp[i-1]; k< cp[i]; k++)
			{
				temp.ir[curnz] 		= ir[k];
				temp.numx[curnz++] 	= numx[k];
			}
			temp.cp[curnzc] = temp.cp[curnzc-1] + (cp[i] - cp[i-1]);
		}
		else
		{
			temp.jc[curnzc++] = jc[i];
			IT ii = cp[i];
			IT jj = rhs.cp[j];
			IT prevnz = curnz;		
			while (ii < cp[i+1] && jj < rhs.cp[j+1])
			{
				if (ir[ii] < rhs.ir[jj])
				{
					temp.ir[curnz] = ir[ii];
					temp.numx[curnz++] = numx[ii++];
				}
				else if (ir[ii] > rhs.ir[jj])
				{
					temp.ir[curnz] = rhs.ir[jj];
					temp.numx[curnz++] = rhs.numx[jj++];
				}
				else
				{
					temp.ir[curnz] = ir[ii];
					temp.numx[curnz++] = numx[ii++] + rhs.numx[jj++];	// might include zeros
				}
			}
			while (ii < cp[i+1])
			{
				temp.ir[curnz] = ir[ii];
				temp.numx[curnz++] = numx[ii++];
			}
			while (jj < rhs.cp[j+1])
			{
				temp.ir[curnz] = rhs.ir[jj];
				temp.numx[curnz++] = rhs.numx[jj++];
			}
			temp.cp[curnzc] = temp.cp[curnzc-1] + curnz-prevnz;
			++i;
			++j;
		}
	}
	while(i< nzc)
	{
		temp.jc[curnzc++] = jc[i++];
		for(IT k = cp[i-1]; k< cp[i]; ++k)
		{
			temp.ir[curnz] 	= ir[k];
			temp.numx[curnz++] = numx[k];
		}
		temp.cp[curnzc] = temp.cp[curnzc-1] + (cp[i] - cp[i-1]);
	}
	while(j < rhs.nzc)
	{
		temp.jc[curnzc++] = rhs.jc[j++];
		for(IT k = rhs.cp[j-1]; k< rhs.cp[j]; ++k)
		{
			temp.ir[curnz] 	= rhs.ir[k];
			temp.numx[curnz++] 	= rhs.numx[k];
		}
		temp.cp[curnzc] = temp.cp[curnzc-1] + (rhs.cp[j] - rhs.cp[j-1]);
	}
	temp.Resize(curnzc, curnz);
	*this = temp;
	return *this;
}
	

/** 
  * Construct an index array called aux
  * Return the size of the contructed array
  * Complexity O(nzc)
 **/ 
template <class IT, class NT>
IT Dcsc<IT,NT>::ConstructAux(IT ndim, IT * & aux)
{
	float cf  = static_cast<float>(ndim+1) / static_cast<float>(nzc);
	IT colchunks = static_cast<IT> ( ceil( static_cast<float>(ndim+1) / ceil(cf)) );

	aux  = (IT *) mallocarray ( (colchunks+1)*sizeof(IT) ); 

	IT chunksize	= static_cast<IT>(ceil(cf));
	IT reg		= zero;
	IT curchunk	= zero;
	aux[curchunk++] = 0;
	for(IT i = 0; i< nzc; ++i)
	{
		if(jc[i] >= curchunk * chunksize)		// beginning of the next chunk
		{
			while(jc[i] >= curchunk * chunksize)	// consider any empty chunks
			{
				aux[curchunk++] = reg;
			}
		}
		reg = i+1;
	}
	while(curchunk <= colchunks)
	{
		aux[curchunk++] = reg;
	}
	return colchunks;
}

/**
  * Resizes cp & jc arrays to nzcnew, ir & numx arrays to nznew
  * Zero overhead in case sizes stay the same 
 **/ 
template <class IT, class NT>
void Dcsc<IT,NT>::Resize(IT nzcnew, IT nznew)
{
	size_t sit = sizeof(IT);
	if(nzcnew == 0)
	{
		deletearray(jc, sit * nzc);
		deletearray(cp, sit * (nzc+1));
		nzc = 0;
	}
	if(nznew == 0)
	{
		deletearray(ir, sit * nz);
		deletearray(numx, sizeof(NT) * nz);
		nz = 0;
	}
	if ( nzcnew == 0 && nznew == 0)
	{
		return;	
	}
	if (nzcnew != nzc)	
	{
		IT * tmpcp = cp; 
		IT * tmpjc = jc;
		
		cp = (IT *) mallocarray ( (nzcnew+1)*sit ); 
		jc = (IT *) mallocarray (  nzcnew * sit ); 	

		if(nzcnew > nzc)	// Grow it (copy all of the old elements)
		{
			memcpy(cp, tmpcp, (nzc+1)*sit);
			memcpy(jc, tmpjc, nzc*sit);		
		}
		else		// Shrink it (copy only a portion of the old elements)
		{
			memcpy(cp, tmpcp, (nzcnew+1)*sit);
			memcpy(jc, tmpjc, nzcnew*sit);	
		}
		deletearray(tmpcp, sit * (nzc+1));	// delete the memory pointed by previous pointers
		deletearray(tmpjc, sit * nzc);
		nzc = nzcnew;
	}
	if (nznew != nz)
	{	
		NT * tmpnumx = numx; 
		IT * tmpir = ir;
	
		numx = (NT *) mallocarray ( nznew * sizeof(NT) ); 
		ir = (IT *) mallocarray (  nznew * sit ); 

		if(nznew > nz)	// Grow it (copy all of the old elements)
		{
			memcpy(numx, tmpnumx, nz*sizeof(NT));
			memcpy(ir, tmpir, nz*sit);	
		}
		else	// Shrink it (copy only a portion of the old elements)
		{
			memcpy(numx, tmpnumx, nznew*sizeof(NT));
			memcpy(ir, tmpir, nznew*sit);	
		}
		deletearray(tmpnumx, nz * sizeof(NT));	// delete the memory pointed by previous pointers
		deletearray(tmpir, nz * sit);
		nz = nznew;
	}
}

/**
  * The first part of the indexing algorithm described in the IPDPS'08 paper
  * @param[IT] colind {Column index to search}
  * Find the column with colind. If it exists, return the position of it. 
  * It it doesn't exist, return value is undefined (implementation specific).
 **/
template<class IT, class NT>
IT Dcsc<IT,NT>::AuxIndex(IT colind, bool & found, IT * aux, IT csize)
{
	IT base = static_cast<IT>(floor((float) (colind/csize)));
	IT start = aux[base];
	IT end = aux[base+1];

	IT * itr = find(jc + start, jc + end, colind);
	
	found = (itr != jc + end);
	return (itr-jc);
}

/**
 ** Split along the cut (a column index)
 ** Should work even when one of the splits have no nonzeros at all
 **/
template<class IT, class NT>
void Dcsc<IT,NT>::Split(Dcsc<IT,NT> * & A, Dcsc<IT,NT> * & B, IT cut)
{
	IT * itr = lower_bound(jc, jc+nzc, cut);
	IT pos = itr - jc;
	
	A = new Dcsc<IT,NT>(cp[pos], pos);
	B = new Dcsc<IT,NT>(nz-cp[pos], nzc-pos);
	
	memcpy(A->jc, jc, pos * sizeof(IT));
	memcpy(A->cp, cp, (pos+1) * sizeof(IT));
	memcpy(A->ir, ir, cp[pos] * sizeof(IT));
	memcpy(A->numx, numx, cp[pos] * sizeof(NT));
	
	memcpy(B->jc, jc+pos, (nzc-pos) * sizeof(IT));
	transform(B->jc, B->jc + (nzc-pos), B->jc, bind2nd(minus<IT>(), cut);
	memcpy(B->cp, cp+pos, (nzc-pos+1) * sizeof(IT));
	transform(B->cp, B->cp + (nzc-pos+1), B->cp, bind2nd(minus<IT>(), cp[pos]);
	memcpy(B->ir, ir + cp[pos], (nz- cp[pos]) * sizeof(IT)); 
	memcpy(B->numx, numx + cp[pos], (nz- cp[pos]) * sizeof(NT)); 
}

template<class IT, class NT>
void Dcsc<IT,NT>::Merge(const Dcsc<IT,NT> * A, const Dcsc<IT,NT> * B, IT cut)
{
	IT cnz = A->nz + B->nz;
	IT cnzc =  A->nzc + B->nzc;
	*this = Dcsc<IT,NT>(cnz, cnzc);		// safe, because "this" can not be NULL inside a member function

	memcpy(jc, A->jc, A->nzc * sizeof(IT));
	memcpy(jc + A->nzc, B->jc, B->nzc * sizeof(IT));
	transform(jc + A->nzc, jc + cnzc, jc + A->nzc, bind2nd(plus<IT>(), cut));

	memcpy(cp, A->cp, A->nzc * sizeof(IT));
	memcpy(cp + A->nzc, B->cp, (B->nzc+1) * sizeof(IT));
	transform(cp + A->nzc, cp+cnzc+1, cp + A->nzc, bind2nd(plus<IT>(), A->cp[A->nzc]));

	memcpy(ir, A->ir, A->nz * sizeof(IT));
	memcpy(ir + A->nz, B->ir, B->nz * sizeof(IT));

	memcpy(numx, A->numx, A->nz * sizeof(NT));
	memcpy(numx + A->nz, B->numx, B->nz * sizeof(NT));
}

template<class IT, class NT>
void Dcsc<IT,NT>::fillcolinds(const vector<IT> & colnums, vector< pair<IT,IT> > & colinds, IT n) const
{
	IT nind = colnums.size();			// number of columns of A that contributes to C(:,i)
	if ( (nzc / nind) < THRESHOLD) 			// use scanning indexing
	{
		IT mink = min(nzc, nind);
		pair<IT,IT> * isect = new pair<IT,IT>[mink];
		pair<IT,IT> * range1 = new pair<IT,IT>[nzc];
		pair<IT,IT> * range2 = new pair<IT,IT>[nind];
		
		for(IT i=0; i < nzc; ++i)
		{
			range1[i] = make_pair(jc[i], i);	// get the actual nonzero value and the index to the ith nonzero
		}
		for(IT i=0; i < nind; ++i)
		{
			range2[i] = make_pair(colnums[i], 0);	// second is dummy as all the intersecting elements are copied from the first range
		}

		pair<IT,IT> * itr = set_intersection(range1, range1 + nzc, range2, range2+nind, isect, SpHelper::first_compare);
		// isect now can iterate on a subset of the elements of range1
		// meaning that the intersection can be accessed directly by isect[i] instead of range1[isect[i]]
		// this is because the intersecting elements are COPIED to the output range "isect"

		IT kisect = static_cast<IT>(itr-isect);		// size of the intersection 
		for(IT j=0, i =0; j< nind; ++j)
		{
			// the elements represented by jc[isect[i]] are a subset of the elements represented by colnums[j]
			if( i == kisect || isect[i].first != colnums[j])
			{
				// not found, signal by setting first = second
				colinds[j].first = 0;
				colinds[j].second = 0;	
			}
			else	// i < kisect && dcsc->jc[isect[i]] == colnums[j]
			{
				IT p = isect[i++].second;
				colinds[j].first = cp[p];
				colinds[j].second = cp[p+1];
			}
		}
		DeleteAll(isect, range1, range2);
	}
	else	 	// use aux based indexing
	{
		float cf  = static_cast<float>(n+1) / static_cast<float>(nzc);
		IT csize = static_cast<IT>(ceil(cf));	// chunk size
		
		IT * aux;
		IT auxsize = ConstructAux(n, aux);

		bool found;
		for(ITYPE j =0; j< nind; ++j)
		{
			IT pos = AuxIndex(colnums[i], found, aux, csize);
			if(found)
			{
				colinds[j].first = cp[pos];
				colinds[j].second = cp[pos+1];
			}
			else 	// not found, signal by setting first = second
			{
				colinds[j].first = 0;
				colinds[j].second = 0;
			}
		}
	}
}


template <class IT, class NT>
Dcsc<IT,NT>::~Dcsc()
{
	size_t sit = sizeof(IT);
	if(nz > 0)			// dcsc may be empty
	{
		deletearray(numx, nz * sizeof(NT));
		deletearray(ir, nz * sit);
	}
	if(nzc > 0)
	{
		deletearray(jc, nzc * sit);
		deletearray(cp, (nzc+1) * sit);
	}
}

