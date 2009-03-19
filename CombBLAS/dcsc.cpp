/****************************************************************/
/* Sequential and Parallel Sparse Matrix Multiplication Library */
/* version 2.3 --------------------------------------------------/
/* date: 01/18/2009 ---------------------------------------------/
/* author: Aydin Buluc (aydin@cs.ucsb.edu) ----------------------/
/****************************************************************/


#include "dcsc.h"

template <class IT, class NT>
Dcsc<IT,NT>::Dcsc ():nz(0), nzc(0), cf(0.0), colchunks(0),pool(NULL)
{
	aux = NULL; 
	mas = NULL; 
	jc = NULL; 
	ir = NULL; 
	numx = NULL;
}

template <class IT, class NT>
Dcsc<IT,NT>::Dcsc (IT nnz, IT nzcol): nz(nnz),nzc(nzcol),cf(0.0), colchunks(0), pool(NULL)
{
	assert (nz != 0);
	size_t sit = sizeof(IT);
	
	mas = (IT *) mallocarray ( (nzc+1)*sit ); 
	jc  = (IT *) mallocarray ( nzc*sit ); 	
	ir  = (IT *) mallocarray ( nz*sit ); 
	numx= (NT *) mallocarray ( nz*sizeof(NT) ); 
	aux = NULL;
}

/** 
 * Constructor that is used when the memory for arrays are already allocated by pinning
 * @remark Aux is left NULL, to be contructed by ConstructAux whenever necessary
 */
template <class IT, class NT>
Dcsc<IT,NT>::Dcsc (IT nnz, IT nzcol, MemoryPool * mpool): nz(nnz),nzc(nzcol),cf(0.0), colchunks(0), pool(mpool)
{
	assert (nz != 0);
	size_t sit = sizeof(IT);
	
	mas = (IT *) mallocarray ( (nzc+1)*sit); 
	jc  = (IT *) mallocarray ( nzc*sit); 	
	ir  = (IT *) mallocarray ( nz*sit); 
	numx= (NT *) mallocarray ( nz*sizeof(NT)); 
	aux = NULL;
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

	temp.mas[0] = 0;
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

			temp.mas[curnzc] = temp.mas[curnzc-1] + columncount;
		}
		else if(jc[i] < cindex)
		{
			temp.jc[curnzc++] = jc[i++];
			for(IT k = mas[i-1]; k< mas[i]; ++k)
			{
				temp.ir[curnz] 		= ir[k];
				temp.numx[curnz++] 	= numx[k];
			}
			temp.mas[curnzc] = temp.mas[curnzc-1] + (mas[i] - mas[i-1]);
		}
		else	// they are equal, merge the column
		{
			temp.jc[curnzc++] = jc[i];
			ITYPE ii = mas[i];
			ITYPE prevnz = curnz;		
			while (ii < mas[i+1] && cindex == jc[i])	// cindex would be MAX if multstack is deplated
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
			while (ii < mas[i+1])
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
			temp.mas[curnzc] = temp.mas[curnzc-1] + curnz-prevnz;
			++i;
		}
	}
	while(i< nzc)
	{
		temp.jc[curnzc++] = jc[i++];
		for(IT k = mas[i-1]; k< mas[i]; ++k)
		{
			temp.ir[curnz] 		= ir[k];
			temp.numx[curnz++] 	= numx[k];
		}
		temp.mas[curnzc] = temp.mas[curnzc-1] + (mas[i] - mas[i-1]);
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

		temp.mas[curnzc] = temp.mas[curnzc-1] + columncount;
	}
	temp.Resize(curnzc, curnz);
	*this = temp;
	return *this;
}


/**
  * \remark Complexity: O(nnz)
  */
template <class IT, class NT>
Dcsc<IT,NT>::Dcsc (StackEntry<NT, pair<IT,IT> > * multstack, IT mdim, IT ndim, IT nnz): nz(nnz), colchunks(0),pool(NULL)
{
	if(nz == 0)	return;

	size_t sit = sizeof(IT);
	
	mas = (IT *) mallocarray ( (nz+1)*sit ); 	// to be shrinked
	jc  = (IT *) mallocarray ( nz*sit ); 		// to be shrinked
	ir  = (IT *) mallocarray ( nz*sit ); 
	numx= (NT *) mallocarray ( nz*sizeof(NT) ); 
	aux  = NULL;

	IT curnzc = 0;				// number of nonzero columns constructed so far
	IT cindex = multstack[0].key.first;
	IT rindex = multstack[0].key.second;

	ir[0]	= rindex;
	numx[0] = multstack[0].value;
	jc[curnzc] = cindex;
	mas[curnzc] = 0; 
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
			mas[curnzc] = i;
			++curnzc;
		}
	}

	// Shrink mas & jc arrays
	nzc = curnzc;
	IT * tmpjc	= jc; 
	IT * tmpmas 	= mas;
	
	mas = (IT *) mallocarray ( (nzc+1)*sit ); 	// new size
	jc  = (IT *) mallocarray ( nzc*sit ); 		// new size

	for(IT i=0; i< nzc; ++i)	// copy only a portion of the old elements
	{
		mas[i]	= tmpmas[i];
		jc[i]	= tmpjc[i];
	}
	mas[nzc] = nz;
	
	deletearray(tmpmas, (nz+1)*sit ); 
	deletearray(tmpjc, nz*sit );
}

/**
  * Create a logical matrix from (row/column) indices array
  * \remark This function should only be used for indexing 
  */
template <class IT, class NT>
Dcsc<IT,NT>::Dcsc (IT nnz, const vector<IT> & indices, bool isRow): nz(nnz),nzc(nnz),colchunks(0),pool(NULL)
{
	size_t sit = sizeof(IT);
	
	mas = (IT *) mallocarray ( (nz+1)*sit ); 	// to be shrinked
	jc  = (IT *) mallocarray ( nz*sit ); 		// to be shrinked
	ir  = (IT *) mallocarray ( nz*sit ); 
	numx= (NT *) mallocarray ( nz*sizeof(NT) ); 
	aux  = NULL;

	for(IT i=0; i <= nz; ++i)
		mas[i] = i;
	for(IT i=0; i < nz; ++i)
		numx[i] = static_cast<NT>(1);

	if(isRow)
	{	
		for(IT i=0; i < nz; ++i)
		{
			jc[i] = indices[i];
			ir[i] = i;
		}
	}
	else
	{
		for(IT i=0; i < nz; ++i)
		{
			ir[i] = indices[i];
			jc[i] = i;
		}
	}
}


template <class IT, class NT>
template <typename NNT>
Dcsc<IT,NNT> Dcsc<IT,NT>::ConvertNumericType ()
{
	Dcsc<IT,NNT> convert(nz, nzc);	// AUX = NULL
	
	for(IT i=0; i< nz; ++i)
	{		
		convert.numx[i] = static_cast<NNT>(convert.rhs.numx[i]);		
		convert.ir[i] = convert.rhs.ir[i];
	}
	
	IT tnzc = nzc+1;
	for(IT i=0; i< nzc; ++i)		
		convert.jc[i] = jc[i];
	for(IT i=0; i< tnzc; ++i)		
		convert.mas[i]= mas[i];
		
	return convert;
}

/**
  * Copy constructor that respects the memory pool
  * Copies the AUX array <=> it already exists
  */
template <class IT, class NT>
Dcsc<IT,NT>::Dcsc (const Dcsc<IT,NT> & rhs): colchunks(rhs.colchunks), nz(rhs.nz), nzc(rhs.nzc), cf(rhs.cf),pool(rhs.pool)
{
	size_t sit = sizeof(IT);
	if(nz > 0)
	{
		numx= (NT *) mallocarray ( nz*sizeof(NT) ); 
		ir  = (IT *) mallocarray ( nz*sit ); 
		for(IT i=0; i< nz; ++i)
		{		
			numx[i] = rhs.numx[i];		
			ir[i] = rhs.ir[i];
		}
	}
	else
	{
		numx = NULL;
		ir = NULL;
	}
	if(nzc > 0)
	{
		IT tnzc = nzc+1;
		jc  = (IT *) mallocarray ( nzc*sit ); 
		mas = (IT *) mallocarray ( tnzc*sit ); 		
	
		for(IT i=0; i< nzc; ++i)		
			jc[i] = rhs.jc[i];
		for(IT i=0; i< tnzc; ++i)		
			mas[i]= rhs.mas[i];
	}
	else
	{
		jc = NULL;
		mas = NULL;
	}
		
	if(colchunks > 0)
	{
		IT tchunks = colchunks + 1;
		aux  = (IT *) mallocarray ( tchunks*sit ); 
		for(IT i=0; i<tchunks; ++i)
			aux[i] = rhs.aux[i];
	}
	else
	{
		aux = NULL;
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
			deletearray(mas, sit * (nzc+1));
		}
		if(colchunks > 0)
		{
			deletearray(aux, sit * (colchunks+1));	
		}
		
		pool = rhs.pool;
		nz = rhs.nz;
		nzc = rhs.nzc;
		colchunks = rhs.colchunks;
		
		if(nz > 0)
		{
			numx= (NT *) mallocarray ( nz*sizeof(NT) ); 
			ir  = (IT *) mallocarray ( nz*sit ); 
			
			for(IT i=0; i< nz; ++i)	
			{	
				numx[i] = rhs.numx[i];		
				ir[i] = rhs.ir[i];
			}
		}
		else
		{
			numx = NULL;
			ir = NULL;
		}
	
		if(nzc > 0)
		{
			IT tnzc = nzc+1;
			jc  = (IT *) mallocarray ( nzc*sit ); 
			mas = (IT *) mallocarray ( tnzc*sit ); 

			for(IT i=0; i< nzc; ++i)		
				jc[i] = rhs.jc[i];
			for(IT i=0; i< tnzc; ++i)		
				mas[i]= rhs.mas[i];
		}
		else
		{
			jc = NULL;
			mas = NULL;
		}

		if(colchunks > 0)
		{
			IT tchunks = colchunks + 1;
			aux  = (IT *) mallocarray ( tchunks*sit ); 

			for(IT i=0; i<tchunks; ++i)
				aux[i] = rhs.aux[i];
		}
		else
		{
			aux = NULL;
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
	temp.mas[0] = 0;
	while(i< nzc && j<rhs.nzc)
	{
		if(jc[i] > rhs.jc[j])
		{
			temp.jc[curnzc++] = rhs.jc[j++];
			for(IT k = rhs.mas[j-1]; k< rhs.mas[j]; ++k)
			{
				temp.ir[curnz] 		= rhs.ir[k];
				temp.numx[curnz++] 	= rhs.numx[k];
			}
			temp.mas[curnzc] = temp.mas[curnzc-1] + (rhs.mas[j] - rhs.mas[j-1]);
		}
		else if(jc[i] < rhs.jc[j])
		{
			temp.jc[curnzc++] = jc[i++];
			for(IT k = mas[i-1]; k< mas[i]; k++)
			{
				temp.ir[curnz] 		= ir[k];
				temp.numx[curnz++] 	= numx[k];
			}
			temp.mas[curnzc] = temp.mas[curnzc-1] + (mas[i] - mas[i-1]);
		}
		else
		{
			temp.jc[curnzc++] = jc[i];
			IT ii = mas[i];
			IT jj = rhs.mas[j];
			IT prevnz = curnz;		
			while (ii < mas[i+1] && jj < rhs.mas[j+1])
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
			while (ii < mas[i+1])
			{
				temp.ir[curnz] = ir[ii];
				temp.numx[curnz++] = numx[ii++];
			}
			while (jj < rhs.mas[j+1])
			{
				temp.ir[curnz] = rhs.ir[jj];
				temp.numx[curnz++] = rhs.numx[jj++];
			}
			temp.mas[curnzc] = temp.mas[curnzc-1] + curnz-prevnz;
			++i;
			++j;
		}
	}
	while(i< nzc)
	{
		temp.jc[curnzc++] = jc[i++];
		for(IT k = mas[i-1]; k< mas[i]; ++k)
		{
			temp.ir[curnz] 	= ir[k];
			temp.numx[curnz++] = numx[k];
		}
		temp.mas[curnzc] = temp.mas[curnzc-1] + (mas[i] - mas[i-1]);
	}
	while(j < rhs.nzc)
	{
		temp.jc[curnzc++] = rhs.jc[j++];
		for(IT k = rhs.mas[j-1]; k< rhs.mas[j]; ++k)
		{
			temp.ir[curnz] 	= rhs.ir[k];
			temp.numx[curnz++] 	= rhs.numx[k];
		}
		temp.mas[curnzc] = temp.mas[curnzc-1] + (rhs.mas[j] - rhs.mas[j-1]);
	}
	temp.Resize(curnzc, curnz);
	*this = temp;
	return *this;
}
	
template <class IT, class NT>
void Dcsc<IT,NT>::DeleteAux()
{
	size_t sit = sizeof(IT);
	if(colchunks > 0)	
	{
		deletearray(aux, (colchunks+1)*sit);
	}
	cf = 0.0;
	colchunks =0;
}

template <class IT, class NT>
void Dcsc<IT,NT>::ConstructAux(IT ndim)
{
	size_t sit = sizeof(ITYPE);
	if(colchunks > 0)	// aux may be empty even when other arrays are not.
	{
		deletearray(aux, (colchunks+1)*sit);	
	}

	// cf and colchunks are recomputed since nzc might have changed !
	cf	  = static_cast<float>(ndim+1) / static_cast<float>(nzc);
	colchunks = static_cast<IT> ( ceil( static_cast<float>(ndim+1) / ceil(cf)) );

	aux  = (IT *) mallocarray ( (colchunks+1)*sit ); 

	IT chunksize	= static_cast<IT>(ceil(cf));
	IT reg		= static_cast<IT>(0);
	IT curchunk	= static_cast<IT>(0);
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
}


template <class IT, class NT>
void Dcsc<IT,NT>::Resize(IT nzcnew, IT nznew)
{
	size_t sit = sizeof(I);

	if(nznew == nz && nzcnew == nzc)
	{
		// No need to do anything!
		return;
	}
	if(nzcnew == 0)
	{
		deletearray(jc, sit * nzc);
		deletearray(mas, sit * (nzc+1));

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

	IT * tmpmas = mas; 
	IT * tmpjc = jc;
	
	mas = (IT *) mallocarray ( (nzcnew+1)*sit ); 
	jc = (IT *) mallocarray (  nzcnew * sit ); 	

	if(nzcnew > nzc)	// Grow it
	{
		for(IT i=0; i< nzc; ++i)	// copy all of the old elements
		{
			mas[i] = tmpmas[i];
			jc[i] = tmpjc[i];
		}
		mas[nzc] = tmpmas[nzc];
	}
	else			// Shrink it 
	{
		for(IT i=0; i< nzcnew; ++i)	// copy only a portion of the old elements
		{
			mas[i] = tmpmas[i];
			jc[i] = tmpjc[i];
		}
		mas[nzcnew] = tmpmas[nzcnew];
	}
	deletearray(tmpmas, sit * (nzc+1));	// delete the memory pointed by previous pointers
	deletearray(tmpjc, sit * nzc);

	nzc = nzcnew;
	
	NT * tmpnumx = numx; 
	IT * tmpir = ir;

	numx = (NT *) mallocarray ( nznew * sizeof(NT) ); 
	ir = (IT *) mallocarray (  nznew * sit ); 

	if(nznew > nz)	// Grow it
	{
		for(IT i=0; i< nz; ++i)		// copy all of the old elements
		{
			numx[i] = tmpnumx[i];
			ir[i] = tmpir[i];
		}
	}
	else	// Shrink it 
	{
		for(IT i=0; i< nznew; ++i)	// copy only a portion of the old elements
		{
			numx[i] = tmpnumx[i];
			ir[i] = tmpir[i];
		}
	}
	deletearray(tmpnumx, nz * sizeof(NT));	// delete the memory pointed by previous pointers
	deletearray(tmpir, nz * sit);
	
	nz = nznew;
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
		deletearray(mas, (nzc+1) * sit);
	}
	if(colchunks > 0)	// aux may be empty even when other arrays are not.
	{
		deletearray(aux, (colchunks+1) * sit);		
	}
}

