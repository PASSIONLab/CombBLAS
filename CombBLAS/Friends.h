#ifndef _FRIENDS_H_
#define _FRIENDS_H_

#include <iostream>
#include "SpMat.h"	// Best to include the base class first
#include "SpHelper.h"
#include "StackEntry.h"
#include "Isect.h"
#include "dcsc.h"
#include "Deleter.h"
#include "Compare.h"

using namespace std;

template <class IU, class NU>	
class SpTuples;

template <class IU, class NU>	
class SpDCCols;


/****************************************************************************/
/**************************** FRIEND FUNCTIONS ******************************/
/****************************************************************************/

/**
 * SpTuples(A*B') (Using OuterProduct Algorithm)
 * Returns the tuples for efficient merging later
 * Support mixed precision multiplication
 * The multiplication is on the specified semiring (passed as parameter)
 */
template<class SR, class IU, class NU1, class NU2>
SpTuples<IU, typename promote_trait<NU1,NU2>::T_promote> * Tuples_AnXBt 
					(const SpDCCols<IU, NU1> & A, 
					 const SpDCCols<IU, NU2> & B)
{
	typedef typename promote_trait<NU1,NU2>::T_promote T_promote;  
	const static IU zero = static_cast<IU>(0);		

	IU mdim = A.m;	
	IU ndim = B.m;	// B is already transposed

	if(A.isZero() || B.isZero())
	{
		return new SpTuples< IU, T_promote >(zero, mdim, ndim);	// just return an empty matrix
	}
	Isect<IU> *isect1, *isect2, *itr1, *itr2, *cols, *rows;
	SpHelper::SpIntersect(*(A.dcsc), *(B.dcsc), cols, rows, isect1, isect2, itr1, itr2);
	
	IU kisect = static_cast<IU>(itr1-isect1);		// size of the intersection ((itr1-isect1) == (itr2-isect2))
	if(kisect == zero)
	{
		DeleteAll(isect1, isect2, cols, rows);
		return new SpTuples< IU, T_promote >(zero, mdim, ndim);
	}
	
	StackEntry< T_promote, pair<IU,IU> > * multstack;
	IU cnz = SpHelper::SpCartesian< SR > (*(A.dcsc), *(B.dcsc), kisect, isect1, isect2, multstack);  
	DeleteAll(isect1, isect2, cols, rows);

	return new SpTuples<IU, T_promote> (cnz, mdim, ndim, multstack);
}

/**
 * SpTuples(A*B) (Using ColByCol Algorithm)
 * Returns the tuples for efficient merging later
 * Support mixed precision multiplication
 * The multiplication is on the specified semiring (passed as parameter)
 */
template<class SR, class IU, class NU1, class NU2>
SpTuples<IU, typename promote_trait<NU1,NU2>::T_promote> * Tuples_AnXBn 
					(const SpDCCols<IU, NU1> & A, 
					 const SpDCCols<IU, NU2> & B )
{
	typedef typename promote_trait<NU1,NU2>::T_promote T_promote; 
	const static IU zero = static_cast<IU>(0);	

	IU mdim = A.m;	
	IU ndim = B.n;	
	if(A.isZero() || B.isZero())
	{
		return new SpTuples<IU, T_promote>(zero, mdim, ndim);
	}
	StackEntry< T_promote, pair<IU,IU> > * multstack;
	IU cnz = SpHelper::SpColByCol< SR > (*(A.dcsc), *(B.dcsc), A.n,  multstack);  
	
	return new SpTuples<IU, T_promote> (cnz, mdim, ndim, multstack);
}


template<class SR, class IU, class NU1, class NU2>
SpTuples<IU, typename promote_trait<NU1,NU2>::T_promote> * Tuples_AtXBt 
					(const SpDCCols<IU, NU1> & A, 
					 const SpDCCols<IU, NU2> & B )
{
	typedef typename promote_trait<NU1,NU2>::T_promote T_promote; 
	const static IU zero = static_cast<IU>(0);
	
	IU mdim = A.n;	
	IU ndim = B.m;	
	cout << "Tuples_AtXBt function has not been implemented yet !" << endl;
		
	return new SpTuples<IU, T_promote> (zero, mdim, ndim);
}

template<class SR, class IU, class NU1, class NU2>
SpTuples<IU, typename promote_trait<NU1,NU2>::T_promote> * Tuples_AtXBn 
					(const SpDCCols<IU, NU1> & A, 
					 const SpDCCols<IU, NU2> & B )
{
	typedef typename promote_trait<NU1,NU2>::T_promote T_promote; 	// T_promote is a type (so the extra "typename"), NOT a value
	const static IU zero = static_cast<IU>(0);	

	IU mdim = A.n;	
	IU ndim = B.n;	
	cout << "Tuples_AtXBn function has not been implemented yet !" << endl;
		
	return new SpTuples<IU, T_promote> (zero, mdim, ndim);
}

// Performs a balanced merge of the array of SpTuples
// Assumes the input parameters are already column sorted
template<class SR, class IU, class NU>
SpTuples<IU,NU> MergeAll( const vector<SpTuples<IU,NU> *> & ArrSpTups)
{
	int hsize =  ArrSpTups.size();		
	assert(hsize > 0);

	IU mstar = ArrSpTups[0]->m;
	IU nstar = ArrSpTups[0]->n;
	for(int i=1; i< hsize; ++i)
	{
		if((mstar != ArrSpTups[i]->m) || nstar != ArrSpTups[i]->n)
		{
			cerr << "Dimensions do not match on MergeAll()" << endl;
			return SpTuples<IU,NU>(0,0,0);
		}
	}
	if(hsize > 1)
	{
		ColLexiCompare<IU,int> heapcomp;
		tuple<IU, IU, int> * heap = new tuple<IU, IU, int> [hsize];	// (rowindex, colindex, source-id)
		IU * curptr = new IU[hsize];
		fill_n(curptr, hsize, static_cast<IU>(0)); 
		IU estnnz = 0;

		for(int i=0; i< hsize; ++i)
		{
			estnnz += ArrSpTups[i]->getnnz();
			heap[i] = make_tuple(tr1::get<0>(ArrSpTups[i]->tuples[0]), tr1::get<1>(ArrSpTups[i]->tuples[0]), i);
		}	
		make_heap(heap, heap+hsize, not2(heapcomp));

		tuple<IU, IU, NU> * ntuples = new tuple<IU,IU,NU>[estnnz]; 
		IU cnz = 0;

		while(hsize > 0)
		{
			pop_heap(heap, heap + hsize, not2(heapcomp));         // result is stored in heap[hsize-1]
			int source = tr1::get<2>(heap[hsize-1]);

			if( (cnz != 0) && 
				((tr1::get<0>(ntuples[cnz-1]) == tr1::get<0>(heap[hsize-1])) && (tr1::get<1>(ntuples[cnz-1]) == tr1::get<1>(heap[hsize-1]))) )
			{
				tr1::get<2>(ntuples[cnz-1])  = SR::add(tr1::get<2>(ntuples[cnz-1]), ArrSpTups[source]->numvalue(curptr[source]++)); 
			}
			else
			{
				ntuples[cnz++] = ArrSpTups[source]->tuples[curptr[source]++];
			}
			
			if(curptr[source] != ArrSpTups[source]->getnnz())	// That array has not been depleted
			{
				heap[hsize-1] = make_tuple(tr1::get<0>(ArrSpTups[source]->tuples[curptr[source]]), 
								tr1::get<1>(ArrSpTups[source]->tuples[curptr[source]]), source);
				push_heap(heap, heap+hsize, not2(heapcomp));
			}
			else
			{
				--hsize;
			}
		}
		SpHelper::ShrinkArray(ntuples, cnz);
		DeleteAll(heap, curptr);
		return SpTuples<IU,NU> (cnz, mstar, nstar, ntuples);
	}
	else
	{
		return SpTuples<IU,NU> (*ArrSpTups[0]);
	}
}



#endif
