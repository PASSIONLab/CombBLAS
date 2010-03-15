#ifndef _FRIENDS_H_
#define _FRIENDS_H_

#include <iostream>
#include "SpMat.h"	// Best to include the base class first
#include "SpHelper.h"
#include "StackEntry.h"
#include "Isect.h"
#include "Deleter.h"
#include "Compare.h"

using namespace std;

template <class IU, class NU>	
class SpTuples;

template <class IU, class NU>	
class SpDCCols;

template <class IU, class NU>	
class Dcsc;



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

/**
  * Differences with MergeAll:
  * Returns a pointer (instead of the object) to the newly created matrix, which is the merged result
  * Deletes the contents of the ArrSpTuples for memory efficiency during the merge
  **/ 
template<class SR, class IU, class NU>
SpTuples<IU,NU> * MergeAllRec(const vector<SpTuples<IU,NU> *> & ArrSpTups, IU mstar =0, IU nstar = 0)
{
	int hsize =  ArrSpTups.size();
        if(hsize == 0)
        {
                return new SpTuples<IU,NU>(0, mstar,nstar);
        }
        else
        {
                mstar = ArrSpTups[0]->m;
                nstar = ArrSpTups[0]->n;
        }
        for(int i=1; i< hsize; ++i)
        {
                if((mstar != ArrSpTups[i]->m) || nstar != ArrSpTups[i]->n)
                {
                        cerr << "Dimensions do not match on MergeAll()" << endl;
                        return new SpTuples<IU,NU>(0,0,0);
                }
        }
        if(hsize == 2)
        {
		ColLexiCompare<IU,NU> collexicogcmp;
		TupleEqual<IU,NU> tupleequal;

		tuple<IU,IU,NU> * __first1 = ArrSpTups[0]->tuples;
		tuple<IU,IU,NU> * __last1 = ArrSpTups[0]->tuples + ArrSpTups[0]->nnz;
		tuple<IU,IU,NU> * __first2 = ArrSpTups[1]->tuples;
		tuple<IU,IU,NU> * __last2 = ArrSpTups[1]->tuples + ArrSpTups[1]->nnz;
		
		tuple<IU,IU,NU> * __result = new tuple<IU,IU,NU>[ArrSpTups[0]->nnz + ArrSpTups[1]->nnz];
		tuple<IU,IU,NU> * begres = __result;

		while (__first1 != __last1 && __first2 != __last2)
        	{
          		if (collexicogcmp(*__first2, *__first1))	// *__first2 < *__first1
            		{
				if( tupleequal(*__first2, *__result) )
				{
					tr1::get<2>(*__result) = SR::add( tr1::get<2>(*__result), tr1::get<2>(*__first2) );
				}
				else
				{ 
              				*__result = *__first2;
					++__result;
				}
				++__first2;
            		}
          		else		// **__first2 > *__first1
            		{
				if( tupleequal(*__first1, *__result) )
				{
					tr1::get<2>(*__result) = SR::add( tr1::get<2>(*__result), tr1::get<2>(*__first1) );
				}
				else
				{ 
              				*__result = *__first1;
					++__result;
				}
              			++__first1;
            		}
        	}
      		std::copy(__first2, __last2, std::copy(__first1, __last1,__result));	// copy whichever remains

		unsigned long nsize = __result-begres;
		delete ArrSpTups[0];
		delete ArrSpTups[1];	
		SpHelper::ShrinkArray(begres, nsize);
		
		return new SpTuples<IU,NU> (nsize, mstar, nstar, begres);
	}
	else if(hsize == 1)
        {
                return new SpTuples<IU,NU> (*ArrSpTups[0]);
        }
	else
	{	
		// TODO: This part looks buggy !
	
		vector<SpTuples<IU,NU> *> smallerlist;

		int i = 0;
		while( i < (hsize-1) )
		{ 
			vector<SpTuples<IU,NU> *> twolists(2);
			twolists[0] = ArrSpTups[i++];
			twolists[1] = ArrSpTups[i++];
		
                        smallerlist.push_back(MergeAllRec<SR>(twolists,mstar,nstar));
		}
		if(i < hsize)
		{
			smallerlist.push_back(ArrSpTups[i]);
		}
		return MergeAllRec<SR>(smallerlist, mstar, nstar);
	}
}


// Performs a balanced merge of the array of SpTuples
// Assumes the input parameters are already column sorted
template<class SR, class IU, class NU>
SpTuples<IU,NU> MergeAll( const vector<SpTuples<IU,NU> *> & ArrSpTups, IU mstar = 0, IU nstar = 0)
{
	int hsize =  ArrSpTups.size();		
	if(hsize == 0)
	{
		return SpTuples<IU,NU>(0, mstar,nstar);
	}
	else
	{
		mstar = ArrSpTups[0]->m;
		nstar = ArrSpTups[0]->n;
	}
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
				double t1 = tr1::get<2>(ntuples[cnz-1]);
				double t2 = ArrSpTups[source]->numvalue(curptr[source]++);
				tr1::get<2>(ntuples[cnz-1]) = SR::add(t1,t2);
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


/**
 * @param[in]   exclude if false,
 *      \n              then operation is A = A .* B
 *      \n              else operation is A = A .* not(B) 
 * \attention The memory pool of the lvalue is preserved:
 * 	\n	If A = A .* B where B uses pinnedPool and A uses NULL before the operation,
 * 	\n	then after the operation A still uses NULL memory (old school 'malloc')
 **/
template <typename IU, typename NU1, typename NU2>
Dcsc<IU, typename promote_trait<NU1,NU2>::T_promote> EWiseMult(const Dcsc<IU,NU1> & A, const Dcsc<IU,NU2> & B, bool exclude)
{
	typedef typename promote_trait<NU1,NU2>::T_promote N_promote;
	IU estnzc, estnz;
	if(exclude)
	{	
		estnzc = A.nzc;
		estnz = A.nz; 
	} 
	else
	{
		estnzc = std::min(A.nzc, B.nzc);
		estnz  = std::min(A.nz, B.nz);
	}

	Dcsc<IU,N_promote> temp(estnz, estnzc);

	IU curnzc = 0;
	IU curnz = 0;
	IU i = 0;
	IU j = 0;
	temp.cp[0] = Dcsc<IU,NU1>::zero;
	
	if(!exclude)	// A = A .* B
	{
		while(i< A.nzc && j<B.nzc)
		{
			if(A.jc[i] > B.jc[j]) 		++j;
			else if(A.jc[i] < B.jc[j]) 	++i;
			else
			{
				IU ii = A.cp[i];
				IU jj = B.cp[j];
				IU prevnz = curnz;		
				while (ii < A.cp[i+1] && jj < B.cp[j+1])
				{
					if (A.ir[ii] < B.ir[jj])	++ii;
					else if (A.ir[ii] > B.ir[jj])	++jj;
					else
					{
						temp.ir[curnz] = A.ir[ii];
						temp.numx[curnz++] = A.numx[ii++] * B.numx[jj++];	
					}
				}
				if(prevnz < curnz)	// at least one nonzero exists in this column
				{
					temp.jc[curnzc++] = A.jc[i];	
					temp.cp[curnzc] = temp.cp[curnzc-1] + curnz-prevnz;
				}
				++i;
				++j;
			}
		}
	}
	else	// A = A .* not(B)
	{
		while(i< A.nzc && j< B.nzc)
		{
			if(A.jc[i] > B.jc[j])		++j;
			else if(A.jc[i] < B.jc[j])
			{
				temp.jc[curnzc++] = A.jc[i++];
				for(IU k = A.cp[i-1]; k< A.cp[i]; k++)	
				{
					temp.ir[curnz] 		= A.ir[k];
					temp.numx[curnz++] 	= A.numx[k];
				}
				temp.cp[curnzc] = temp.cp[curnzc-1] + (A.cp[i] - A.cp[i-1]);
			}
			else
			{
				IU ii = A.cp[i];
				IU jj = B.cp[j];
				IU prevnz = curnz;		
				while (ii < A.cp[i+1] && jj < B.cp[j+1])
				{
					if (A.ir[ii] > B.ir[jj])	++jj;
					else if (A.ir[ii] < B.ir[jj])
					{
						temp.ir[curnz] = A.ir[ii];
						temp.numx[curnz++] = A.numx[ii++];
					}
					else	// eliminate those existing nonzeros
					{
						++ii;	
						++jj;	
					}
				}
				while (ii < A.cp[i+1])
				{
					temp.ir[curnz] = A.ir[ii];
					temp.numx[curnz++] = A.numx[ii++];
				}

				if(prevnz < curnz)	// at least one nonzero exists in this column
				{
					temp.jc[curnzc++] = A.jc[i];	
					temp.cp[curnzc] = temp.cp[curnzc-1] + curnz-prevnz;
				}
				++i;
				++j;
			}
		}
		while(i< A.nzc)
		{
			temp.jc[curnzc++] = A.jc[i++];
			for(IU k = A.cp[i-1]; k< A.cp[i]; ++k)
			{
				temp.ir[curnz] 	= A.ir[k];
				temp.numx[curnz++] = A.numx[k];
			}
			temp.cp[curnzc] = temp.cp[curnzc-1] + (A.cp[i] - A.cp[i-1]);
		}
	}

	temp.Resize(curnzc, curnz);
	return temp;
}	


template<typename IU, typename NU1, typename NU2>
SpDCCols<IU, typename promote_trait<NU1,NU2>::T_promote > EWiseMult (const SpDCCols<IU,NU1> & A, const SpDCCols<IU,NU2> & B, bool exclude)
{
	typedef typename promote_trait<NU1,NU2>::T_promote N_promote; 
	assert(A.m == B.m);
	assert(A.n == B.n);

	Dcsc<IU, N_promote> * tdcsc = NULL;
	if(A.nnz > 0 && B.nnz > 0)
	{ 
		tdcsc = new Dcsc<IU, N_promote>(EWiseMult(*(A.dcsc), *(B.dcsc), exclude));
		return 	SpDCCols<IU, N_promote> (A.m , A.n, tdcsc);
	}
	else
	{
		return 	SpDCCols<IU, N_promote> (A.m , A.n, tdcsc);
	}
}


#endif
