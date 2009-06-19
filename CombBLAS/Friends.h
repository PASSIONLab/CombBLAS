#ifndef _FRIENDS_H_
#define _FRIENDS_H_

#include <iostream>
#include "SpMat.h"	// Best to include the base class first
#include "SpHelper.h"
#include "StackEntry.h"
#include "Isect.h"
#include "dcsc.h"
#include "Deleter.h"
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
template<class IU, class NU1, class NU2, class SR>
SpTuples<IU, typename promote_trait<NU1,NU2>::T_promote> Tuples_AnXBt 
					(const SpDCCols<IU, NU1> & A, 
					 const SpDCCols<IU, NU2> & B)
{
	typedef typename promote_trait<NU1,NU2>::T_promote T_promote;  
	const static IU zero = static_cast<IU>(0);		

	IU mdim = A.m;	
	IU ndim = B.m;	// B is already transposed

	if(A.isZero() || B.isZero())
	{
		return SpTuples< IU, T_promote >(zero, mdim, ndim);	// just return an empty matrix
	}
	Isect<IU> *isect1, *isect2, *itr1, *itr2, *cols, *rows;
	SpHelper::SpIntersect(*(A->dcsc), *(B->dcsc), cols, rows, isect1, isect2, itr1, itr2);
	
	IU kisect = static_cast<IU>(itr1-isect1);		// size of the intersection ((itr1-isect1) == (itr2-isect2))
	if(kisect == zero)
	{
		DeleteAll(isect1, isect2, cols, rows);
		return SpTuples< IU, T_promote >(zero, mdim, ndim);
	}
	
	StackEntry< T_promote, pair<IU,IU> > * multstack;
	IU cnz = SpHelper::SpCartesian< SR > (*(A.dcsc), *(B.dcsc), kisect, isect1, isect2, multstack);  
	DeleteAll(isect1, isect2, cols, rows);

	return SpTuples<IU, T_promote> (cnz, mdim, ndim, multstack);
}

/**
 * SpTuples(A*B) (Using ColByCol Algorithm)
 * Returns the tuples for efficient merging later
 * Support mixed precision multiplication
 * The multiplication is on the specified semiring (passed as parameter)
 */
template<class IU, class NU1, class NU2, class SR>
SpTuples<IU, typename promote_trait<NU1,NU2>::T_promote> Tuples_AnXBn 
					(const SpDCCols<IU, NU1> & A, 
					 const SpDCCols<IU, NU2> & B )
{
	typedef typename promote_trait<NU1,NU2>::T_promote T_promote; 
	const static IU zero = static_cast<IU>(0);	

	IU mdim = A.m;	
	IU ndim = B.n;	
	if(A.isZero() || B.isZero())
	{
		return SpTuples<IU, T_promote>(zero, mdim, ndim);
	}
	StackEntry< T_promote, pair<IU,IU> > * multstack;
	IU cnz = SpHelper::SpColByCol< SR > (*(A.dcsc), *(B.dcsc), multstack);  
	
	return SpTuples<IU, T_promote> (cnz, mdim, ndim, multstack);
}


template<class IU, class NU1, class NU2, class SR>
SpTuples<IU, typename promote_trait<NU1,NU2>::T_promote> Tuples_AtXBt 
					(const SpDCCols<IU, NU1> & A, 
					 const SpDCCols<IU, NU2> & B )
{
	typedef typename promote_trait<NU1,NU2>::T_promote T_promote; 
	const static IU zero = static_cast<IU>(0);
	
	IU mdim = A.n;	
	IU ndim = B.m;	
	cout << "Tuples_AtXBt function has not been implemented yet !" << endl;
		
	return SpTuples<IU, T_promote> (zero, mdim, ndim);
}

template<class IU, class NU1, class NU2, class SR>
SpTuples<IU, typename promote_trait<NU1,NU2>::T_promote> Tuples_AtXBn 
					(const SpDCCols<IU, NU1> & A, 
					 const SpDCCols<IU, NU2> & B )
{
	typedef typename promote_trait<NU1,NU2>::T_promote T_promote; 	// T_promote is a type (so the extra "typename"), NOT a value
	const static IU zero = static_cast<IU>(0);	

	IU mdim = A.n;	
	IU ndim = B.n;	
	cout << "Tuples_AtXBn function has not been implemented yet !" << endl;
		
	return SpTuples<IU, T_promote> (zero, mdim, ndim);
}


#endif
