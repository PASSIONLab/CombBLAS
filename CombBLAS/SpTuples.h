/****************************************************************/
/* Sequential and Parallel Sparse Matrix Multiplication Library */
/* version 2.3 --------------------------------------------------/
/* date: 01/18/2009 ---------------------------------------------/
/* author: Aydin Buluc (aydin@cs.ucsb.edu) ----------------------/
/****************************************************************/

#ifndef _SPARSE_TRIPLETS_H
#define _SPARSE_TRIPLETS_H

#include <iostream>
#include <fstream>
#include <cmath>
#include "SpDefs.h"
#include "Compare.h"
#include <boost/tuple/tuple.hpp>
#include <boost/tuple/tuple_comparison.hpp>

using namespace std;
using namespace boost;


template <class IU, class NU>
class SparseDColumn;

template <class IU, class NU>
class Dcsc;


/**
 * Triplets are represented using the boost::tuple class of the Boost library
 * \remarks Indices start from 0 in this class
 */
template <class IT, class NT>
class SpTuples: public SpMat<IT, NT, SpTuples<IT,NT> >
{
public:
	// Constructors 
	SpTuples (IT size, IT nRow, IT nCol);
	SpTuples (IT size, IT nRow, IT nCol, tuple<IT, IT, NT> * mytuples);	
	SpTuples (const SpTuples<IT,NT> & rhs);	 	// Actual Copy constructor
	SpTuples (const SpDCCols<IT,NT> & rhs); 	// Copy constructor for conversion from SpDCCols
	~SpTuples();

	SpTuples<IT,NT> & operator=(const SpTuples<IT,NT> & rhs);

	virtual SpTuples<IT, NT> * operator() (const vector<IT> & ri, const vector<IT> & ci) const
	{
		return const_cast<SpTuples<IT,NT> *>(this); 
	};

	IT rowindex (IT i) { return get<0>(tuples[i]); }
	IT colindex (IT i) { return get<1>(tuples[i]); }
	NT numvalue (IT i) { return get<2>(tuples[i]); }


	void SortRowBased()
	{
		sort(tuples , tuples+nz);	// Default "operator<" for tuples uses lexicographical ordering 
	};

	void SortColBased()
	{
		ColSortCompare<IT,NT> colcmp;
		sort(tuples , tuples+nz, colcmp );
	};

	// Functions declared as friends
	template <class IU, class NU>
	friend ofstream& operator<< (ofstream& outfile, const SpTuples<IU,NU> & s);		
	template <class IU, class NU>
	friend ifstream& operator>> (ifstream& infile, SpTuples<IU,NU> & s); 

	virtual void printInfo() 
	{
		cout << "This is a SpTuples class" << endl;
	}

private:
	tuple<IT, IT, NT> * tuples; 	// boost:tuple
	/** 
	 **	tuple elements with indices:
	 **	0) IT * ir ;	    	//  array of row indices, size nnz 
	 **	1) IT * jc ;	    	//  array of col indices, size nnz 
	 **	2) NT * numx;		//  array of generic values, size nnz
	 **/

	IT m;
	IT n;
	IT nnz;		

	SpTuples (){};		// Default constructor does nothing, hide it
	
	void FillTuples (Dcsc<IT,NT> * mydcsc);

	template <class IU, class NU>
	friend class SpDCCols;
};


#include "SpTuples.cpp"	
#endif
