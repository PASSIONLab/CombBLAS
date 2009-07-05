/****************************************************************/
/* Sequential and Parallel Sparse Matrix Multiplication Library */
/* version 2.3 --------------------------------------------------/
/* date: 01/18/2009 ---------------------------------------------/
/* author: Aydin Buluc (aydin@cs.ucsb.edu) ----------------------/
/****************************************************************/

#ifndef _SP_TUPLES_H
#define _SP_TUPLES_H

#include <iostream>
#include <fstream>
#include <cmath>
#include <cassert>
#include <tr1/tuple>
#include "SpMat.h"
#include "SpDefs.h"
#include "StackEntry.h"
#include "Compare.h"

using namespace std;
using namespace std::tr1;

template <class IU, class NU>
class SpDCCols;

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
	SpTuples (IT size, IT nRow, IT nCol, StackEntry<NT, pair<IT,IT> > * & multstack);		
	SpTuples (const SpTuples<IT,NT> & rhs);	 	// Actual Copy constructor
	SpTuples (const SpDCCols<IT,NT> & rhs); 	// Copy constructor for conversion from SpDCCols
	~SpTuples();

	SpTuples<IT,NT> & operator=(const SpTuples<IT,NT> & rhs);

	IT & rowindex (IT i) { return tr1::get<0>(tuples[i]); }
	IT & colindex (IT i) { return tr1::get<1>(tuples[i]); }
	NT & numvalue (IT i) { return tr1::get<2>(tuples[i]); }

	IT rowindex (IT i) const { return tr1::get<0>(tuples[i]); }
	IT colindex (IT i) const { return tr1::get<1>(tuples[i]); } 
	NT numvalue (IT i) const { return tr1::get<2>(tuples[i]); } 

	void SortRowBased()
	{
		sort(tuples , tuples+nnz);	// Default "operator<" for tuples uses lexicographical ordering 
	}

	void SortColBased()
	{
		ColLexiCompare<IT,NT> collexicogcmp;
		sort(tuples , tuples+nnz, collexicogcmp );
	}

	pair<IT,IT> RowLimits()
	{	
		RowCompare<IT,NT> rowcmp;
		tuple<IT,IT,NT> * maxit = max_element(tuples, tuples+nnz, rowcmp);	
		tuple<IT,IT,NT> * minit = min_element(tuples, tuples+nnz, rowcmp);

		return make_pair(tr1::get<0>(*minit), tr1::get<0>(*maxit));
	}
	pair<IT,IT> ColLimits()
	{	
		ColCompare<IT,NT> colcmp;
		tuple<IT,IT,NT> * maxit = max_element(tuples, tuples+nnz, colcmp);
		tuple<IT,IT,NT> * minit = min_element(tuples, tuples+nnz, colcmp);

		return make_pair(tr1::get<1>(*minit), tr1::get<1>(*maxit));
	}

	// Performs a balanced merge of the array of SpTuples
	template<typename SR, typename IU, typename NU>
	friend SpTuples<IU,NU> MergeAll(const vector<SpTuples<IU,NU> *> & ArrSpTups); 

	ofstream& put (ofstream& outfile) const;		
	ifstream& get (ifstream& infile); 

	IT getnrow() const { return m; }
	IT getncol() const { return n; }
	IT getnnz() const { return nnz; }

	void PrintInfo();

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

	const static IT zero = static_cast<IT>(0);	

	SpTuples (){};		// Default constructor does nothing, hide it
	
	void FillTuples (Dcsc<IT,NT> * mydcsc);

	template <class IU, class NU>
	friend class SpDCCols;
};


#include "SpTuples.cpp"	
#endif
