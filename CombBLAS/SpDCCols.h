/****************************************************************/
/* Sequential and Parallel Sparse Matrix Multiplication Library */
/* version 2.3 --------------------------------------------------/
/* date: 01/18/2009 ---------------------------------------------/
/* author: Aydin Buluc (aydin@cs.ucsb.edu) ----------------------/
/****************************************************************/


#ifndef _SP_DCCOLS_H
#define _SP_DCCOLS_H

#include <iostream>
#include <fstream>
#include <cmath>
// #include "SpMat.h"	// Best to include the base class first
#include "SpTuples.h"
#include "SpHelper.h"
#include "StackEntry.h"
#include "dcsc.h"
#include "Isect.h"
#include "Semirings.h"
#include "MemoryPool.h"


template <class IT, class NT>
class SpDCCols 		//: public SpMat<IT, NT, SpDCCols<IT, NT> >
{
public:
	// Constructors :
	SpDCCols ();
	SpDCCols (ITYPE size, ITYPE nRow, ITYPE nCol, ITYPE nzc);
	SpDCCols (const SpTuples<IT,NT> & rhs, bool transpose, MemoryPool * mpool = NULL);
	SpDCCols (const SpDCCols<IT,NT> & rhs);	// Actual copy constructor		
	~SpDCCols();

	template <typename NNT>
	SpDCCols<IT, NNT> ConvertNumericType();		//!< NNT: New numeric type

	// Member Functions and Operators: 
	SpDCCols<IT,NT> & operator= (const SpDCCols<IT, NT> & rhs);
	SpDCCols<IT,NT> & operator+= (const SpDCCols<IT, NT> & rhs);
	SpDCCols<IT,NT> operator() (const vector<IT> & ri, const vector<IT> & ci) const;

	SpDCCols<IT,NT> Transpose();			//!< \attention Destroys calling object (*this)
	SpDCCols<IT,NT> TransposeConst() const;		//!< Const version, doesn't touch the existing object

	void Split(SpDCCols<IT,NT> & partA, SpDCCols<IT,NT> & partB); 	//!< \attention Destroys calling object (*this)
	void Merge(SpDCCols<IT,NT> & partA, SpDCCols<IT,NT> & partB);	//!< \attention Destroys its parameters (partA & partB)

	Arr<IT,NT> GetArrays() const;
	void createImpl(vector<IT> & essentials);
	vector<IT> GetEssentials() const;

	bool isZero() { return (nnz == zero); }
	IT getnrow() const { return m; }
	IT getncol() const { return n; }
	IT getnnz() const { return nnz; }
	
	ofstream& put(ofstream& outfile) const;
	void PrintInfo() const;

	int PlusEq_AtXBt(const SpDCCols<T> & A, const SpDCCols<T> & B);  
	int PlusEq_AtXBn(const SpDCCols<T> & A, const SpDCCols<T> & B);
	int PlusEq_AnXBt(const SpDCCols<T> & A, const SpDCCols<T> & B);  
	int PlusEq_AnXBn(const SpDCCols<T> & A, const SpDCCols<T> & B);

private:
	void CopyDcsc(Dcsc<T> * source);
	SpDCCols<IT,NT> SpDCCols<IT,NT>::ColIndex(const vector<IT> & ci);	//!< col indexing without multiplication	

	SpDCCols<T> OrdOutProdMult(const SpDCCols<T>& rhs) const;	// Ordered outer product multiply
	SpDCCols<T> OrdColByCol(const SpDCCols<T> & rhs) const;		// Ordered column-by-column multiply
	
	SpDCCols (ITYPE size, ITYPE nRow, ITYPE nCol, const vector<IT> & indices, bool isRow);	// Constructor for indexing
	SpDCCols (ITYPE size, ITYPE nRow, ITYPE nCol, Dcsc<T> * mydcsc);			// Constructor for multiplication

	// Private member variables
	Dcsc<IT, NT> * dcsc;

	IT m;
	IT n;
	IT nnz;
	const static IT zero = static_cast<IT>(0);
	const static IT esscount = static_cast<IT>(4);

	//! store a pointer to the memory pool, to transfer it to other matrices returned by functions like Transpose
	MemoryPool * localpool;

	template <class IU, class NU>
	friend class SpTuples;

	template<class IU, class NU1, class NU2, class SR>
	friend SpTuples<IU, promote_trait<NU1,NU2>::T_promote> Tuples_AnXBn (const SpDCCols<IU, NU1> & A, const SpDCCols<IU, NU2> & B, const SR & sring);

	template<class IU, class NU1, class NU2, class SR>
	friend SpTuples<IU, promote_trait<NU1,NU2>::T_promote> Tuples_AnXBt (const SpDCCols<IU, NU1> & A, const SpDCCols<IU, NU2> & B, const SR & sring);

	template<class IU, class NU1, class NU2, class SR>
	friend SpTuples<IU, promote_trait<NU1,NU2>::T_promote> Tuples_AtXBn (const SpDCCols<IU, NU1> & A, const SpDCCols<IU, NU2> & B, const SR & sring);

	template<class IU, class NU1, class NU2, class SR>
	friend SpTuples<IU, promote_trait<NU1,NU2>::T_promote> Tuples_AtXBt (const SpDCCols<IU, NU1> & A, const SpDCCols<IU, NU2> & B, const SR & sring);
};

#include "SpDCCols.cpp"
#endif

