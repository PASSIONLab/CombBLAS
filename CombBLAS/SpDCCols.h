/****************************************************************/
/* Sequential and Parallel Sparse Matrix Multiplication Library */
/* version 2.3 --------------------------------------------------/
/* date: 01/18/2009 ---------------------------------------------/
/* design detail: AUX array is not created by the constructor, 	*/
/* instead it is generated on demand only for:			*/	
/*		Col Indexing, Algorithm-2			*/
/* author: Aydin Buluc (aydin@cs.ucsb.edu) ----------------------/
/****************************************************************/


#ifndef _SP_DCCOLS_H
#define _SP_DCCOLS_H

#include <iostream>
#include <fstream>
#include <cmath>
#include "SpMat.h"	// Best to include the base class first
#include "SpTuples.h"
#include "SpHelper.h"
#include "StackEntry.h"
#include "dcsc.h"
#include "MMmul.h"
#include "Isect.h"
#include "Semirings.h"
#include "MemoryPool.h"


template <class IT, class NT>
class SpDCCols: public SpMat<IT, NT, SpDCCols<IT, NT> >
{
public:
	// Constructors :
	SpDCCols ();
	SpDCCols (ITYPE size, ITYPE nRow, ITYPE nCol, ITYPE nzc);

	// Copy constructor for converting from SparseTriplets (may use a private memory heap)
	SpDCCols (const SpTuples<IT, NT> & rhs, bool transpose, MemoryPool * mpool = NULL);

	SpDCCols (const SpDCCols<IT, NT> & rhs);		// Actual copy constructor		
	SpDCCols (const MMmul< SpDCCols<IT,NT> > & mmmul);
	~SpDCCols();

	template <typename NNT>
	SpDCCols<IT, NNT> ConvertNumericType();		//!< NNT: New numeric type

	// Member Functions and Operators: 
	SpDCCols<IT,NT> & operator=(const SpDCCols<IT, NT> & rhs);
	SpDCCols<IT,NT> & operator=(const MMmul< SpDCCols<IT, NT> > & mmmul);		//!< Delayed evaluation using compositors
	SpDCCols<IT,NT> & operator+=(const SpDCCols<IT, NT> & rhs);
	SpDCCols<IT,NT> & operator+=(const MMmul< SpDCCols<IT, NT> > & mmmul);		//!< Delayed evaluation using compositors

	SpDCCols<IT,NT> Transpose();			//!< \attention Destroys calling object (*this)
	SpDCCols<IT,NT> TransposeConst() const;		//!< Const version, doesn't touch the existing object

	void Split(SpDCCols<IT,NT> & partA, SpDCCols<IT,NT> & partB); 	//!< \attention Destroys calling object (*this)
	void Merge(SpDCCols<IT,NT> & partA, SpDCCols<IT,NT> & partB);	//!< \attention Destroys its parameters (partA & partB)

	Arr<IT,NT> GetArrays() const
	{
		Arr<IT,NT> arr(3,1);
		arr.indarrs[0] = LocArr(dscs->cp, dcsc->nzc+1);
		arr.indarrs[1] = LocArr(dscs->jc, dcsc->nzc);
		arr.indarrs[2] = LocArr(dscs->ir, dcsc->nz);
		arr.numarrs[0] = LocArr(dscs->num, dcsc->nz);
	}
	void createImpl(vector<IT> & essentials);
	vector<IT> GetEssentials() const
	{
		vector<IT> essentials(4);
		essentials[0] = nnz;
		essentials[1] = m;
		essentials[2] = n;
		essentials[3] = dcsc->nzc;
		return essentials;
	}
		
	bool isZero() { return (nnz == zero); }
	IT getnrow() const { return m; }
	IT getncol() const { return n; }
	IT getnnz() const { return nnz; }
	
	template <typename T2> 
       	SpDCCols<T> Multiply (const SpDCCols<T> & A, const SpDCCols<T2> & B, bool isAT, bool isBT );
	
	SpMat<IT, NT, SpDCCols<IT, NT> > * operator() (const vector<IT> & ri, const vector<IT> & ci) const;

	SpDCCols<T> SubsRefCol(const vector<IT> & ci) const;	//!< col indexing with multiplication

	ofstream& put(ofstream& outfile) const;
	virtual void printInfo();
	SparseMatrix<T, SpDCCols<T> > * ColIndex(const vector<ITYPE> & ci);	//!< col indexing without multiplication


	int PlusEq_AtXBt(const SpDCCols<T> & A, const SpDCCols<T> & B);  
	int PlusEq_AtXBn(const SpDCCols<T> & A, const SpDCCols<T> & B);
	int PlusEq_AnXBt(const SpDCCols<T> & A, const SpDCCols<T> & B);  
	int PlusEq_AnXBn(const SpDCCols<T> & A, const SpDCCols<T> & B);

private:
	template <class IU, class NU>
    	friend class SpMat<IU, NU, SpDCCols<IU, NU> >;		//!< So that the base can call create() or clone()

	// Private Methods
	void CopyDcsc(Dcsc<T> * source);

	SpDCCols<T> OrdOutProdMult(const SpDCCols<T>& rhs) const;	// Ordered outer product multiply
	SpDCCols<T> OrdColByCol(const SpDCCols<T> & rhs) const;		// Ordered column-by-column multiply


	template <typename T2>
	int MultAlg1(const SpDCCols<T2> & rhs, ITYPE & cnz,  Dcsc<T> * & mydcsc) const;
	
	int MultAlg2(const SpDCCols<T> & rhs, ITYPE & cnz,  Dcsc<T> * & mydcsc) const;

	void FillColInds(const vector<ITYPE> & colnums, vector<IPAIR> & colinds) const;

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
	friend StackEntry<promote_trait<NU1,NU2>::T_promote, IU> 
	MultiplyReturnStack (const SpDCCols<IU, NU1> & A, const SpDCCols<IU, NU2> & B, const SR & sring);

};

#include "SpDCCols.cpp"
#endif

