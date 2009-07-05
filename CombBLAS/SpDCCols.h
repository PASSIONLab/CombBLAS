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
#include "SpMat.h"	// Best to include the base class first
#include "SpHelper.h"
#include "StackEntry.h"
#include "dcsc.h"
#include "Isect.h"
#include "Semirings.h"
#include "MemoryPool.h"
#include "LocArr.h"


template <class IT, class NT>
class SpDCCols: public SpMat<IT, NT, SpDCCols<IT, NT> >
{
public:
	// Constructors :
	SpDCCols ();
	SpDCCols (IT size, IT nRow, IT nCol, IT nzc);
	SpDCCols (const SpTuples<IT,NT> & rhs, bool transpose, MemoryPool * mpool = NULL);
	SpDCCols (const SpDCCols<IT,NT> & rhs);		// Actual copy constructor		
	~SpDCCols();

	template <typename NNT>
	SpDCCols<IT, NNT> ConvertNumericType();		//!< NNT: New numeric type

	// Member Functions and Operators: 
	SpDCCols<IT,NT> & operator= (const SpDCCols<IT, NT> & rhs);
	SpDCCols<IT,NT> & operator+= (const SpDCCols<IT, NT> & rhs);
	SpDCCols<IT,NT> operator() (const vector<IT> & ri, const vector<IT> & ci) const;

	void ElementWiseMult (const SpDCCols<IT,NT> & rhs, bool exclude);
	void Transpose();				//!< Mutator version, replaces the calling object 
	SpDCCols<IT,NT> TransposeConst() const;		//!< Const version, doesn't touch the existing object

	void Split(SpDCCols<IT,NT> & partA, SpDCCols<IT,NT> & partB); 	//!< \attention Destroys calling object (*this)
	void Merge(SpDCCols<IT,NT> & partA, SpDCCols<IT,NT> & partB);	//!< \attention Destroys its parameters (partA & partB)

	void CreateImpl(const vector<IT> & essentials);
	void CreateImpl(IT size, IT nRow, IT nCol, tuple<IT, IT, NT> * mytuples);

	Arr<IT,NT> GetArrays() const;
	vector<IT> GetEssentials() const;
	const static IT esscount = static_cast<IT>(4);

	bool isZero() const { return (nnz == zero); }
	IT getnrow() const { return m; }
	IT getncol() const { return n; }
	IT getnnz() const { return nnz; }
	
	ofstream& put(ofstream& outfile) const;
	ifstream& get(ifstream& infile);
	void PrintInfo() const;

	template <typename SR> 
	int PlusEq_AtXBt(const SpDCCols<IT,NT> & A, const SpDCCols<IT,NT> & B);  
	
	template <typename SR>
	int PlusEq_AtXBn(const SpDCCols<IT,NT> & A, const SpDCCols<IT,NT> & B);
	
	template <typename SR>
	int PlusEq_AnXBt(const SpDCCols<IT,NT> & A, const SpDCCols<IT,NT> & B);  
	
	template <typename SR>
	int PlusEq_AnXBn(const SpDCCols<IT,NT> & A, const SpDCCols<IT,NT> & B);

private:
	void CopyDcsc(Dcsc<IT,NT> * source);
	SpDCCols<IT,NT> ColIndex(const vector<IT> & ci) const;	//!< col indexing without multiplication	

	template <typename SR, typename NTR>
	SpDCCols< IT, typename promote_trait<NT,NTR>::T_promote > OrdOutProdMult(const SpDCCols<IT,NTR> & rhs) const;	

	template <typename SR, typename NTR>
	SpDCCols< IT, typename promote_trait<NT,NTR>::T_promote > OrdColByCol(const SpDCCols<IT,NTR> & rhs) const;	
	
	SpDCCols (IT size, IT nRow, IT nCol, const vector<IT> & indices, bool isRow);	// Constructor for indexing
	SpDCCols (IT size, IT nRow, IT nCol, Dcsc<IT,NT> * mydcsc);			// Constructor for multiplication

	// Private member variables
	Dcsc<IT, NT> * dcsc;

	IT m;
	IT n;
	IT nnz;
	const static IT zero = static_cast<IT>(0);
	
	//! store a pointer to the memory pool, to transfer it to other matrices returned by functions like Transpose
	MemoryPool * localpool;

	template <class IU, class NU>
	friend class SpDCCols;		// Let other template instantiations (of the same class) access private members
	
	template <class IU, class NU>
	friend class SpTuples;

	template<class SR, class IU, class NU1, class NU2>
	friend SpTuples<IU, typename promote_trait<NU1,NU2>::T_promote> * Tuples_AnXBn 
		(const SpDCCols<IU, NU1> & A, const SpDCCols<IU, NU2> & B);

	template<class SR, class IU, class NU1, class NU2>
	friend SpTuples<IU, typename promote_trait<NU1,NU2>::T_promote> * Tuples_AnXBt 
		(const SpDCCols<IU, NU1> & A, const SpDCCols<IU, NU2> & B);

	template<class SR, class IU, class NU1, class NU2>
	friend SpTuples<IU, typename promote_trait<NU1,NU2>::T_promote> * Tuples_AtXBn 
		(const SpDCCols<IU, NU1> & A, const SpDCCols<IU, NU2> & B);

	template<class SR, class IU, class NU1, class NU2>
	friend SpTuples<IU, typename promote_trait<NU1,NU2>::T_promote> * Tuples_AtXBt 
		(const SpDCCols<IU, NU1> & A, const SpDCCols<IU, NU2> & B);
};

DECLARE_PROMOTE((SpDCCols<int,int>), (SpDCCols<int,int>), (SpDCCols<int,int>));
DECLARE_PROMOTE((SpDCCols<int,int>), (SpDCCols<int,bool>), (SpDCCols<int,int>));
DECLARE_PROMOTE((SpDCCols<int,unsigned>), (SpDCCols<int,bool>), (SpDCCols<int,unsigned>));
DECLARE_PROMOTE((SpDCCols<int,float>), (SpDCCols<int,bool>), (SpDCCols<int,float>));
DECLARE_PROMOTE((SpDCCols<int,double>), (SpDCCols<int,bool>), (SpDCCols<int,double>));
DECLARE_PROMOTE((SpDCCols<int,bool>), (SpDCCols<int,int>), (SpDCCols<int,int>));
DECLARE_PROMOTE((SpDCCols<int,bool>), (SpDCCols<int,unsigned>), (SpDCCols<int,unsigned>));
DECLARE_PROMOTE((SpDCCols<int,bool>), (SpDCCols<int,float>), (SpDCCols<int,float>));
DECLARE_PROMOTE((SpDCCols<int,bool>), (SpDCCols<int,double>), (SpDCCols<int,double>));
DECLARE_PROMOTE((SpDCCols<int,int>), (SpDCCols<int,float>), (SpDCCols<int,float>));
DECLARE_PROMOTE((SpDCCols<int,int>), (SpDCCols<int,double>), (SpDCCols<int,double>));
DECLARE_PROMOTE((SpDCCols<int,float>), (SpDCCols<int,int>), (SpDCCols<int,float>));
DECLARE_PROMOTE((SpDCCols<int,double>), (SpDCCols<int,int>), (SpDCCols<int,double>)); 


#include "SpDCCols.cpp"
#endif

