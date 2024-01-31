/****************************************************************/
/* Parallel Combinatorial BLAS Library (for Graph Computations) */
/* version 1.6 -------------------------------------------------*/
/* date: 6/15/2017 ---------------------------------------------*/
/* authors: Ariful Azad, Aydin Buluc  --------------------------*/
/****************************************************************/
/*
 Copyright (c) 2010-2017, The Regents of the University of California
 
 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:
 
 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.
 
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE.
 */


#ifndef _SP_PAR_MAT_H_
#define _SP_PAR_MAT_H_

#include <iostream>
#include <fstream>
#include <cmath>
#include <mpi.h>
#include <vector>
#include <iterator>

#include "SpMat.h"
#include "SpTuples.h"
#include "SpDCCols.h"
#include "CommGrid.h"
#include "MPIType.h"
#include "LocArr.h"
#include "SpDefs.h"
#include "Deleter.h"
#include "SpHelper.h"
#include "SpParHelper.h"
#include "DenseParMat.h"
#include "FullyDistVec.h"
#include "Friends.h"
#include "Operations.h"
#include "DistEdgeList.h"
#include "CombBLAS.h"

namespace combblas {

/**
  * Fundamental 2D distributed sparse matrix class
  * The index type IT is encapsulated by the class in a way that it is only
  * guarantee that the implementation will ensure the requested semantics. 
  * For instance, if IT=int64 then the implementation can still use 32 bit 
  * local indices but it should return correct 64-bit numbers in its functions. 
  * In other words, DER can be SpDCCols<int32_t, double> while IT=int64_t
  */
template <class IT, class NT, class DER>
class SpParMat
{
public:	
	typedef typename DER::LocalIT LocalIT;
	typedef typename DER::LocalNT LocalNT;
	typedef IT GlobalIT;
	typedef NT GlobalNT;
	
	// Constructors
	SpParMat ();
    	SpParMat (MPI_Comm world); 	// ABAB: there is risk that any integer would call this constructor due to MPICH representation
	SpParMat (std::shared_ptr<CommGrid> grid);
	SpParMat (DER * myseq, std::shared_ptr<CommGrid> grid);
		
	SpParMat (std::ifstream & input, MPI_Comm & world);
	SpParMat (DER * myseq, MPI_Comm & world);	

	template <class DELIT>
	SpParMat (const DistEdgeList< DELIT > & rhs, bool removeloops = true);	// conversion from distributed edge list

	SpParMat (const SpParMat< IT,NT,DER > & rhs);				// copy constructor
    
	SpParMat (IT total_m, IT total_n, const FullyDistVec<IT,IT> & , const FullyDistVec<IT,IT> & , const FullyDistVec<IT,NT> & , bool SumDuplicates = false);	// matlab sparse
	SpParMat (IT total_m, IT total_n, const FullyDistVec<IT,IT> & , const FullyDistVec<IT,IT> & , const NT & , bool SumDuplicates = false);	// matlab sparse
	SpParMat< IT,NT,DER > & operator=(const SpParMat< IT,NT,DER > & rhs);	// assignment operator
	SpParMat< IT,NT,DER > & operator+=(const SpParMat< IT,NT,DER > & rhs);
	~SpParMat ();

	template <typename SR>
	void Square (); 

	float LoadImbalance() const;
	void Transpose();
	void FreeMemory();
	void EWiseMult (const SpParMat< IT,NT,DER >  & rhs, bool exclude);
	void SetDifference (const SpParMat< IT,NT,DER >  & rhs);
	void EWiseScale (const DenseParMat<IT,NT> & rhs);
	void Find (FullyDistVec<IT,IT> & , FullyDistVec<IT,IT> & , FullyDistVec<IT,NT> & ) const;
	void Find (FullyDistVec<IT,IT> & , FullyDistVec<IT,IT> & ) const;

	DER InducedSubgraphs2Procs(const FullyDistVec<IT,IT>& Assignments, std::vector<IT>& LocalIdxs) const;

	template <typename _BinaryOperation>
	void DimApply(Dim dim, const FullyDistVec<IT, NT>& v, _BinaryOperation __binary_op);

	template <typename _BinaryOperation, typename _UnaryOperation >	
	FullyDistVec<IT,NT> Reduce(Dim dim, _BinaryOperation __binary_op, NT id, _UnaryOperation __unary_op) const;

	template <typename _BinaryOperation>	
	FullyDistVec<IT,NT> Reduce(Dim dim, _BinaryOperation __binary_op, NT id) const;
    
	template <typename VT, typename GIT, typename _BinaryOperation, typename _UnaryOperation >	
	void Reduce(FullyDistVec<GIT,VT> & rvec, Dim dim, _BinaryOperation __binary_op, VT id, _UnaryOperation __unary_op) const;

	template <typename VT, typename GIT, typename _BinaryOperation>	
	void Reduce(FullyDistVec<GIT,VT> & rvec, Dim dim, _BinaryOperation __binary_op, VT id) const;

    template <typename VT, typename GIT>
    bool Kselect(FullyDistVec<GIT,VT> & rvec, IT k_limit, int kselectVersion) const;
    template <typename VT, typename GIT>
    bool Kselect(FullyDistSpVec<GIT,VT> & kth, IT k_limit, int kselectVersion) const; //sparse case
    
    template <typename VT, typename GIT, typename _UnaryOperation>
    bool Kselect1(FullyDistVec<GIT,VT> & rvec, IT k_limit, _UnaryOperation __unary_op) const; // TODO: make private
    template <typename VT, typename GIT, typename _UnaryOperation>
    bool Kselect1(FullyDistSpVec<GIT,VT> & rvec, IT k_limit, _UnaryOperation __unary_op) const; // TODO: make private
    template <typename VT, typename GIT>
    bool Kselect1(FullyDistVec<GIT,VT> & rvec, IT k_limit) const; // TODO: make private
    template <typename VT, typename GIT>
    bool Kselect2(FullyDistVec<GIT,VT> & rvec, IT k_limit) const; // TODO: make private

    IT Bandwidth() const;
    IT Profile() const;
    
    template <typename VT, typename GIT, typename _BinaryOperation>
    void MaskedReduce(FullyDistVec<GIT,VT> & rvec, FullyDistSpVec<GIT,VT> & mask, Dim dim, _BinaryOperation __binary_op, VT id, bool exclude=false) const;
    template <typename VT, typename GIT, typename _BinaryOperation, typename _UnaryOperation >
    void MaskedReduce(FullyDistVec<GIT,VT> & rvec, FullyDistSpVec<GIT,VT> & mask, Dim dim, _BinaryOperation __binary_op, VT id, _UnaryOperation __unary_op, bool exclude=false) const;
    
	template <typename _UnaryOperation>
	void Apply(_UnaryOperation __unary_op)
	{
		spSeq->Apply(__unary_op);	
	}

	IT RemoveLoops();	// returns the number of loops removed
	void AddLoops(NT loopval, bool replaceExisting=false);
    void AddLoops(FullyDistVec<IT,NT> loopvals, bool replaceExisting=false);
	
	template <typename LIT, typename OT>
	void OptimizeForGraph500(OptBuf<LIT,OT> & optbuf);
	void ActivateThreading(int numsplits);	//<! As of version 1.2, only works with boolean matrices 

	template <typename _UnaryOperation>
	SpParMat<IT,NT,DER> PruneI(_UnaryOperation __unary_op, bool inPlace = true) //<! Prune any nonzero entries based on both row/column indices and value
	{
		IT grow=0, gcol=0; 
		GetPlaceInGlobalGrid(grow, gcol);
		if (inPlace)
		{
			spSeq->PruneI(__unary_op, inPlace, grow, gcol);
			return SpParMat<IT,NT,DER>(getcommgrid()); // return blank to match signature
		}
		else
		{
			return SpParMat<IT,NT,DER>(spSeq->PruneI(__unary_op, inPlace, grow, gcol), commGrid);
		}
	}

	template <typename _UnaryOperation>
	SpParMat<IT,NT,DER> Prune(_UnaryOperation __unary_op, bool inPlace = true) //<! Prune any nonzero entries for which the __unary_op evaluates to true (solely based on value)
	{
		if (inPlace)
		{
			spSeq->Prune(__unary_op, inPlace);
			return SpParMat<IT,NT,DER>(getcommgrid()); // return blank to match signature
		}
		else
		{
			return SpParMat<IT,NT,DER>(spSeq->Prune(__unary_op, inPlace), commGrid);
		}
	}

    template <typename _BinaryOperation>
    SpParMat<IT,NT,DER> PruneColumn(const FullyDistVec<IT,NT> & pvals, _BinaryOperation __binary_op, bool inPlace=true);

    template <typename _BinaryOperation>
    SpParMat<IT,NT,DER> PruneColumn(const FullyDistSpVec<IT,NT> & pvals, _BinaryOperation __binary_op, bool inPlace=true);

    template <typename IRRELEVANT_NT>
    void PruneColumnByIndex(const FullyDistSpVec<IT,IRRELEVANT_NT>& ci);

	template <typename _BinaryOperation>
	void UpdateDense(DenseParMat<IT, NT> & rhs, _BinaryOperation __binary_op) const;

	void Dump(std::string filename) const;
	void PrintInfo() const;

	template <typename NNT, typename NDER> operator SpParMat< IT,NNT,NDER > () const;	//!< Type conversion operator
	template <typename NIT, typename NNT, typename NDER> operator SpParMat< NIT,NNT,NDER > () const;	//!< Type conversion operator (for indices as well)

	IT getnrow() const;
	IT getncol() const;
	IT getnnz() const;

    template <typename LIT>
    int Owner(IT total_m, IT total_n, IT grow, IT gcol, LIT & lrow, LIT & lcol) const;
    
	SpParMat<IT,NT,DER> SubsRefCol (const std::vector<IT> & ci) const;				//!< Column indexing with special parallel semantics

	//! General indexing with serial semantics
	template <typename SelectFirstSR, typename SelectSecondSR>
	SpParMat<IT,NT,DER> SubsRef_SR (const FullyDistVec<IT,IT> & ri, const FullyDistVec<IT,IT> & ci, bool inplace=false);

	// Column- or row-only indexing
	template<typename SelectFirstSR,
			 typename SelectSecondSR>
	SpParMat<IT, NT, DER>
	SubsRef_SR (const FullyDistVec<IT, IT> &v, Dim dim, bool inplace = false);
	
	SpParMat<IT,NT,DER> operator() (const FullyDistVec<IT,IT> & ri, const FullyDistVec<IT,IT> & ci, bool inplace=false)
	{
		return SubsRef_SR<BoolCopy1stSRing<NT>, BoolCopy2ndSRing<NT> >(ri, ci, inplace);
	}

	SpParMat<IT, NT, DER>
	operator() (const FullyDistVec<IT, IT> &v, Dim dim, bool inplace = false)
	{
		return SubsRef_SR<BoolCopy1stSRing<NT>,
						  BoolCopy2ndSRing<NT>>(v, dim, inplace);
	}
	
	void Prune(const FullyDistVec<IT,IT> & ri, const FullyDistVec<IT,IT> & ci);	//!< prune all entries whose row indices are in ri AND column indices are in ci
	void PruneFull(const FullyDistVec<IT,IT> & ri, const FullyDistVec<IT,IT> & ci);	//!< prune all entries whose row indices are in ri OR column indices are in ci
	void SpAsgn(const FullyDistVec<IT,IT> & ri, const FullyDistVec<IT,IT> & ci, SpParMat<IT,NT,DER> & B);
	
	bool operator== (const SpParMat<IT,NT,DER> & rhs) const;

	class ScalarReadSaveHandler
	{
	public:
		NT getNoNum(IT row, IT col) { return static_cast<NT>(1); }
		void binaryfill(FILE * rFile, IT & row, IT & col, NT & val) 
		{
			if (fread(&row, sizeof(IT), 1,rFile) != 1)
				std::cout << "binaryfill(): error reading row index" << std::endl;
			if (fread(&col, sizeof(IT), 1,rFile) != 1)
				std::cout << "binaryfill(): error reading col index" << std::endl;
			if (fread(&val, sizeof(NT), 1,rFile) != 1)
				std::cout << "binaryfill(): error reading value" << std::endl;
			return; 
		}
		size_t entrylength() { return 2*sizeof(IT)+sizeof(NT); }
		
		template <typename c, typename t>
		NT read(std::basic_istream<c,t>& is, IT row, IT col)
		{
			NT v;
			is >> v;
			return v;
		}
	
		template <typename c, typename t>
		void save(std::basic_ostream<c,t>& os, const NT& v, IT row, IT col)
		{
			os << v;
		}
	};
	
   	template <typename _BinaryOperation>
    	void ParallelReadMM (const std::string & filename, bool onebased, _BinaryOperation BinOp);
    
    template <class HANDLER>
    void ParallelWriteMM(const std::string & filename, bool onebased, HANDLER handler);
    void ParallelWriteMM(const std::string & filename, bool onebased) { ParallelWriteMM(filename, onebased, ScalarReadSaveHandler()); };

    void ParallelBinaryWrite(std::string filename) const;
    
    template <typename _BinaryOperation>
    FullyDistVec<IT,std::array<char, MAXVERTNAME>> ReadGeneralizedTuples(const std::string&, _BinaryOperation);
    
	template <class HANDLER>
	void ReadDistribute (const std::string & filename, int master, bool nonum, HANDLER handler, bool transpose = false, bool pario = false);
	void ReadDistribute (const std::string & filename, int master, bool nonum=false, bool pario = false) 
	{ 
		ReadDistribute(filename, master, nonum, ScalarReadSaveHandler(), false, pario); 
	}

	template <class HANDLER>
	void SaveGathered(std::string filename, HANDLER handler, bool transpose = false) const;
	void SaveGathered(std::string filename) const { SaveGathered(filename, ScalarReadSaveHandler(), false); }
	
	std::ofstream& put(std::ofstream& outfile) const;

	std::shared_ptr<CommGrid> getcommgrid() const { return commGrid; } 	
	typename DER::LocalIT getlocalrows() const { return spSeq->getnrow(); }
	typename DER::LocalIT getlocalcols() const { return spSeq->getncol();} 
	typename DER::LocalIT getlocalnnz() const { return spSeq->getnnz(); }
	DER & seq() const { return (*spSeq); }
	DER * seqptr() const { return spSeq; }
    
    template <typename _BinaryOperation, typename LIT>
    void SparseCommon(std::vector< std::vector < std::tuple<LIT,LIT,NT> > > & data, LIT locsize, IT total_m, IT total_n, _BinaryOperation BinOp);
    //void SparseCommon(std::vector< std::vector < std::tuple<typename DER::LocalIT,typename DER::LocalIT,NT> > > & data, typename DER::LocalIT locsize, IT total_m, IT total_n, _BinaryOperation BinOp);

	// @TODO-OGUZ allow different index type for blocked matrices
	std::vector<std::vector<SpParMat<IT, NT, DER>>>
	BlockSplit (int br, int bc);
	

	//! Friend declarations
	template <typename SR, typename IU, typename NU1, typename NU2, typename UDER1, typename UDER2> 
	friend IU
	EstimateFLOP (SpParMat<IU,NU1,UDER1> & A, SpParMat<IU,NU2,UDER2> & B, bool clearA, bool clearB);

	template <typename SR, typename NUO, typename UDERO, typename IU, typename NU1, typename NU2, typename UDER1, typename UDER2> 
	friend SpParMat<IU, NUO, UDERO> 
	Mult_AnXBn_DoubleBuff (SpParMat<IU,NU1,UDER1> & A, SpParMat<IU,NU2,UDER2> & B, bool clearA, bool clearB);

	template <typename SR, typename NUO, typename UDERO, typename IU, typename NU1, typename NU2, typename UDER1, typename UDER2> 
	friend SpParMat<IU,NUO,UDERO> 
	Mult_AnXBn_Synch (SpParMat<IU,NU1,UDER1> & A, SpParMat<IU,NU2,UDER2> & B, bool clearA, bool clearB);

	template <typename SR, typename NUO, typename UDERO, typename IU, typename NU1, typename NU2, typename UDER1, typename UDER2> 
	friend SpParMat<IU,NUO,UDERO> 
	Mult_AnXBn_Overlap (SpParMat<IU,NU1,UDER1> & A, SpParMat<IU,NU2,UDER2> & B, bool clearA, bool clearB);
    
    template <typename IU, typename NU1, typename NU2, typename UDERA, typename UDERB>
    friend int64_t EstPerProcessNnzSUMMA(SpParMat<IU,NU1,UDERA> & A, SpParMat<IU,NU2,UDERB> & B, bool hashEstimate);

	template <typename SR, typename IU, typename NU1, typename NU2, typename UDER1, typename UDER2> 
	friend SpParMat<IU,typename promote_trait<NU1,NU2>::T_promote,typename promote_trait<UDER1,UDER2>::T_promote> 
	Mult_AnXBn_ActiveTarget (const SpParMat<IU,NU1,UDER1> & A, const SpParMat<IU,NU2,UDER2> & B );

	template <typename SR, typename IU, typename NU1, typename NU2, typename UDER1, typename UDER2> 
	friend SpParMat<IU,typename promote_trait<NU1,NU2>::T_promote,typename promote_trait<UDER1,UDER2>::T_promote> 
	Mult_AnXBn_PassiveTarget (const SpParMat<IU,NU1,UDER1> & A, const SpParMat<IU,NU2,UDER2> & B );

	template <typename SR, typename IU, typename NU1, typename NU2, typename UDER1, typename UDER2> 
	friend SpParMat<IU,typename promote_trait<NU1,NU2>::T_promote,typename promote_trait<UDER1,UDER2>::T_promote> 
	Mult_AnXBn_Fence (const SpParMat<IU,NU1,UDER1> & A, const SpParMat<IU,NU2,UDER2> & B );
    
	template <typename SR, typename NUO, typename UDERO, typename IU, typename NU1, typename NU2, typename UDER1, typename UDER2> 
	friend SpParMat<IU, NUO, UDERO> 
	Mult_AnXBn_SUMMA (SpParMat<IU,NU1,UDER1> & A, SpParMat<IU,NU2,UDER2> & B, bool clearA, bool clearB);

    template <typename SR, typename NUO, typename UDERO, typename IU, typename NU1, typename NU2, typename UDERA, typename UDERB>
    friend SpParMat<IU,NUO,UDERO> MemEfficientSpGEMM (SpParMat<IU,NU1,UDERA> & A, SpParMat<IU,NU2,UDERB> & B,
                                               int phases, NUO hardThreshold, IU selectNum, IU recoverNum, NUO recoverPct, int kselectVersion, int computationKernel, int64_t perProcessMem);

    template <typename SR, typename ITA, typename NTA, typename DERA>
    friend SpParMat<ITA, NTA, DERA> IncrementalMCLSquare (SpParMat<ITA, NTA, DERA> & A,
                                               int phases, NTA hardThreshold, NTA selectNum, ITA recoverNum, NTA recoverPct, int kselectVersion, int computationKernel, int64_t perProcessMem);

    template <typename SR, typename NUO, typename UDERO, typename IU, typename NU1, typename NU2, typename UDERA, typename UDERB>
    friend int CalculateNumberOfPhases (SpParMat<IU,NU1,UDERA> & A, SpParMat<IU,NU2,UDERB> & B,
                                               NUO hardThreshold, IU selectNum, IU recoverNum, NUO recoverPct, int kselectVersion, int64_t perProcessMem);

	template <typename SR, typename IU, typename NUM, typename NUV, typename UDER> 
	friend FullyDistSpVec<IU,typename promote_trait<NUM,NUV>::T_promote>  
	SpMV (const SpParMat<IU,NUM,UDER> & A, const FullyDistSpVec<IU,NUV> & x );

	template <typename SR, typename IU, typename NUM, typename NUV, typename UDER> 
	friend FullyDistVec<IU,typename promote_trait<NUM,NUV>::T_promote>  
	SpMV (const SpParMat<IU,NUM,UDER> & A, const FullyDistVec<IU,NUV> & x );

	template <typename SR, typename IU, typename NUM, typename UDER> 
	friend FullyDistSpVec<IU,typename promote_trait<NUM,IU>::T_promote>  
	SpMV (const SpParMat<IU,NUM,UDER> & A, const FullyDistSpVec<IU,IU> & x, bool indexisvalue);

	// output type is part of the signature
	template <typename SR, typename IVT, typename OVT, typename IU, typename NUM, typename UDER>
	friend void SpMV (const SpParMat<IU,NUM,UDER> & A, const FullyDistSpVec<IU,IVT> & x, FullyDistSpVec<IU,OVT> & y, bool indexisvalue);
	
	template <typename SR, typename IVT, typename OVT, typename IU, typename NUM, typename UDER>
	friend void SpMV (const SpParMat<IU,NUM,UDER> & A, const FullyDistSpVec<IU,IVT> & x, FullyDistSpVec<IU,OVT> & y,bool indexisvalue, OptBuf<int32_t, OVT > & optbuf);

	template <typename IU, typename NU1, typename NU2, typename UDER1, typename UDER2> 
	friend SpParMat<IU,typename promote_trait<NU1,NU2>::T_promote,typename promote_trait<UDER1,UDER2>::T_promote> 
	EWiseMult (const SpParMat<IU,NU1,UDER1> & A, const SpParMat<IU,NU2,UDER2> & B , bool exclude);

	template <typename RETT, typename RETDER, typename IU, typename NU1, typename NU2, typename UDERA, typename UDERB, typename _BinaryOperation> 
	friend SpParMat<IU,RETT,RETDER>
	EWiseApply (const SpParMat<IU,NU1,UDERA> & A, const SpParMat<IU,NU2,UDERB> & B, _BinaryOperation __binary_op, bool notB, const NU2& defaultBVal);

	template <typename RETT, typename RETDER, typename IU, typename NU1, typename NU2, typename UDERA, typename UDERB, typename _BinaryOperation, typename _BinaryPredicate> 
	friend SpParMat<IU,RETT,RETDER>
	EWiseApply (const SpParMat<IU,NU1,UDERA> & A, const SpParMat<IU,NU2,UDERB> & B, _BinaryOperation __binary_op, _BinaryPredicate do_op, bool allowANulls, bool allowBNulls, const NU1& ANullVal, const NU2& BNullVal, const bool allowIntersect, const bool useExtendedBinOp);

	template<typename SR, typename IVT, typename OVT, typename IU, typename NUM, typename UDER>
	friend void LocalSpMV(const SpParMat<IU,NUM,UDER> & A, int rowneighs, OptBuf<int32_t, OVT > & optbuf, int32_t * & indacc, IVT * & numacc,
                           int32_t * & sendindbuf, OVT * & sendnumbuf, int * & sdispls, int * sendcnt, int accnz, bool indexisvalue, PreAllocatedSPA<OVT> & SPA);

	template<typename VT, typename IU, typename UDER>
	friend void LocalSpMV(const SpParMat<IU,bool,UDER> & A, int rowneighs, OptBuf<int32_t, VT > & optbuf, int32_t * & indacc, VT * & numacc, int * sendcnt, int accnz);

private:
	typedef std::array<char, MAXVERTNAME> STRASARRAY;
	typedef std::pair< STRASARRAY, uint64_t> TYPE2SEND;

	class CharArraySaveHandler
	{
		public:
    		// no reader
    		template <typename c, typename t>
    		void save(std::basic_ostream<c,t>& os, STRASARRAY & chararray, int64_t index)
    		{
			          auto locnull = std::find(chararray.begin(), chararray.end(), '\0'); // find the null character (or string::end)
                std::string strtmp(chararray.begin(), locnull); // range constructor 
			os << strtmp;
    		}
	};
    
	MPI_File TupleRead1stPassNExchange (const std::string & filename, TYPE2SEND * & senddata, IT & totsend, FullyDistVec<IT,STRASARRAY> & distmapper, uint64_t & totallength);

	template <typename VT, typename GIT, typename _BinaryOperation, typename _UnaryOperation >
    	void Reduce(FullyDistVec<GIT,VT> & rvec, Dim dim, _BinaryOperation __binary_op, VT id, _UnaryOperation __unary_op, MPI_Op mympiop) const;
    

    	template <typename VT, typename GIT>	// GIT: global index type of vector
    	void TopKGather(std::vector<NT> & all_medians, std::vector<IT> & nnz_per_col, int & thischunk, int & chunksize,
                    const std::vector<NT> & medians, const std::vector<IT> & nnzperc, int itersuntil, std::vector< std::vector<NT> > & localmat,
                    const std::vector<IT> & actcolsmap, std::vector<IT> & klimits, std::vector<IT> & toretain, std::vector<std::vector<std::pair<IT,NT>>> & tmppair,
                    IT coffset, const FullyDistVec<GIT,VT> & rvec) const;
    
    void GetPlaceInGlobalGrid(IT& rowOffset, IT& colOffset) const;
	
	void HorizontalSend(IT * & rows, IT * & cols, NT * & vals, IT * & temprows, IT * & tempcols, NT * & tempvals, std::vector < std::tuple <IT,IT,NT> > & localtuples,
						int * rcurptrs, int * rdispls, IT buffperrowneigh, int rowneighs, int recvcount, IT m_perproc, IT n_perproc, int rankinrow);
	
        template <class HANDLER>
	void ReadAllMine(FILE * binfile, IT * & rows, IT * & cols, NT * & vals, std::vector< std::tuple<IT,IT,NT> > & localtuples, int * rcurptrs, int * ccurptrs, int * rdispls, int * cdispls, 
			IT m_perproc, IT n_perproc, int rowneighs, int colneighs, IT buffperrowneigh, IT buffpercolneigh, IT entriestoread, HANDLER handler, int rankinrow, bool transpose);

	void VerticalSend(IT * & rows, IT * & cols, NT * & vals, std::vector< std::tuple<IT,IT,NT> > & localtuples, int * rcurptrs, int * ccurptrs, int * rdispls, int * cdispls, 
				IT m_perproc, IT n_perproc, int rowneighs, int colneighs, IT buffperrowneigh, IT buffpercolneigh, int rankinrow);
	
	void AllocateSetBuffers(IT * & rows, IT * & cols, NT * & vals,  int * & rcurptrs, int * & ccurptrs, int rowneighs, int colneighs, IT buffpercolneigh);
	void BcastEssentials(MPI_Comm & world, IT & total_m, IT & total_n, IT & total_nnz, int master);
	
	std::shared_ptr<CommGrid> commGrid; 
	DER * spSeq;
	
	template <class IU, class NU>
	friend class DenseParMat;

	template <typename IU, typename NU, typename UDER> 	
	friend std::ofstream& operator<< (std::ofstream& outfile, const SpParMat<IU,NU,UDER> & s);	
};

template <typename SR, typename NUO, typename UDERO, typename IU, typename NU1, typename NU2, typename UDER1, typename UDER2> 
void PSpGEMM(SpParMat<IU,NU1,UDER1> & A, SpParMat<IU,NU2,UDER2> & B, SpParMat<IU,NUO,UDERO> & out, bool clearA = false, bool clearB = false)
{
	out = Mult_AnXBn_Synch<SR, NUO, UDERO> (A, B, clearA, clearB );
}

template <typename SR, typename IU, typename NU1, typename NU2, typename UDER1, typename UDER2> 
SpParMat<IU,typename promote_trait<NU1,NU2>::T_promote,typename promote_trait<UDER2,UDER2>::T_promote>
	PSpGEMM	(SpParMat<IU,NU1,UDER1> & A, SpParMat<IU,NU2,UDER2> & B, bool clearA = false, bool clearB = false)
{
	typedef typename promote_trait<NU1,NU2>::T_promote N_promote;
	typedef typename promote_trait<UDER1,UDER2>::T_promote DER_promote;
	return Mult_AnXBn_Synch<SR, N_promote, DER_promote> (A, B, clearA, clearB );
}

}



#include "SpParMat.cpp"

#endif
