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


#ifndef _SP_TUPLES_H
#define _SP_TUPLES_H

#include <iostream>
#include <fstream>
#include <cmath>
#include <cassert>
#include "CombBLAS.h"
#include "SpMat.h"
#include "SpDefs.h"
#include "StackEntry.h"
#include "Compare.h"

namespace combblas {

template <class IU, class NU>
class SpDCCols;

template <class IU, class NU>
class Dcsc;

template <class IU, class NU>
class SpCCols;

template <class IU, class NU>
class Csc;

/**
 * Triplets are represented using the boost::tuple class of the Boost library
 * Number of entries are 64-bit addressible, but each entry is only <class IT> addressible
 * Therefore, size is int64_t but nrows/ncols (representing range of first two entries in tuple) is of type IT 
 * \remarks Indices start from 0 in this class
 * \remarks Sorted with respect to columns (Column-sorted triples)
 */
template <class IT, class NT>
class SpTuples: public SpMat<IT, NT, SpTuples<IT,NT> >
{
public:
	// Constructors 
	SpTuples (int64_t size, IT nRow, IT nCol);
	SpTuples (int64_t size, IT nRow, IT nCol, std::tuple<IT, IT, NT> * mytuples, bool sorted = false, bool isOpNew = false);
	SpTuples (int64_t maxnnz, IT nRow, IT nCol, std::vector<IT> & edges, bool removeloops = true);	// Graph500 contructor
	SpTuples (int64_t size, IT nRow, IT nCol, StackEntry<NT, std::pair<IT,IT> > * & multstack);
	SpTuples (const SpTuples<IT,NT> & rhs);	 	// Actual Copy constructor
	SpTuples (const SpDCCols<IT,NT> & rhs); 	// Copy constructor for conversion from SpDCCols
	SpTuples (const SpCCols<IT,NT> & rhs);
	~SpTuples();

	SpTuples<IT,NT> & operator=(const SpTuples<IT,NT> & rhs);

	IT & rowindex (IT i) { return joker::get<0>(tuples[i]); }
	IT & colindex (IT i) { return joker::get<1>(tuples[i]); }
	NT & numvalue (IT i) { return joker::get<2>(tuples[i]); }

	IT rowindex (IT i) const { return joker::get<0>(tuples[i]); }
	IT colindex (IT i) const { return joker::get<1>(tuples[i]); } 
	NT numvalue (IT i) const { return joker::get<2>(tuples[i]); } 


	template <typename BINFUNC>
	void RemoveDuplicates(BINFUNC BinOp);
	
	void SortRowBased()
	{
		RowLexiCompare<IT,NT> rowlexicogcmp;
		if(!SpHelper::is_sorted(tuples, tuples+nnz, rowlexicogcmp))
			sort(tuples , tuples+nnz, rowlexicogcmp);	

		// Default "operator<" for tuples uses lexicographical ordering 
		// However, cray compiler complains about it, so we use rowlexicogcmp
	}

	void SortColBased()
	{
		ColLexiCompare<IT,NT> collexicogcmp;
		if(!SpHelper::is_sorted(tuples, tuples+nnz, collexicogcmp))
			sort(tuples , tuples+nnz, collexicogcmp );
	}

	/**
	 * @pre {should only be called on diagonal processors (others will add non-loop nonzeros)}
	 * @pre {both the implementation and its semantics is meaningless for non-square matrices}
	 **/
	IT AddLoops(NT loopval, bool replaceExisting=false)
	{
		std::vector<bool> existing(n,false);	// none of the diagonals exist	
		IT loop = 0;
		for(IT i=0; i< nnz; ++i)
		{
			if(joker::get<0>(tuples[i]) == joker::get<1>(tuples[i])) 
			{	
				++loop;
				existing[joker::get<0>(tuples[i])] = true;
                if(replaceExisting)
                    joker::get<2>(tuples[i]) = loopval;
			}
		}
		std::vector<IT> missingindices;
		for(IT i = 0; i < n; ++i)
		{
			if(!existing[i])	missingindices.push_back(i);
		}
		IT toadd = n - loop;	// number of new entries needed (equals missingindices.size())
		std::tuple<IT, IT, NT> * ntuples = new std::tuple<IT,IT,NT>[nnz+toadd];

    std::copy(tuples,tuples+nnz, ntuples);
		
		// MCL: As for the loop weights that are chosen, experience shows that a neutral value works well. It is possible to choose larger weights, 
		// and this will increase cluster granularity. The effect is secondary however to that of varying the inflation parameter, 
		// and the algorithm is not very sensitive to changes in the loop weights.
		for(IT i=0; i< toadd; ++i)
		{
			ntuples[nnz+i] = std::make_tuple(missingindices[i], missingindices[i], loopval);
		}
        if(isOperatorNew)
            ::operator delete(tuples);
        else
            delete [] tuples;
		tuples = ntuples;
        isOperatorNew = false;
		nnz = nnz+toadd;
        
		return loop;
	}
    
    
    /**
     * @pre {should only be called on diagonal processors (others will add non-loop nonzeros)}
     * @pre {both the implementation and its semantics is meaningless for non-square matrices}
     **/
    IT AddLoops(std::vector<NT> loopvals, bool replaceExisting=false)
    {
        // expectation n == loopvals.size())
        
        std::vector<bool> existing(n,false);	// none of the diagonals exist
        IT loop = 0;
        for(IT i=0; i< nnz; ++i)
        {
            if(joker::get<0>(tuples[i]) == joker::get<1>(tuples[i]))
            {
                ++loop;
                existing[joker::get<0>(tuples[i])] = true;
                if(replaceExisting)
                    joker::get<2>(tuples[i]) = loopvals[joker::get<0>(tuples[i])];
            }
        }
        std::vector<IT> missingindices;
        for(IT i = 0; i < n; ++i)
        {
            if(!existing[i])	missingindices.push_back(i);
        }
        IT toadd = n - loop;	// number of new entries needed (equals missingindices.size())
        std::tuple<IT, IT, NT> * ntuples = new std::tuple<IT,IT,NT>[nnz+toadd];
        
        std::copy(tuples,tuples+nnz, ntuples);
        
        for(IT i=0; i< toadd; ++i)
        {
            ntuples[nnz+i] = std::make_tuple(missingindices[i], missingindices[i], loopvals[missingindices[i]]);
        }
        if(isOperatorNew)
            ::operator delete(tuples);
        else
            delete [] tuples;
        tuples = ntuples;
        isOperatorNew = false;
        nnz = nnz+toadd;
        return loop;
    }

	/**
	 *  @pre {should only be called on diagonal processors (others will remove non-loop nonzeros)}
	 **/
	IT RemoveLoops()
	{
		IT loop = 0;
		for(IT i=0; i< nnz; ++i)
		{
			if(joker::get<0>(tuples[i]) == joker::get<1>(tuples[i])) ++loop;
		}
		std::tuple<IT, IT, NT> * ntuples = new std::tuple<IT,IT,NT>[nnz-loop];

		IT ni = 0;
		for(IT i=0; i< nnz; ++i)
		{
			if(joker::get<0>(tuples[i]) != joker::get<1>(tuples[i])) 
			{
				ntuples[ni++] = tuples[i];
			}
		}
        if(isOperatorNew)
            ::operator delete(tuples);
        else
            delete [] tuples;
        tuples = ntuples;
        isOperatorNew = false;
		nnz = nnz-loop;
		return loop;
	}

	std::pair<IT,IT> RowLimits()
	{
		if(nnz > 0)
		{	
			RowCompare<IT,NT> rowcmp;
			std::tuple<IT,IT,NT> * maxit = std::max_element(tuples, tuples+nnz, rowcmp);	
			std::tuple<IT,IT,NT> * minit = std::min_element(tuples, tuples+nnz, rowcmp);
			return std::make_pair(joker::get<0>(*minit), joker::get<0>(*maxit));
		}
		else
			return std::make_pair(0,0);
	}
	std::pair<IT,IT> ColLimits()
	{	
		if(nnz > 0)
		{
			ColCompare<IT,NT> colcmp;
			std::tuple<IT,IT,NT> * maxit = std::max_element(tuples, tuples+nnz, colcmp);
			std::tuple<IT,IT,NT> * minit = std::min_element(tuples, tuples+nnz, colcmp);
			return std::make_pair(joker::get<1>(*minit), joker::get<1>(*maxit));
		}
		else
			return std::make_pair(0,0);
	}
	std::tuple<IT, IT, NT> front() { return tuples[0]; };
	std::tuple<IT, IT, NT> back() { return tuples[nnz-1]; };

	// Performs a balanced merge of the array of SpTuples
	template<typename SR, typename IU, typename NU>
	friend SpTuples<IU,NU> MergeAll(const std::vector<SpTuples<IU,NU> *> & ArrSpTups, IU mstar, IU nstar, bool delarrs); 

	template<typename SR, typename IU, typename NU>
	friend SpTuples<IU,NU> * MergeAllRec(const std::vector<SpTuples<IU,NU> *> & ArrSpTups, IU mstar, IU nstar); 
	
	std::ofstream& putstream (std::ofstream& outfile) const;
    std::ofstream& put (std::ofstream& outfile) const
    { return putstream(outfile); }

	std::ifstream& getstream (std::ifstream& infile);
    std::ifstream& get (std::ifstream& infile) { return getstream(infile); }


	bool isZero() const { return (nnz == 0); }	
	IT getnrow() const { return m; }
	IT getncol() const { return n; }
	int64_t getnnz() const { return nnz; }

	void PrintInfo();
    std::tuple<IT, IT, NT> * tuples; 	
    bool tuples_deleted = false; // This is a temporary patch to avoid memory leak in 3d-memory multiplication

private:

	IT m;
	IT n;
	int64_t nnz;
    bool isOperatorNew; // if Operator New was used to allocate memory

	SpTuples (){};		// Default constructor does nothing, hide it
	
	void FillTuples (Dcsc<IT,NT> * mydcsc);

	template <class IU, class NU>
	friend class SpDCCols;
    
    template <class IU, class NU>
    friend class SpCCols;
};


// At this point, complete type of of SpTuples is known, safe to declare these specialization (but macros won't work as they are preprocessed)
template <> struct promote_trait< SpTuples<int,int> , SpTuples<int,int> >       
    {                                           
        typedef SpTuples<int,int> T_promote;                    
    };
template <> struct promote_trait< SpTuples<int,float> , SpTuples<int,float> >       
    {                                           
        typedef SpTuples<int,float> T_promote;                    
    };
template <> struct promote_trait< SpTuples<int,double> , SpTuples<int,double> >       
    {                                           
        typedef SpTuples<int,double> T_promote;                    
    };
template <> struct promote_trait< SpTuples<int,bool> , SpTuples<int,int> >       
    {                                           
        typedef SpTuples<int,int> T_promote;                    
    };
template <> struct promote_trait< SpTuples<int,int> , SpTuples<int,bool> >       
    {                                           
        typedef SpTuples<int,int> T_promote;                    
    };
template <> struct promote_trait< SpTuples<int,int> , SpTuples<int,float> >       
    {                                           
        typedef SpTuples<int,float> T_promote;                    
    };
template <> struct promote_trait< SpTuples<int,float> , SpTuples<int,int> >       
    {                                           
        typedef SpTuples<int,float> T_promote;                    
    };
template <> struct promote_trait< SpTuples<int,int> , SpTuples<int,double> >       
    {                                           
        typedef SpTuples<int,double> T_promote;                    
    };
template <> struct promote_trait< SpTuples<int,double> , SpTuples<int,int> >       
    {                                           
        typedef SpTuples<int,double> T_promote;                    
    };
template <> struct promote_trait< SpTuples<int,unsigned> , SpTuples<int,bool> >       
    {                                           
        typedef SpTuples<int,unsigned> T_promote;                    
    };
template <> struct promote_trait< SpTuples<int,bool> , SpTuples<int,unsigned> >       
    {                                           
        typedef SpTuples<int,unsigned> T_promote;                    
    };
template <> struct promote_trait< SpTuples<int,bool> , SpTuples<int,double> >       
    {                                           
        typedef SpTuples<int,double> T_promote;                    
    };
template <> struct promote_trait< SpTuples<int,bool> , SpTuples<int,float> >       
    {                                           
        typedef SpTuples<int,float> T_promote;                    
    };
template <> struct promote_trait< SpTuples<int,double> , SpTuples<int,bool> >       
    {                                           
        typedef SpTuples<int,double> T_promote;                    
    };
template <> struct promote_trait< SpTuples<int,float> , SpTuples<int,bool> >       
    {                                           
        typedef SpTuples<int,float> T_promote;                    
    };
template <> struct promote_trait< SpTuples<int64_t,int> , SpTuples<int64_t,int> >       
    {                                           
        typedef SpTuples<int64_t,int> T_promote;                    
    };
template <> struct promote_trait< SpTuples<int64_t,float> , SpTuples<int64_t,float> >       
    {                                           
        typedef SpTuples<int64_t,float> T_promote;                    
    };
template <> struct promote_trait< SpTuples<int64_t,double> , SpTuples<int64_t,double> >       
    {                                           
        typedef SpTuples<int64_t,double> T_promote;                    
    };
template <> struct promote_trait< SpTuples<int64_t,int64_t> , SpTuples<int64_t,int64_t> >       
    {                                           
        typedef SpTuples<int64_t,int64_t> T_promote;                    
    };
template <> struct promote_trait< SpTuples<int64_t,bool> , SpTuples<int64_t,int> >       
    {                                           
        typedef SpTuples<int64_t,int> T_promote;                    
    };
template <> struct promote_trait< SpTuples<int64_t,int> , SpTuples<int64_t,bool> >       
    {                                           
        typedef SpTuples<int64_t,int> T_promote;                    
    };
template <> struct promote_trait< SpTuples<int64_t,int> , SpTuples<int64_t,float> >       
    {                                           
        typedef SpTuples<int64_t,float> T_promote;                    
    };
template <> struct promote_trait< SpTuples<int64_t,float> , SpTuples<int64_t,int> >       
    {                                           
        typedef SpTuples<int64_t,float> T_promote;                    
    };
template <> struct promote_trait< SpTuples<int64_t,int> , SpTuples<int64_t,double> >       
    {                                           
        typedef SpTuples<int64_t,double> T_promote;                    
    };
template <> struct promote_trait< SpTuples<int64_t,double> , SpTuples<int64_t,int> >       
    {                                           
        typedef SpTuples<int64_t,double> T_promote;                    
    };
template <> struct promote_trait< SpTuples<int64_t,unsigned> , SpTuples<int64_t,bool> >       
    {                                           
        typedef SpTuples<int64_t,unsigned> T_promote;                    
    };
template <> struct promote_trait< SpTuples<int64_t,bool> , SpTuples<int64_t,unsigned> >       
    {                                           
        typedef SpTuples<int64_t,unsigned> T_promote;                    
    };
template <> struct promote_trait< SpTuples<int64_t,bool> , SpTuples<int64_t,double> >       
    {                                           
        typedef SpTuples<int64_t,double> T_promote;                    
    };
template <> struct promote_trait< SpTuples<int64_t,bool> , SpTuples<int64_t,float> >       
    {                                           
        typedef SpTuples<int64_t,float> T_promote;                    
    };
template <> struct promote_trait< SpTuples<int64_t,double> , SpTuples<int64_t,bool> >       
    {                                           
        typedef SpTuples<int64_t,double> T_promote;                    
    };
template <> struct promote_trait< SpTuples<int64_t,float> , SpTuples<int64_t,bool> >       
    {                                           
        typedef SpTuples<int64_t,float> T_promote;                    
    };
}

#include "SpTuples.cpp"	

#endif
