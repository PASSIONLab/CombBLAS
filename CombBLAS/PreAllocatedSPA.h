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


#ifndef _PRE_ALLOCATED_SPA_H
#define _PRE_ALLOCATED_SPA_H
#include "BitMap.h"

/**
  * This special data structure is used for optimizing BFS iterations
  * by providing a pre-allocated SPA data structure
  */
template <class IT, class NT, class OVT >
class PreAllocatedSPA
{
public:
    PreAllocatedSPA():initialized(false) {};   // hide default constructor

    template <class DER>
    PreAllocatedSPA(SpMat<IT,NT,DER> & A):initialized(true)  // the one and only constructor
	{
        IT mA = A.getnrow();
        if( A.getnsplit() > 0)  // multithreaded
        {
            IT perpiece =  mA / A.getnsplit();
            for(int i=0; i<A.getnsplit(); ++i)
            {
                if(i != A.getnsplit()-1)
                {
                    V_isthere.push_back(BitMap(perpiece));
                    V_localy.push_back(vector<OVT>(perpiece));
                    
                    vector<bool> isthere(perpiece, false);
                    for(typename DER::SpColIter colit = A.begcol(i); colit != A.endcol(i); ++colit)
                    {
                        for(typename DER::SpColIter::NzIter nzit = A.begnz(colit,i); nzit != A.endnz(colit,i); ++nzit)
                        {
                            size_t rowid = nzit.rowid();
                            if(!isthere[rowid])     isthere[rowid] = true;
                        }
                    }
                    size_t maxvector = std::count(isthere.begin(), isthere.end(), true);
                    V_inds.push_back(vector<uint32_t>(maxvector));
                }
                else
                {
                    V_isthere.push_back(BitMap(mA - i*perpiece));
                    V_localy.push_back(vector<OVT>(mA - i*perpiece));
                    
                    vector<bool> isthere(mA - i*perpiece, false);
                    for(typename DER::SpColIter colit = A.begcol(i); colit != A.endcol(i); ++colit)
                    {
                        for(typename DER::SpColIter::NzIter nzit = A.begnz(colit,i); nzit != A.endnz(colit,i); ++nzit)
                        {
                            size_t rowid = nzit.rowid();
                            if(!isthere[rowid])     isthere[rowid] = true;
                        }
                    }
                    size_t maxvector = std::count(isthere.begin(), isthere.end(), true);
                    V_inds.push_back(vector<uint32_t>(maxvector));
                }
            }
        }
        else    // single threaded
        {
            V_isthere.push_back(BitMap(mA));
            V_localy.push_back(vector<OVT>(mA));
            
            vector<bool> isthere(mA, false);
            for(typename DER::SpColIter colit = A.begcol(); colit != A.endcol(); ++colit)
            {
                for(typename DER::SpColIter::NzIter nzit = A.begnz(colit); nzit != A.endnz(colit); ++nzit)
                {
                    size_t rowid = nzit.rowid();
                    if(!isthere[rowid])     isthere[rowid] = true;
                }
            }
            size_t maxvector = std::count(isthere.begin(), isthere.end(), true);
            V_inds.push_back(vector<uint32_t>(maxvector));
             
        }
    };
    
    // for manual splitting. just a hack. need to be fixed
    
    template <class DER>
    PreAllocatedSPA(SpMat<IT,NT,DER> & A, int split):initialized(true)
    {
        IT mA = A.getnrow();
        V_isthere.push_back(BitMap(mA));
        V_localy.push_back(vector<OVT>(mA));
        V_inds.push_back(vector<uint32_t>(mA)); // for better indexing among threads
        V_isthereBool.push_back(vector<bool>(mA));
        
        
        vector<int32_t> nnzSplitA(split,0);
        int32_t rowPerSplit = mA / split;

        //vector<bool> isthere(mA, false);
        for(typename DER::SpColIter colit = A.begcol(); colit != A.endcol(); ++colit)
        {
            for(typename DER::SpColIter::NzIter nzit = A.begnz(colit); nzit != A.endnz(colit); ++nzit)
            {
                size_t rowid = nzit.rowid();
                //if(!isthere[rowid])     isthere[rowid] = true;
                size_t splitId = (rowid/rowPerSplit > split-1) ? split-1 : rowid/rowPerSplit;
                nnzSplitA[splitId]++;
            }
        }
        
        
        // prefix sum
        disp.resize(split+1);
        disp[0] = 0;
        for(int i=0; i<split; i++)
        {
            disp[i+1] = disp[i] + nnzSplitA[i];
        }
        
        indSplitA.resize(disp[split]);
        numSplitA.resize(disp[split]);


    };
    
    
    vector< vector<uint32_t> > V_inds;  // ABAB: is this big enough?
    vector< BitMap > V_isthere;
    vector< vector<bool> > V_isthereBool; // for thread safe access
    vector< vector<OVT> > V_localy;
    bool initialized;
    vector<int32_t> indSplitA;
    vector<OVT> numSplitA;
    vector<uint32_t> disp;
};

#endif

