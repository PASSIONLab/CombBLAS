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

namespace combblas {

/**
  * This special data structure is used for optimizing BFS iterations
  * by providing a pre-allocated SPA data structure
  */

template <class OVT > // output value type
class PreAllocatedSPA
{
public:
    PreAllocatedSPA():initialized(false) {};   // hide default constructor

    template <class LMAT>
    PreAllocatedSPA(LMAT & A):initialized(true)  // the one and only constructor
	{
        int64_t mA = A.getnrow();
        if( A.getnsplit() > 0)  // multithreaded
        {
            int64_t perpiece =  mA / A.getnsplit();
            for(int i=0; i<A.getnsplit(); ++i)
            {
                if(i != A.getnsplit()-1)
                {
                    V_isthere.push_back(BitMap(perpiece));
                    V_localy.push_back(std::vector<OVT>(perpiece));
                    
                    std::vector<bool> isthere(perpiece, false);
                    for(auto colit = A.begcol(i); colit != A.endcol(i); ++colit)
                    {
                        for(auto nzit = A.begnz(colit,i); nzit != A.endnz(colit,i); ++nzit)
                        {
                            size_t rowid = nzit.rowid();
                            if(!isthere[rowid])     isthere[rowid] = true;
                        }
                    }
                    size_t maxvector = std::count(isthere.begin(), isthere.end(), true);
                    V_inds.push_back(std::vector<uint32_t>(maxvector));
                }
                else
                {
                    V_isthere.push_back(BitMap(mA - i*perpiece));
                    V_localy.push_back(std::vector<OVT>(mA - i*perpiece));
                    
                    std::vector<bool> isthere(mA - i*perpiece, false);
                    for(auto colit = A.begcol(i); colit != A.endcol(i); ++colit)
                    {
                        for(auto nzit = A.begnz(colit,i); nzit != A.endnz(colit,i); ++nzit)
                        {
                            size_t rowid = nzit.rowid();
                            if(!isthere[rowid])     isthere[rowid] = true;
                        }
                    }
                    size_t maxvector = std::count(isthere.begin(), isthere.end(), true);
                    V_inds.push_back(std::vector<uint32_t>(maxvector));
                }
            }
        }
        else    // single threaded
        {
            V_isthere.push_back(BitMap(mA));
            V_localy.push_back(std::vector<OVT>(mA));
            
            std::vector<bool> isthere(mA, false);
            for(auto colit = A.begcol(); colit != A.endcol(); ++colit)
            {
                for(auto nzit = A.begnz(colit); nzit != A.endnz(colit); ++nzit)
                {
                    size_t rowid = nzit.rowid();
                    if(!isthere[rowid])     isthere[rowid] = true;
                }
            }
            size_t maxvector = std::count(isthere.begin(), isthere.end(), true);
            V_inds.push_back(std::vector<uint32_t>(maxvector));
             
        }
    };
    
    // for manual splitting. just a hack. need to be fixed
    
    template <class LMAT>
    PreAllocatedSPA(LMAT & A, int splits):initialized(true)
    {
        buckets = splits;
        int64_t mA = A.getnrow();
        V_isthere.push_back(BitMap(mA));
        V_localy.push_back(std::vector<OVT>(mA));
        V_inds.push_back(std::vector<uint32_t>(mA)); // for better indexing among threads
        
       
        
        
        
        std::vector<int32_t> nnzSplitA(buckets,0);
        int32_t rowPerSplit = mA / splits;
        
        
        //per thread because writing vector<bool> is not thread safe
        for(int i=0; i<splits-1; i++)
            V_isthereBool.push_back(std::vector<bool>(rowPerSplit));
         V_isthereBool.push_back(std::vector<bool>(mA - (splits-1)*rowPerSplit));


        //vector<bool> isthere(mA, false);
        for(auto colit = A.begcol(); colit != A.endcol(); ++colit)
        {
            for(auto nzit = A.begnz(colit); nzit != A.endnz(colit); ++nzit)
            {
                size_t rowid = nzit.rowid();
                //if(!isthere[rowid])     isthere[rowid] = true;
                size_t splitId = (rowid/rowPerSplit > splits-1) ? splits-1 : rowid/rowPerSplit;
                nnzSplitA[splitId]++;
            }
        }
        
        
        // prefix sum
        disp.resize(splits+1);
        disp[0] = 0;
        for(int i=0; i<splits; i++)
        {
            disp[i+1] = disp[i] + nnzSplitA[i];
        }
        
        indSplitA.resize(disp[splits]);
        numSplitA.resize(disp[splits]);


    };
    
    // initialize an uninitialized SPA
    template <class LMAT>
    void Init(LMAT & A, int splits) // not done for DCSC matrices with A.getnsplit()
    {
        if(!initialized)
        {
            initialized = true;
            buckets = splits;
            int64_t mA = A.getnrow();
            V_isthere.push_back(BitMap(mA));
            V_localy.push_back(std::vector<OVT>(mA));
            V_inds.push_back(std::vector<uint32_t>(mA)); // for better indexing among threads
            
            std::vector<int32_t> nnzSplitA(buckets,0);
            int32_t rowPerSplit = mA / splits;
            
            for(int i=0; i<splits-1; i++)
                V_isthereBool.push_back(std::vector<bool>(rowPerSplit));
            V_isthereBool.push_back(std::vector<bool>(mA - (splits-1)*rowPerSplit));
            
            //vector<bool> isthere(mA, false);
            for(auto colit = A.begcol(); colit != A.endcol(); ++colit)
            {
                for(auto nzit = A.begnz(colit); nzit != A.endnz(colit); ++nzit)
                {
                    size_t rowid = nzit.rowid();
                    //if(!isthere[rowid])     isthere[rowid] = true;
                    size_t splitId = (rowid/rowPerSplit > splits-1) ? splits-1 : rowid/rowPerSplit;
                    nnzSplitA[splitId]++;
                }
            }
            
            
            // prefix sum
            disp.resize(splits+1);
            disp[0] = 0;
            for(int i=0; i<splits; i++)
            {
                disp[i+1] = disp[i] + nnzSplitA[i];
            }
            
            indSplitA.resize(disp[splits]);
            numSplitA.resize(disp[splits]);
        }
    };
    
    int buckets; // number of buckets
    std::vector< std::vector<uint32_t> > V_inds;  // ABAB: is this big enough?
    std::vector< BitMap > V_isthere;
    std::vector< std::vector<bool> > V_isthereBool; // for thread safe access
    std::vector< std::vector<OVT> > V_localy;
    bool initialized;
    std::vector<int32_t> indSplitA;
    std::vector<OVT> numSplitA;
    std::vector<uint32_t> disp;
};

}

#endif

