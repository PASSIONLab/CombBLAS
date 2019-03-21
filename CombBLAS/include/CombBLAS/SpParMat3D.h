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


#ifndef _SP_PAR_MAT_3D_H_
#define _SP_PAR_MAT_3D_H_

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
//#include "CommGrid3D.h"

#include "MPIType.h"
#include "LocArr.h"
#include "SpDefs.h"
#include "Deleter.h"
#include "SpHelper.h"
#include "SpParHelper.h"
#include "FullyDistVec.h"
#include "Friends.h"
#include "Operations.h"
#include "DistEdgeList.h"
#include "CombBLAS.h"

namespace combblas
{
    
    template <class IT, class NT, class DER>
    class SpParMat3D
    {
    public:
        typedef typename DER::LocalIT LocalIT;
        typedef typename DER::LocalNT LocalNT;
        typedef IT GlobalIT;
        typedef NT GlobalNT;
        
        // Constructors
        SpParMat3D ();
        SpParMat3D (const SpParMat< IT,NT,DER > & A2D, int nlayers, bool colsplit); // 2D to 3D converter
      
        SpParMat<IT, NT, DER> Convert2D();
        //~SpParMat3D () ;
        
        
        float LoadImbalance() const;
        void FreeMemory();
        void PrintInfo() const;
        
        
        IT getnrow() const;
        IT getncol() const;
        IT getnnz() const;
        
        template <typename LIT>
        int Owner(IT total_m, IT total_n, IT grow, IT gcol, LIT & lrow, LIT & lcol) const;
        void LocalDim(IT total_m, IT total_n, IT &localm, IT& localn) const;
        
        std::shared_ptr<CommGrid3D> getcommgrid3D() const { return commGrid3D; }
       // DER & seq() { return (*spSeq); }
        //DER * seqptr() { return spSeq; }
        
    private:
        
        std::shared_ptr<CommGrid3D> commGrid3D;
        SpParMat<IT, NT, DER>* layermat;
        bool colsplit;
        
        
    };
    
    
}

#include "SpParMat3D.cpp"

#endif

