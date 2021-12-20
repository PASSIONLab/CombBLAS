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
#include "CommGrid3D.h"

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
#include "mtSpGEMM.h"
#include "MultiwayMerge.h"
#include "CombBLAS.h"

namespace combblas
{
    
    template <class IT, class NT, class DER>
    class SpParMat3D{
    public:
        typedef typename DER::LocalIT LocalIT;
        typedef typename DER::LocalNT LocalNT;
        typedef IT GlobalIT;
        typedef NT GlobalNT;
        
        // Constructors
        SpParMat3D (int nlayers);
        SpParMat3D (const SpParMat < IT,NT,DER > & A2D, int nlayers, bool colsplit, bool special = false);
        SpParMat3D (DER * myseq, std::shared_ptr<CommGrid3D> grid3d, bool colsplit, bool special = false);
        SpParMat3D (const SpParMat3D <IT,NT,DER> & A3D, bool colsplit);
      
        ~SpParMat3D () ;
        
        SpParMat<IT, NT, DER> Convert2D();

        float LoadImbalance() const;
        void FreeMemory();
        void PrintInfo() const;
        
        IT getnrow() const;
        IT getncol() const;
        IT getnnz() const;
        
        std::shared_ptr< SpParMat<IT, NT, DER> > GetLayerMat() {return layermat;}
        DER * seqptr() const {return layermat->seqptr();}
        bool isSpecial() const {return special;}
        bool isColSplit() const {return colsplit;}

        template <typename LIT>
        int Owner(IT total_m, IT total_n, IT grow, IT gcol, LIT & lrow, LIT & lcol) const;

        void LocalDim(IT total_m, IT total_n, IT &localm, IT& localn) const;

        void CalculateColSplitDistributionOfLayer(vector<typename DER::LocalIT> & divisions3d);
        bool CheckSpParMatCompatibility();       
        std::shared_ptr<CommGrid3D> getcommgrid() const { return commGrid3D; } 	
        std::shared_ptr<CommGrid3D> getcommgrid3D() const {return commGrid3D;}

        /* 3D SUMMA*/
        template <typename SR, typename NUO, typename UDERO, typename IU, typename NU1, typename NU2, typename UDER1, typename UDER2>
        friend SpParMat3D<IU,NUO,UDERO> Mult_AnXBn_SUMMA3D(SpParMat3D<IU,NU1,UDER1> & A, SpParMat3D<IU,NU2,UDER2> & B);
        
        /* Memory efficient 3D SUMMA*/
        template <typename SR, typename NUO, typename UDERO, typename IU, typename NU1, typename NU2, typename UDER1, typename UDER2>
        friend SpParMat3D<IU,NUO,UDERO> MemEfficientSpGEMM3D(SpParMat3D<IU,NU1,UDER1> & A, SpParMat3D<IU,NU2,UDER2> & B,
                int phases, NUO hardThreshold, IU selectNum, IU recoverNum, NUO recoverPct, int kselectVersion, int computationKernel, int64_t perProcessMemory);

    private:
        std::shared_ptr<CommGrid3D> commGrid3D;
        //SpParMat<IT, NT, DER>* layermat;
        std::shared_ptr< SpParMat<IT, NT, DER> > layermat;
        bool colsplit;
        bool special;
        int nlayers;
        
    };
    
    
}

#include "SpParMat3D.cpp"

#endif

