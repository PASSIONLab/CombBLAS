#ifndef _COMM_GRID_3D_H_
#define _COMM_GRID_3D_H_


using namespace std;

namespace combblas {

class CommGrid3D
{
public:
    CommGrid3D(MPI_Comm world, int nlayers, int nrowproc, int ncolproc): gridLayers(nlayers), gridRows(nrowproc), gridCols(ncolproc)
    {
        
        int nproc;
        MPI_Comm_dup(world, & world3D);
        MPI_Comm_rank(world3D, &myrank);
        MPI_Comm_size(world3D,&nproc);
        
        if(nlayers<1)
        {
            cerr << "A 3D grid can not be created with less than one layer" << endl;
            MPI_Abort(MPI_COMM_WORLD,NOTSQUARE);
        }
        if(nproc % nlayers != 0)
        {
            cerr << "Number of processes is not divisible by number of layers" << endl;
            MPI_Abort(MPI_COMM_WORLD,NOTSQUARE);
        }
        
        int procPerLayer = nproc / nlayers;
        if(gridRows == 0 && gridCols == 0)
        {
            gridRows = (int)std::sqrt((float)procPerLayer);
            gridCols = gridRows;
            
            if(gridRows * gridCols != procPerLayer)
            {
                cerr << "This version of the Combinatorial BLAS only works on a square logical processor grid in a layer of the 3D grid" << endl;
                MPI_Abort(MPI_COMM_WORLD,NOTSQUARE);
            }
        }
        assert((nproc == (gridRows * gridCols * gridLayers)));
        
        
        rankInFiber = myrank % gridLayers;
        int rankInLayer = myrank / gridLayers;
        // MPI_Comm_split(MPI_Comm comm, int color, int key, MPI_Comm *newcomm)
        MPI_Comm layerWorld;
        MPI_Comm_split(world3D, rankInFiber, rankInLayer, &layerWorld);
        MPI_Comm_split(world3D, rankInLayer, rankInFiber, &fiberWorld);
        commGridLayer.reset(new CommGrid(layerWorld, gridRows, gridCols));
/*
#ifdef DEBUG
        printf("Rank %d maps to layer %d (rankinlayer: %d), row %d, and col %d\n", myrank, layer_grid, RankInLayer, RankInCol, RankInRow);
#endif
 */
        
    }
    
    ~CommGrid3D()
    {
        MPI_Comm_free(&world3D);
        MPI_Comm_free(&fiberWorld);
    }
    
    /*
    CCGrid(CommGrid grid2d) // special formula
    {
        
    };
     */

    // Processor grid is (gridLayers x gridRows X gridCols)
    int GetRank(int layerrank, int rowrank, int colrank) { return layerrank * gridRows * gridCols + rowrank * gridCols + colrank; }
    int GetSize() { return gridLayers * gridRows * gridCols; }
    MPI_Comm GetWorld(){return world3D;}
    int myrank;
	int gridRows;
	int gridCols;
    int gridLayers;
    int rankInFiber;
    MPI_Comm world3D;
    MPI_Comm fiberWorld;
    std::shared_ptr<CommGrid> commGridLayer;
};

}

#endif
