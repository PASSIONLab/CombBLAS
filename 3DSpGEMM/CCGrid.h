#ifndef _CC_GRID_
#define _CC_GRID_

namespace combblas {

class CCGrid
{
public:
    CCGrid(int c_factor, int gr_cols): GridLayers(c_factor), GridCols(gr_cols), GridRows(gr_cols)
    {
        MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
        MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
        
        layer_grid = myrank % c_factor;         /* RankInFiber = layer_grid, indexed from 1 to c_factor */
        RankInLayer = myrank / c_factor;		/* indexed from 1 to layer_length */
        RankInCol = RankInLayer / GridCols;   	/* RankInCol = MYPROCROW */
        RankInRow = RankInLayer % GridCols;		/* RankInRow = MYPROCCOL */
        
        // MPI_Comm_split(MPI_Comm comm, int color, int key, MPI_Comm *newcomm)
        MPI_Comm_split(MPI_COMM_WORLD, layer_grid, RankInLayer, &layerWorld);
        MPI_Comm_split(MPI_COMM_WORLD, RankInLayer, layer_grid, &fiberWorld);
        
        MPI_Comm_split(MPI_COMM_WORLD, layer_grid * GridRows + RankInLayer / GridRows, RankInRow, &rowWorld);
        MPI_Comm_split(MPI_COMM_WORLD, layer_grid * GridCols + RankInLayer % GridRows, RankInCol, &colWorld);
        
        
#ifdef DEBUG
        printf("Rank %d maps to layer %d (rankinlayer: %d), row %d, and col %d\n", myrank, layer_grid, RankInLayer, RankInCol, RankInRow);
#endif
    };

    int nprocs;
    int myrank;
	int GridRows;
	int GridCols;
    int GridLayers; // GridLayers =  c_factor
	int RankInRow;
	int RankInCol;
	int RankInLayer;
	int layer_grid; // layer_grid = RankInFiber
    MPI_Comm layerWorld;
    MPI_Comm fiberWorld;
    MPI_Comm rowWorld;
    MPI_Comm colWorld;
};

}

#endif
