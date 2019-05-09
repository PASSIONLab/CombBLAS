#ifndef _COMM_GRID_3D_H_
#define _COMM_GRID_3D_H_


using namespace std;

namespace combblas {

class CommGrid3D
{
public:
    // Create a special 3D CommGrid with all the processors involved in the given world
    // Parameters:
    //      - A communication world with all the processors with which 3D grid would be made
    //      - Number of layers in the 3D grid
    //      - Number of processors along the row of each layer in the 3D grid
    //      - Number of processors along the column of each layer in the 3D grid
    //      - A flag which will denote if matrix distribution in the comm grid will be in column split manner or row split manner
    //      - A flag which will denote that this CommGrid3D object will be formed in special fashion for communication optimization
    // Information about number of layers, rows and colunmns is being saved to member variables before the execution of this constructor with the help of copy constructors
    CommGrid3D(MPI_Comm world, int nlayers, int nrowproc, int ncolproc, bool colsplit, bool special): gridLayers(nlayers), gridRows(nrowproc), gridCols(ncolproc)
    {
        int nproc;
        MPI_Comm_dup(world, & world3D);
        MPI_Comm_rank(world3D, &myrank);
        // Take the number of processors in the communication world
        MPI_Comm_size(world3D,&nproc);
        
        // Do some error handling depending on the number of available processors with which 3D CommGrid would be formed and
        // number of layer it has been asked to create in it
        if(nlayers<1)
        {
            // If number of intended layers is less than 1 then it is invalid
            cerr << "A 3D grid can not be created with less than one layer" << endl;
            MPI_Abort(MPI_COMM_WORLD,NOTSQUARE);
        }
        if(nproc % nlayers != 0)
        {
            // If number of layers doesn't evenly divide total number of processors in the world then it's invalid
            cerr << "Number of processes is not divisible by number of layers" << endl;
            MPI_Abort(MPI_COMM_WORLD,NOTSQUARE);
        }
        if(((int)std::sqrt((float)nlayers) * (int)std::sqrt((float)nlayers)) != nlayers)
        {
            // Number of layers need to be a square number for this special distribution.
            cerr << "Number of layers is not a square number" << endl;
            MPI_Abort(MPI_COMM_WORLD,NOTSQUARE);
        }
        
        // The execution flow comes here then it passed previous validations.
        // Now calculate number of processors that would be in each layer from previous info
        int procPerLayer = nproc / nlayers;
        // If given number of rows and columns in a layer has not been provided by the caller then we need to calculate that as well
        if(gridRows == 0 && gridCols == 0)
        {
            // Calculate number of rows and columns in a layer by taking square root of previously calculated number of processors in each layer
            gridRows = (int)std::sqrt((float)procPerLayer);
            gridCols = gridRows;
            
            // Do an additional error handling depending on given processors and number of layers whether the layers can be square 2D grid or not
            if(gridRows * gridCols != procPerLayer)
            {
                cerr << "This version of the Combinatorial BLAS only works on a square logical processor grid in a layer of the 3D grid" << endl;
                MPI_Abort(MPI_COMM_WORLD,NOTSQUARE);
            }
        }
        // Another extra layer of assertion if number of layers, rows and columns it covers all the given processors or not
        assert((nproc == (gridRows * gridCols * gridLayers)));

        int nCol2D = (int)std::sqrt((float)nproc);
        int rankInRow2D = myrank / nCol2D;
        int rankInCol2D = myrank % nCol2D;
        int sqrtLayer = (int)std::sqrt((float)nlayers);
        // Determine on which layer does the currently running processor belong
        //if(colsplit) rankInFiber = (rankInCol2D % sqrtLayer) * sqrtLayer + (rankInRow2D % sqrtLayer);
        //else rankInFiber = (rankInRow2D % sqrtLayer) * sqrtLayer + (rankInCol2D % sqrtLayer);
        rankInFiber = (rankInCol2D % sqrtLayer) * sqrtLayer + (rankInRow2D % sqrtLayer);
        // Determine ID of running processor in the scope of it's corresponding layer
        rankInLayer = (rankInRow2D / sqrtLayer) * gridCols + (rankInCol2D / sqrtLayer);
        rankInSpecialWorld = rankInFiber % sqrtLayer;
        // MPI_Comm_split(MPI_Comm comm, int color, int key, MPI_Comm *newcomm)
        // MPI_Allgather call to gather color and key from all participating processors
        // Count the number of processors with the same color; create a communicator with that many processes.
        // Use key to order the ranks
        MPI_Comm_split(world3D, rankInFiber, rankInLayer, &layerWorld);
        MPI_Comm_split(world3D, rankInLayer, rankInFiber, &fiberWorld);
        MPI_Comm_split(fiberWorld, rankInFiber / sqrtLayer, rankInFiber % sqrtLayer, &specialWorld);
        // Create a 2D CommmGrid object corresponding to the layer a processor belongs to
        commGridLayer.reset(new CommGrid(layerWorld, gridRows, gridCols));
        //printf("myrank: %d --> layer: %d layerrank: %d | rankInSpecialWorld: %d\n", myrank, rankInFiber, rankInLayer, rankInSpecialWorld);
    }
    
    ~CommGrid3D()
    {
        MPI_Comm_free(&world3D);
        MPI_Comm_free(&fiberWorld);
        MPI_Comm_free(&layerWorld);
    }
    
    /*
    CCGrid(CommGrid grid2d) // special formula
    {
        
    };
     */

    // Processor grid is (gridLayers x gridRows X gridCols)
    // Function to return rank of a processor in global 2D CommGrid from it's layer, row and column information in 3D CommGrid
    int GetRank(int layerrank, int rowrank, int colrank) { return layerrank * gridRows * gridCols + rowrank * gridCols + colrank; }
    // Function to return total number of processors in total in the 3D processor grid
    int GetSize() { return gridLayers * gridRows * gridCols; }
    // Function to return all the communicators involved in the 3D processor grid
    MPI_Comm GetWorld(){return world3D;}
    MPI_Comm GetFiberWorld(){return fiberWorld;}
    MPI_Comm GetLayerWorld(){return layerWorld;}
    int myrank; // ID of the running processor in the communication world
	int gridRows; // Number of processors along row of each layer in this 3D CommGrid
	int gridCols; // Number of processors along column of each layer in this 3D CommGrid
    int gridLayers; // Number of layers in this 3D CommGrid
    int rankInFiber;
    int rankInLayer;
    int rankInSpecialWorld;
    MPI_Comm world3D;
    MPI_Comm layerWorld;
    MPI_Comm fiberWorld;
    MPI_Comm specialWorld;
    std::shared_ptr<CommGrid> commGridLayer; // 2D CommGrid corresponding to the layer to which running processor belongs
};

}

#endif
