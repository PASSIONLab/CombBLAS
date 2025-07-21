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
    // Information about number of layers, rows and colunmns is being saved to member variables
    // before the execution of this constructor with the help of copy constructors
    CommGrid3D(MPI_Comm world, int nlayers, int nrowproc, int ncolproc, bool special = false)
    :gridLayers(nlayers), gridRows(nrowproc), gridCols(ncolproc), special(special)
    {
        int nproc;
        MPI_Comm_dup(world, & world3D);
        MPI_Comm_rank(world3D, & myrank);
        MPI_Comm_size(world3D, & nproc);
        
        if(nlayers<1){
            cerr << "A 3D grid can not be created with less than one layer" << endl;
            MPI_Abort(MPI_COMM_WORLD,NOTSQUARE);
        }
        if(nproc % nlayers != 0){
            cerr << "Number of processes is not divisible by number of layers" << endl;
            MPI_Abort(MPI_COMM_WORLD,NOTSQUARE);
        }
        if(special){
            if(((int)std::sqrt((float)nlayers) * (int)std::sqrt((float)nlayers)) != nlayers){
                // Number of layers need to be a square number for this special distribution.
                cerr << "Number of layers is not a square number" << endl;
                MPI_Abort(MPI_COMM_WORLD,NOTSQUARE);
            }
        }
        
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
        
        if(special){
            int nCol2D = (int)std::sqrt((float)nproc);
            int rankInRow2D = myrank / nCol2D;
            int rankInCol2D = myrank % nCol2D;
            int sqrtLayer = (int)std::sqrt((float)nlayers);
            rankInFiber = (rankInCol2D % sqrtLayer) * sqrtLayer + (rankInRow2D % sqrtLayer);
            rankInLayer = (rankInRow2D / sqrtLayer) * gridCols + (rankInCol2D / sqrtLayer);
            MPI_Comm_split(world3D, rankInFiber, rankInLayer, &layerWorld);
            MPI_Comm_split(world3D, rankInLayer, rankInFiber, &fiberWorld);
        }
        else{
            rankInFiber = myrank / procPerLayer;
            rankInLayer = myrank % procPerLayer;
            MPI_Comm_split(world3D, rankInFiber, rankInLayer, &layerWorld);
            MPI_Comm_split(world3D, rankInLayer, rankInFiber, &fiberWorld);
        }

        commGridLayer.reset(new CommGrid(layerWorld, gridRows, gridCols));
    }
    
    ~CommGrid3D(){
        MPI_Comm_free(&world3D);
        MPI_Comm_free(&fiberWorld);
        MPI_Comm_free(&layerWorld);
    }

    // Processor grid is (gridLayers x gridRows X gridCols)
    // Function to return rank of a processor in global 2D CommGrid from it's layer, row and column information in 3D CommGrid
    int GetRank(int layerrank, int rowrank, int colrank) { 
        if(!special) return layerrank * gridRows * gridCols + rowrank * gridCols + colrank;
        else {
            std::cerr << "for specical cases do something! Now just return the same as non-special one to avoid warning!!!!" << std::endl;
            return layerrank * gridRows * gridCols + rowrank * gridCols + colrank;
        }
    }
    int GetGridLayers() {return gridLayers;}
    int GetGridRows() {return gridRows;}
    int GetGridCols() {return gridCols;}
    int GetSize() { return gridLayers * gridRows * gridCols; }
    bool isSpecial() { return special; }
    MPI_Comm & GetWorld(){return world3D;}
    MPI_Comm & GetFiberWorld(){return fiberWorld;}
    MPI_Comm & GetLayerWorld(){return layerWorld;}
    std::shared_ptr<CommGrid> GetCommGridLayer(){return commGridLayer;}
    int GetRankInWorld(){return myrank;}
    int GetRankInFiber(){return rankInFiber;}
    int GetRankInLayer(){return rankInLayer;}
private:
    bool special;
    int gridRows; // Number of processors along row of each layer in this 3D CommGrid
    int gridCols; // Number of processors along column of each layer in this 3D CommGrid
    int gridLayers; // Number of layers in this 3D CommGrid
    int myrank; // ID of the running processor in the communication world
    int rankInFiber;
    int rankInLayer;
    MPI_Comm world3D;
    MPI_Comm layerWorld;
    MPI_Comm fiberWorld;
    std::shared_ptr<CommGrid> commGridLayer; // 2D CommGrid corresponding to the layer to which running processor belongs
};

}

#endif
