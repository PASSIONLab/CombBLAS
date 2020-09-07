#include <mpi.h>
#include <sys/time.h> 
#include <iostream>
#include <stdio.h>
#include <functional>
#include <algorithm>
#include <vector>
#include <sstream>
#include <ctime>

#include "../../trunk/CombBLAS/CombBLAS.h"

#include "Node.h"

using namespace std;

class BinaryOp {
  public: 
    Node operator() (Node a, Node b) {
      if (a.dist >= b.dist)
        return Node(b);
      else 
        return Node(a);
    }
};

class DoOp {
  public: 
    bool operator() (Node a, Node b) {
      if (a.dist >= b.dist)
        return false;
      else
        return true;
    }
};

int main(int argc, char* argv[])
{
  MPI::Init(argc, argv);
  int nprocs = MPI::COMM_WORLD.Get_size();
  int myrank = MPI::COMM_WORLD.Get_rank();
  extern MPI_Op staticMPIop;

  {
    // int id, double dist, int parent
    MPI::Datatype types[3] = {MPI::INT, MPI::DOUBLE, MPI::INT};
    int lengths[3] = {1, 1, 1};
    Node n;
    MPI::Aint disp[3];
    disp[0] = MPI::Get_address(&n.id) - MPI::Get_address(&n);
    disp[1] = MPI::Get_address(&n.dist) - MPI::Get_address(&n);
    disp[2] = MPI::Get_address(&n.parent) - MPI::Get_address(&n);

    Node_MPI_datatype = MPI::Datatype::Create_struct(3, lengths, disp, types);
    Node_MPI_datatype.Commit();
    MPI_Op_create(apply, true , &staticMPIop);
  }

  {
    if (argc != 4) {
      cout << endl << "Require 3 args..." << endl <<
        "fileName startV testV" << endl;
      MPI::Finalize();
      return -1;
    }

    char* fileName = argv[1];
    stringstream sstr(argv[2]);
    int startVert;
    sstr >> startVert;
    stringstream sstr2(argv[3]);
    int testVert;
    sstr2 >> testVert;

    if (myrank == 0)
      cout << "startV: " << startVert << endl;

    MPI::COMM_WORLD.Barrier();

    // the graph
    SpParMat<int, double, SpDCCols <int, double> > G;
    G.ReadDistribute(fileName, 0);
    int numVerts = G.getncol();
    if (myrank == 0)
      cout << "numVerts: " << numVerts << endl;

    if (startVert > numVerts || startVert <= 0) {
      cout << "Invalid start vertex id." << endl;
      return -1;
    }

    G.Transpose();
    Node zero(double(0), -1);

    time_t startTime, endTime;
    double elapsedTime;
    if (myrank == 0) {
      startTime = time(NULL);
      cout << "start computing" << endl;
    }

    int iteration;
    bool finished = false;

    FullyDistVec<int, Node> result(G.getcommgrid(), G.getncol(), Node());
    FullyDistSpVec<int, Node> frontier(G.getcommgrid(), G.getncol());

    frontier.SetElement(startVert - 1, zero);
    frontier.setNumToInd();

    BinaryOp binaryOp;
    DoOp doOp;

    frontier = EWiseApply<Node>(frontier, result, binaryOp, doOp, false, Node());
    result.EWiseApply(frontier, binaryOp, false, Node());

    for(iteration = 1; iteration < numVerts; iteration++) {
      frontier = SpMV<SPSRing>(G, frontier);
      frontier.setNumToInd();
      frontier = EWiseApply<Node>(frontier, result, binaryOp, doOp, false, Node());
      if (frontier.getnnz() == 0) {
        finished = true;
        break;
      }
      result.EWiseApply(frontier, binaryOp, false, Node());
    }

    Node res = result[testVert - 1];
    if (myrank == 0) {
      endTime = time(NULL);
      elapsedTime = difftime(endTime, startTime);
      if(finished) {
        cout << "finished" << endl;
        cout << res << endl;
      } else {
        cout << "negative loop" << endl;
      }
      cout << "number of iterations: " << iteration << endl;
      cout << "running time: " << elapsedTime << "s" << endl;
    }
    // G.Transpose();
  }

  MPI::Finalize();
  return 0;
}

