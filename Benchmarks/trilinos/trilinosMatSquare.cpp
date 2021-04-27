
#include <cstring>
#include <cstdio>
#include <iostream>
#include <fstream>

#include <Epetra_ConfigDefs.h>

#ifdef EPETRA_MPI
#include <mpi.h>
#include <Epetra_MpiComm.h>
#endif

#include <Epetra_SerialComm.h>
#include <Epetra_Time.h>
#include <Epetra_Import.h>
#include <Epetra_Map.h>
#include <Epetra_LocalMap.h>
#include <Epetra_CrsGraph.h>
#include <Epetra_CrsMatrix.h>
#include <Epetra_Vector.h>
#include <EpetraExt_MatrixMatrix.h>

#include <EpetraExt_BlockMapIn.h>
#include <EpetraExt_CrsMatrixIn.h>
#include <EpetraExt_RowMatrixOut.h>


int main(int argc, char** argv) 
{

 if(argc!=2) 
 {
	std::cout << "usage: ./trilinosMatSquare A.mtx\n";
        return (-1);
 }
#ifdef EPETRA_MPI
  MPI_Init(&argc,&argv);
  Epetra_MpiComm Comm(MPI_COMM_WORLD);
#else
  Epetra_SerialComm Comm;
#endif

  char* A_file = argv[1];
  int localProc = Comm.MyPID();

  if (localProc == 0) 
  {
    std::cout << "Squaring Matrix"<< A_file <<  std::endl;
  }

  Epetra_CrsMatrix* A = NULL;
  Epetra_CrsMatrix* B = NULL;
  Epetra_CrsMatrix* C = NULL;

  Epetra_Map* A_row_map = NULL;
  Epetra_Map* A_col_map = NULL;
  Epetra_Map* A_range_map = NULL;
  Epetra_Map* A_domain_map = NULL;


  double tstart = MPI_Wtime();

  // create maps
  int err = EpetraExt::MatrixMarketFileToBlockMaps(A_file,
                                         Comm,
                                         (Epetra_BlockMap*&)A_row_map,
                                         (Epetra_BlockMap*&)A_col_map,
                                         (Epetra_BlockMap*&)A_range_map,
                                         (Epetra_BlockMap*&)A_domain_map);  

  if (err != 0) 
  {
    std::cout << "create_maps A returned err=="<<err<<std::endl;
    return(err);
  }

  // Read the matrix 
  // do not use A_col_map in this function MatrixMarketFileToCrsMatrix
  err = EpetraExt::MatrixMarketFileToCrsMatrix(A_file, *A_row_map, *A_range_map, *A_domain_map, A);
  if (err != 0) 
  {
    std::cout << "Error, MatrixMarketFileToCrsMatrix returned " << err << std::endl;
    return(err);
  }
  B = new Epetra_CrsMatrix(*A); 
  
  double tio = MPI_Wtime() - tstart;
   if (localProc == 0)
   {
     std::cout << "Total I/O time: " << tio << std::endl;
   }

  
  tstart = MPI_Wtime();
  C = new Epetra_CrsMatrix(Copy, A->RowMap(), 1);
  err = EpetraExt::MatrixMatrix::Multiply(*A, false, *B, false, *C);
  
  if (err != 0) 
  {
    std::cout << "err "<<err<<" from MatrixMatrix::Multiply"<<std::endl;
    return(err);
  }

 double tmul = MPI_Wtime() - tstart;
 if (localProc == 0)
 {
     std::cout << "nnz in C=AB: " << C->NumGlobalNonzeros() << std::endl;
     std::cout << "Total multiplication time: " << tmul << std::endl;

 }


 delete C;
 delete A;
 delete B;

 delete A_row_map;
 delete A_col_map;
 delete A_range_map;
 delete A_domain_map;


#ifdef EPETRA_MPI
  MPI_Finalize();
#endif


  return(0);
}



