import os, sys, argparse, logging
from scipy.io import mmread

sys.path.append(os.environ["PETSC_DIR"]+"/lib/petsc/bin")
import PetscBinaryIO

if __name__=="__main__":
    mtxFile = ""
    petscFile = ""
    for i in range(1, len(sys.argv) ):
        if sys.argv[i] == "--mtx":
            mtxFile = sys.argv[i+1]
        if sys.argv[i] == "--petsc":
            petscFile = sys.argv[i+1]
    print(">>> Reading mtx file")
    A = mmread(mtxFile)
    print("<<< Reading mtx file done")

    print(">>> Writing petsc file")
    PetscBinaryIO.PetscBinaryIO().writeMatSciPy(open(petscFile,'w'), A)
    print("<<< Writing petsc file done")
