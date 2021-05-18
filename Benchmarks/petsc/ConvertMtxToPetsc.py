import os, sys, argparse, logging
from scipy.io import mmread

# change if you use a different dir
sys.path.append('/opt/cray/pe/petsc/3.11.2.0/real/GNU64/8.2/haswell/lib/petsc/bin')
import PetscBinaryIO

parser = argparse.ArgumentParser()
parser.add_argument('matrix')
parser.add_argument('-o', '--outfile')

args = parser.parse_args()

# logging setup
logging.basicConfig(stream=sys.stdout,
                    format='%(asctime)s ::: %(levelname)s ::: %(filename)s ::: '
                    '%(funcName)s ::: line %(lineno)d ::: %(message)s',
                    level=logging.INFO)

A = mmread(args.matrix)
outfile = args.matrix.replace('.mtx', '.petsc')

if (args.outfile != None):
    outfile = args.outfile

PetscBinaryIO.PetscBinaryIO().writeMatSciPy(open(outfile,'w'), A)
