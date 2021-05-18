import os, sys, argparse, logging
from scipy.io import mmread
from scipy.sparse import coo_matrix

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
outfile = args.matrix.replace('.mtx', '.ctf')
if (args.outfile != None):
    outfile = args.outfile

f = open(outfile, 'w')
for (i,j,v) in zip(A.row, A.col, A.data):
    f.write('%d %d %f\n' % (i,j,v))
f.close()

