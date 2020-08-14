
#include <petscmat.h>

static char help[] = "Read in a  matrix in MatrixMarket format. \n\
  Assemble it to a PETSc sparse SEQAIJ matrix. Permute the matrix (if -permute is present)\n\
  Write it in a AIJ matrix (entire matrix) to a file. \n\
  Input parameters are:            \n\
    -fin <filename> : input file   \n\
    -fout <filename> : output file \n\n";

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  Mat            A;
  char           filein[PETSC_MAX_PATH_LEN],fileout[PETSC_MAX_PATH_LEN],buf[PETSC_MAX_PATH_LEN];
  PetscInt       i,m,n,nnz;
  PetscErrorCode ierr;
  PetscMPIInt    size;
  PetscScalar    *val,zero=0.0;
  FILE           *file;
  PetscViewer    view;
  int            *row,*col,*rownz;
  PetscBool      flg, permute;
  char           ordering[256] = MATORDERINGRCM;
  IS             rowperm       = NULL,colperm = NULL;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  if (size > 1) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_USER,"Uniprocessor Example only\n");


 /* Read in matrix*/
  ierr = PetscOptionsGetString(NULL,NULL,"-fin",filein,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_SELF,1,"Must indicate input file with -fin option");

FILE *fp = fopen(filein, "r");
int max_cline = 256;
char cline[max_cline];
fgets(cline, max_cline, fp);   // read first line, chould be something like: %%MatrixMarket matrix coordinate real general
enum MMsym {GENERAL, SYMMETRIC, SKEWSYMMETRIC, HERMITIAN};
enum MMsym s = GENERAL;
if (strstr(cline, "skew-symmetric")) { s = SKEWSYMMETRIC; }
else if (strstr(cline, "symmetric")) { s = SYMMETRIC; }
else if (strstr(cline, "hermitian")) { s = HERMITIAN; }

while (fgets(cline, max_cline, fp)) {
    if (cline[0] != '%') { // first (non comment) line should be: m n nnz
      sscanf(cline, "%d %d %d", &m, &n, &nnz);
      if (s != GENERAL) nnz = 2 * nnz - n;   // only 1 triangle is stored
      break;
    }
}

/* reseve memory for matrices */
  ierr = PetscMalloc4(nnz,&row,nnz,&col,nnz,&val,m,&rownz);CHKERRQ(ierr);
for (i=0; i<m; i++) rownz[i] = 1; /* add 0.0 to diagonal entries */



// you need to guess the number of nonzeros for preallocation
//ierr = MatCreateAIJ(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE,
	//	      m, n, 3*nnz/n/2, NULL, 3*nnz/n/2, NULL, &A); CHKERRQ(ierr); // multiplication by 3 is a hack, need to fix it so that we allocate enough memory for all matrices
//MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
//ierr = MatSetFromOptions(A);
//ierr = MatSetUp(A);

int r, c;
double v;
i=0;
while (fscanf(fp, "%d %d %lf\n", &r, &c, &v) != EOF) {
	row[i] = r-1;
	col[i] = c-1;
	val[i++] = v;
	rownz[r-1]++; 
    	if (r != c) {
    	 switch (s) {
    	 case SKEWSYMMETRIC: 
		row[i] = c-1;
                col[i] = r-1;
                val[i++] = -v;
                rownz[c-1]++;
             	break;
    	 case SYMMETRIC:
		row[i] = c-1;
        	col[i] = r-1;
        	val[i++] = v;
        	rownz[c-1]++; 
             break;
    	 case HERMITIAN:  /*TODO complex*/ break;
    	 default: break;
    }
 }
}


fclose(fp);
  ierr = PetscPrintf(PETSC_COMM_SELF,"Read file completes.\n");CHKERRQ(ierr);
/* Creat and asseble SBAIJ matrix */
  ierr = MatCreate(PETSC_COMM_SELF,&A);CHKERRQ(ierr);
  //ierr = MatSetType(A,MATSBAIJ);CHKERRQ(ierr);
  ierr = MatSetType(A,MATSEQAIJ);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,m,n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  //ierr = MatSeqSBAIJSetPreallocation(A,1,0,rownz);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(A,0,rownz);CHKERRQ(ierr);

  /* Add zero to diagonals, in case the matrix missing diagonals */
  for (i=0; i<m; i++){
    ierr = MatSetValues(A,1,&i,1,&i,&zero,INSERT_VALUES);CHKERRQ(ierr);
  }
  for (i=0; i<nnz; i++) {
    ierr = MatSetValues(A,1,&col[i],1,&row[i],&val[i],INSERT_VALUES);CHKERRQ(ierr);
  }

  ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF,"Assemble SeqAIJ matrix completes.\n");CHKERRQ(ierr);


 ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"Matrix market read and permute example options","");CHKERRQ(ierr);
  {
  permute          = PETSC_FALSE;
  ierr             = PetscOptionsFList("-permute","Permute matrix and vector to solving in new ordering","",MatOrderingList,ordering,ordering,sizeof(ordering),&permute);CHKERRQ(ierr);
  }
ierr = PetscOptionsEnd();CHKERRQ(ierr);
    if (permute) {
    Mat Aperm;
    double t0 = MPI_Wtime();
    ierr = MatGetOrdering(A,ordering,&rowperm,&colperm);CHKERRQ(ierr);
    double t1 = MPI_Wtime();
    ierr = MatPermute(A,rowperm,colperm,&Aperm);CHKERRQ(ierr);
    double t2 = MPI_Wtime();
    ierr = MatDestroy(&A);CHKERRQ(ierr);
    A    = Aperm;               /* Replace original operator with permuted version */
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Permutation is performed. obtaining ordering time: %lf reordering time: %lf\n", t1-t0, t2-t1);CHKERRQ(ierr);
 }

  /* Write the entire matrix in AIJ format to a file */
  ierr = PetscOptionsGetString(NULL,NULL,"-fout",fileout,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscPrintf(PETSC_COMM_SELF,"Write the entire matrix in AIJ format to file %s\n",fileout);CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,fileout,FILE_MODE_WRITE,&view);CHKERRQ(ierr);
    ierr = MatView(A,view);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&view);CHKERRQ(ierr);
  }

  ierr = PetscFree4(row,col,val,rownz);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}

