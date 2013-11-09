#include "f2c/f2c.h"
#include "mpidt.h"

// Limitations: Should use all of MPI_COMM_WORLD for now
// until we find a way to solve the interoperability of 
// Fortran integers and the general MPI_Comm type

extern "C" {
  extern int pdsaupd_(integer *comm, integer *ido, char *bmat,
      integer *n, char *which, integer *nev, doublereal *tol, doublereal *
      resid, integer *ncv, doublereal *v, integer *ldv, integer *iparam,
      integer *ipntr, doublereal *workd, doublereal *workl, integer *lworkl,
      integer *info, ftnlen bmat_len, ftnlen which_len);
  extern int pdseupd_(integer *comm, logical *rvec, char *howmny,
      logical *select, doublereal *d__, doublereal *z__, integer *ldz,
      doublereal *sigma, char *bmat, integer *n, char *which, integer *nev,
      doublereal *tol, doublereal *resid, integer *ncv, doublereal *v,
      integer *ldv, integer *iparam, integer *ipntr, doublereal *workd,
      doublereal *workl, integer *lworkl, integer *info, ftnlen howmny_len,
      ftnlen bmat_len, ftnlen which_len);

  // For logging configuration
  extern struct {
    long int logfil, ndigit, mgetv0, msaupd, msaup2, msaitr, mseigt, msapps,
             msgets, mseupd, mnaupd, mnaup2, mnaitr, mneigh, mnapps, mngets,
             mneupd, mcaupd, mcaup2, mcaitr, mceigh, mcapps, mcgets, mceupd;
  } debug_;
}
extern "C" int mpi_allreduce__(doublereal* s, doublereal* r,
integer* c, integer* type, integer* op, integer* comm, integer* err) {
  return MPI_Allreduce(s, r, *c, gettype(*type), getop(*op), MPI_COMM_WORLD);
}

extern "C" int mpi_comm_rank__(integer* comm, integer* myid,
                               integer* error) {
  int id;
  int result = MPI_Comm_rank(MPI_COMM_WORLD, &id);	
  *myid = id;
  return result;
}


typedef SpParMat < int64_t, int, SpDCCols<int32_t,int> > PSpMat_s32p64_Int;

template <class PARMAT>
EigenSolver::EigenSolver(PARMAT A, int nev, int ncv)
    : num_local_row_(num_local_row),
      eigen_space_(eigen_space),
      eigen_desire_(eigen_desire),
      max_iterations_(300),
      tolerance_(0),
      eigen_type_(eigen_type) 
{
	typedef typename PARMAT::GlobalIT GIT;
	typedef typename PARMAT::GlobalNT GNT;

	assert(A.getnrow() == A.getncol());	// symmetry
	GIT glen = A.getncol();
	
	xvec = new FullyDistVec<GIT, GNT> ( A.getcommgrid(), glen, 0);	// identity is 0
	yvec = new FullyDistVec<GIT, GNT> ( A.getcommgrid(), glen, 0);	// identity is 0
	
	int mloc = xvec->MyLocLength();
	int lworkl = ncv * (ncv + 8);		// At least (play with this to increase speed)
  select_ = new long int[ncv];
  resid_ = new double[mloc];
  memset(resid_, 1, sizeof(*resid_) * mloc);
  v_ = new double[mloc * ncv];
  workd_ = new double[3 * mloc];
  workl_ = new double[lworkl];
  d_ = new double[2 * ncv];
  // logging:voluminous.
  debug_.logfil = 6;
  debug_.ndigit = -3;
  debug_.mgetv0 = 0;
  debug_.msaupd = 1;
  debug_.msaup2 = 0;
  debug_.msaitr = 0;
  debug_.mseigt = 0;
  debug_.msapps = 0;
  debug_.msgets = 0;
  debug_.mseupd = 0;
}

EigenSolverSymmetric::~EigenSolverSymmetric() {
  delete[] select_;
  delete[] resid_;
  delete[] v_;
  delete[] workd_;
  delete[] workl_;
  delete[] d_;
}


