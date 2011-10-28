/* psgetv0.f -- translated by f2c (version 20050501).
   You must link the resulting object file with libf2c:
	on Microsoft Windows system, link with libf2c.lib;
	on Linux or Unix systems, link with .../path/to/libf2c.a -lm
	or, if you install libf2c.a in a standard place, with -lf2c -lm
	-- in that order, at the end of the command line, as in
		cc *.o -lf2c -lm
	Source for libf2c is in /netlib/f2c/libf2c.zip, e.g.,

		http://www.netlib.org/f2c/libf2c.zip
*/

#include "f2c.h"

/* Common Block Declarations */

struct {
    integer mpi_bottom__, mpi_integer__, mpi_real__, mpi_double_precision__, 
	    mpi_complex__, mpi_double_complex__, mpi_logical__, 
	    mpi_character__, mpi_byte__, mpi_2integer__, mpi_2real__, 
	    mpi_2double_precision__, mpi_2complex__, mpi_2double_complex__, 
	    mpi_integer1__, mpi_integer2__, mpi_integer4__, mpi_real2__, 
	    mpi_real4__, mpi_real8__, mpi_ub__, mpi_lb__, mpi_comm_world__, 
	    mpi_comm_self__, mpi_group_empty__, mpi_sum__, mpi_max__, 
	    mpi_min__, mpi_prod__, mpi_land__, mpi_band__, mpi_lor__, 
	    mpi_bor__, mpi_lxor__, mpi_bxor__, mpi_minloc__, mpi_maxloc__, 
	    mpi_op_null__, mpi_tag_ub__, mpi_host__, mpi_io__, 
	    mpi_errors_are_fatal__, mpi_errors_return__, mpi_packed__;
} mpipriv_;

#define mpipriv_1 mpipriv_

struct {
    integer logfil, ndigit, mgetv0, msaupd, msaup2, msaitr, mseigt, msapps, 
	    msgets, mseupd, mnaupd, mnaup2, mnaitr, mneigh, mnapps, mngets, 
	    mneupd, mcaupd, mcaup2, mcaitr, mceigh, mcapps, mcgets, mceupd;
} debug_;

#define debug_1 debug_

struct {
    integer nopx, nbx, nrorth, nitref, nrstrt;
    real tsaupd, tsaup2, tsaitr, tseigt, tsgets, tsapps, tsconv, tnaupd, 
	    tnaup2, tnaitr, tneigh, tngets, tnapps, tnconv, tcaupd, tcaup2, 
	    tcaitr, tceigh, tcgets, tcapps, tcconv, tmvopx, tmvbx, tgetv0, 
	    titref, trvec;
} timing_;

#define timing_1 timing_

/* Table of constant values */

static integer c__1 = 1;
static real c_b24 = 1.f;
static real c_b26 = 0.f;
static real c_b29 = -1.f;

/* ----------------------------------------------------------------------- */
/* \BeginDoc */

/* \Name: psgetv0 */

/* Message Passing Layer: MPI */

/* \Description: */
/*  Generate a random initial residual vector for the Arnoldi process. */
/*  Force the residual vector to be in the range of the operator OP. */

/* \Usage: */
/*  call psgetv0 */
/*     ( COMM, IDO, BMAT, ITRY, INITV, N, J, V, LDV, RESID, RNORM, */
/*       IPNTR, WORKD, WORKL, IERR ) */

/* \Arguments */
/*  COMM    MPI Communicator for the processor grid.  (INPUT) */

/*  IDO     Integer.  (INPUT/OUTPUT) */
/*          Reverse communication flag.  IDO must be zero on the first */
/*          call to psgetv0. */
/*          ------------------------------------------------------------- */
/*          IDO =  0: first call to the reverse communication interface */
/*          IDO = -1: compute  Y = OP * X  where */
/*                    IPNTR(1) is the pointer into WORKD for X, */
/*                    IPNTR(2) is the pointer into WORKD for Y. */
/*                    This is for the initialization phase to force the */
/*                    starting vector into the range of OP. */
/*          IDO =  2: compute  Y = B * X  where */
/*                    IPNTR(1) is the pointer into WORKD for X, */
/*                    IPNTR(2) is the pointer into WORKD for Y. */
/*          IDO = 99: done */
/*          ------------------------------------------------------------- */

/*  BMAT    Character*1.  (INPUT) */
/*          BMAT specifies the type of the matrix B in the (generalized) */
/*          eigenvalue problem A*x = lambda*B*x. */
/*          B = 'I' -> standard eigenvalue problem A*x = lambda*x */
/*          B = 'G' -> generalized eigenvalue problem A*x = lambda*B*x */

/*  ITRY    Integer.  (INPUT) */
/*          ITRY counts the number of times that psgetv0 is called. */
/*          It should be set to 1 on the initial call to psgetv0. */

/*  INITV   Logical variable.  (INPUT) */
/*          .TRUE.  => the initial residual vector is given in RESID. */
/*          .FALSE. => generate a random initial residual vector. */

/*  N       Integer.  (INPUT) */
/*          Dimension of the problem. */

/*  J       Integer.  (INPUT) */
/*          Index of the residual vector to be generated, with respect to */
/*          the Arnoldi process.  J > 1 in case of a "restart". */

/*  V       Real N by J array.  (INPUT) */
/*          The first J-1 columns of V contain the current Arnoldi basis */
/*          if this is a "restart". */

/*  LDV     Integer.  (INPUT) */
/*          Leading dimension of V exactly as declared in the calling */
/*          program. */

/*  RESID   Real array of length N.  (INPUT/OUTPUT) */
/*          Initial residual vector to be generated.  If RESID is */
/*          provided, force RESID into the range of the operator OP. */

/*  RNORM   Real scalar.  (OUTPUT) */
/*          B-norm of the generated residual. */

/*  IPNTR   Integer array of length 3.  (OUTPUT) */

/*  WORKD   Real work array of length 2*N.  (REVERSE COMMUNICATION). */
/*          On exit, WORK(1:N) = B*RESID to be used in SSAITR. */

/*  WORKL   Real work space used for Gram Schmidt orthogonalization */

/*  IERR    Integer.  (OUTPUT) */
/*          =  0: Normal exit. */
/*          = -1: Cannot generate a nontrivial restarted residual vector */
/*                in the range of the operator OP. */

/* \EndDoc */

/* ----------------------------------------------------------------------- */

/* \BeginLib */

/* \Local variables: */
/*     xxxxxx  real */

/* \References: */
/*  1. D.C. Sorensen, "Implicit Application of Polynomial Filters in */
/*     a k-Step Arnoldi Method", SIAM J. Matr. Anal. Apps., 13 (1992), */
/*     pp 357-385. */
/*  2. R.B. Lehoucq, "Analysis and Implementation of an Implicitly */
/*     Restarted Arnoldi Iteration", Rice University Technical Report */
/*     TR95-13, Department of Computational and Applied Mathematics. */

/* \Routines called: */
/*     second   ARPACK utility routine for timing. */
/*     psvout   Parallel ARPACK utility routine for vector output. */
/*     pslarnv  Parallel wrapper for LAPACK routine slarnv (generates a random vector). */
/*     sgemv    Level 2 BLAS routine for matrix vector multiplication. */
/*     scopy    Level 1 BLAS that copies one vector to another. */
/*     sdot     Level 1 BLAS that computes the scalar product of two vectors. */
/*     psnorm2  Parallel version of  Level 1 BLAS that computes the norm of a vector. */

/* \Author */
/*     Danny Sorensen               Phuong Vu */
/*     Richard Lehoucq              Cray Research, Inc. & */
/*     Dept. of Computational &     CRPC / Rice University */
/*     Applied Mathematics          Houston, Texas */
/*     Rice University */
/*     Houston, Texas */

/* \Parallel Modifications */
/*     Kristi Maschhoff */

/* \Revision history: */
/*     Starting Point: Serial Code FILE: getv0.F   SID: 2.3 */

/* \SCCS Information: */
/* FILE: getv0.F   SID: 1.4   DATE OF SID: 3/19/97 */

/* \EndLib */

/* ----------------------------------------------------------------------- */

/* Subroutine */ int psgetv0_(integer *comm, integer *ido, char *bmat, 
	integer *itry, logical *initv, integer *n, integer *j, real *v, 
	integer *ldv, real *resid, real *rnorm, integer *ipntr, real *workd, 
	real *workl, integer *ierr, ftnlen bmat_len)
{
    /* Initialized data */

    static logical inits = TRUE_;

    /* System generated locals */
    integer v_dim1, v_offset, i__1;

    /* Builtin functions */
    double sqrt(doublereal);

    /* Local variables */
    static real t0, t1, t2, t3, rnorm_buf__;
    static integer jj, iter;
    extern doublereal sdot_(integer *, real *, integer *, real *, integer *);
    static logical orth;
    static integer iseed[4], idist;
    extern /* Subroutine */ int sgemv_(char *, integer *, integer *, real *, 
	    real *, integer *, real *, integer *, real *, real *, integer *, 
	    ftnlen);
    static logical first;
    extern /* Subroutine */ int scopy_(integer *, real *, integer *, real *, 
	    integer *), mpi_allreduce__(real *, real *, integer *, integer *, 
	    integer *, integer *, integer *);
    static real rnorm0;
    static integer msglvl;
    extern /* Subroutine */ int second_(real *), psvout_(integer *, integer *,
	     integer *, real *, integer *, char *, ftnlen);
    extern doublereal psnorm2_(integer *, integer *, real *, integer *);
    extern /* Subroutine */ int pslarnv_(integer *, integer *, integer *, 
	    integer *, real *);



/*     %---------------% */
/*     | MPI Variables | */
/*     %---------------% */

/* /+ */
/* * */
/* *  (C) 1993 by Argonne National Laboratory and Mississipi State University. */
/* *      All rights reserved.  See COPYRIGHT in top-level directory. */
/* +/ */

/* /+ user include file for MPI programs, with no dependencies +/ */

/* /+ return codes +/ */







/*     We handle datatypes by putting the variables that hold them into */
/*     common.  This way, a Fortran program can directly use the various */
/*     datatypes and can even give them to C programs. */

/*     MPI_BOTTOM needs to be a known address; here we put it at the */
/*     beginning of the common block.  The point-to-point and collective */
/*     routines know about MPI_BOTTOM, but MPI_TYPE_STRUCT as yet does not. */

/*     The types MPI_INTEGER1,2,4 and MPI_REAL4,8 are OPTIONAL. */
/*     Their values are zero if they are not available.  Note that */
/*     using these reduces the portability of code (though may enhance */
/*     portability between Crays and other systems) */



/*     All other MPI routines are subroutines */

/*     The attribute copy/delete functions are symbols that can be passed */
/*     to MPI routines */

/*     %----------------------------------------------------% */
/*     | Include files for debugging and timing information | */
/*     %----------------------------------------------------% */


/* \SCCS Information: @(#) */
/* FILE: debug.h   SID: 2.3   DATE OF SID: 11/16/95   RELEASE: 2 */

/*     %---------------------------------% */
/*     | See debug.doc for documentation | */
/*     %---------------------------------% */

/*     %------------------% */
/*     | Scalar Arguments | */
/*     %------------------% */

/*     %--------------------------------% */
/*     | See stat.doc for documentation | */
/*     %--------------------------------% */

/* \SCCS Information: @(#) */
/* FILE: stat.h   SID: 2.2   DATE OF SID: 11/16/95   RELEASE: 2 */



/*     %-----------------% */
/*     | Array Arguments | */
/*     %-----------------% */


/*     %------------% */
/*     | Parameters | */
/*     %------------% */


/*     %------------------------% */
/*     | Local Scalars & Arrays | */
/*     %------------------------% */



/*     %----------------------% */
/*     | External Subroutines | */
/*     %----------------------% */


/*     %--------------------% */
/*     | External Functions | */
/*     %--------------------% */


/*     %---------------------% */
/*     | Intrinsic Functions | */
/*     %---------------------% */


/*     %-----------------% */
/*     | Data Statements | */
/*     %-----------------% */

    /* Parameter adjustments */
    --workd;
    --resid;
    --workl;
    v_dim1 = *ldv;
    v_offset = 1 + v_dim1;
    v -= v_offset;
    --ipntr;

    /* Function Body */

/*     %-----------------------% */
/*     | Executable Statements | */
/*     %-----------------------% */


/*     %-----------------------------------% */
/*     | Initialize the seed of the LAPACK | */
/*     | random number generator           | */
/*     %-----------------------------------% */

    if (inits) {
	iseed[0] = 1;
	iseed[1] = 3;
	iseed[2] = 5;
	iseed[3] = 7;
	inits = FALSE_;
    }

    if (*ido == 0) {

/*        %-------------------------------% */
/*        | Initialize timing statistics  | */
/*        | & message level for debugging | */
/*        %-------------------------------% */

	second_(&t0);
	msglvl = debug_1.mgetv0;

	*ierr = 0;
	iter = 0;
	first = FALSE_;
	orth = FALSE_;

/*        %-----------------------------------------------------% */
/*        | Possibly generate a random starting vector in RESID | */
/*        | Use a LAPACK random number generator used by the    | */
/*        | matrix generation routines.                         | */
/*        |    idist = 1: uniform (0,1)  distribution;          | */
/*        |    idist = 2: uniform (-1,1) distribution;          | */
/*        |    idist = 3: normal  (0,1)  distribution;          | */
/*        %-----------------------------------------------------% */

	if (! (*initv)) {
	    idist = 2;
	    pslarnv_(comm, &idist, iseed, n, &resid[1]);
	}

/*        %----------------------------------------------------------% */
/*        | Force the starting vector into the range of OP to handle | */
/*        | the generalized problem when B is possibly (singular).   | */
/*        %----------------------------------------------------------% */

	second_(&t2);
	if (*(unsigned char *)bmat == 'G') {
	    ++timing_1.nopx;
	    ipntr[1] = 1;
	    ipntr[2] = *n + 1;
	    scopy_(n, &resid[1], &c__1, &workd[1], &c__1);
	    *ido = -1;
	    goto L9000;
	}
    }

/*     %-----------------------------------------% */
/*     | Back from computing OP*(initial-vector) | */
/*     %-----------------------------------------% */

    if (first) {
	goto L20;
    }

/*     %-----------------------------------------------% */
/*     | Back from computing B*(orthogonalized-vector) | */
/*     %-----------------------------------------------% */

    if (orth) {
	goto L40;
    }

    second_(&t3);
    timing_1.tmvopx += t3 - t2;

/*     %------------------------------------------------------% */
/*     | Starting vector is now in the range of OP; r = OP*r; | */
/*     | Compute B-norm of starting vector.                   | */
/*     %------------------------------------------------------% */

    second_(&t2);
    first = TRUE_;
    if (*(unsigned char *)bmat == 'G') {
	++timing_1.nbx;
	scopy_(n, &workd[*n + 1], &c__1, &resid[1], &c__1);
	ipntr[1] = *n + 1;
	ipntr[2] = 1;
	*ido = 2;
	goto L9000;
    } else if (*(unsigned char *)bmat == 'I') {
	scopy_(n, &resid[1], &c__1, &workd[1], &c__1);
    }

L20:

    if (*(unsigned char *)bmat == 'G') {
	second_(&t3);
	timing_1.tmvbx += t3 - t2;
    }

    first = FALSE_;
    if (*(unsigned char *)bmat == 'G') {
	rnorm_buf__ = sdot_(n, &resid[1], &c__1, &workd[1], &c__1);
	mpi_allreduce__(&rnorm_buf__, &rnorm0, &c__1, &mpipriv_1.mpi_real__, &
		mpipriv_1.mpi_sum__, comm, ierr);
	rnorm0 = sqrt((dabs(rnorm0)));
    } else if (*(unsigned char *)bmat == 'I') {
	rnorm0 = psnorm2_(comm, n, &resid[1], &c__1);
    }
    *rnorm = rnorm0;

/*     %---------------------------------------------% */
/*     | Exit if this is the very first Arnoldi step | */
/*     %---------------------------------------------% */

    if (*j == 1) {
	goto L50;
    }

/*     %---------------------------------------------------------------- */
/*     | Otherwise need to B-orthogonalize the starting vector against | */
/*     | the current Arnoldi basis using Gram-Schmidt with iter. ref.  | */
/*     | This is the case where an invariant subspace is encountered   | */
/*     | in the middle of the Arnoldi factorization.                   | */
/*     |                                                               | */
/*     |       s = V^{T}*B*r;   r = r - V*s;                           | */
/*     |                                                               | */
/*     | Stopping criteria used for iter. ref. is discussed in         | */
/*     | Parlett's book, page 107 and in Gragg & Reichel TOMS paper.   | */
/*     %---------------------------------------------------------------% */

    orth = TRUE_;
L30:

    i__1 = *j - 1;
    sgemv_("T", n, &i__1, &c_b24, &v[v_offset], ldv, &workd[1], &c__1, &c_b26,
	     &workl[*j + 1], &c__1, (ftnlen)1);
    i__1 = *j - 1;
    mpi_allreduce__(&workl[*j + 1], &workl[1], &i__1, &mpipriv_1.mpi_real__, &
	    mpipriv_1.mpi_sum__, comm, ierr);
    i__1 = *j - 1;
    sgemv_("N", n, &i__1, &c_b29, &v[v_offset], ldv, &workl[1], &c__1, &c_b24,
	     &resid[1], &c__1, (ftnlen)1);

/*     %----------------------------------------------------------% */
/*     | Compute the B-norm of the orthogonalized starting vector | */
/*     %----------------------------------------------------------% */

    second_(&t2);
    if (*(unsigned char *)bmat == 'G') {
	++timing_1.nbx;
	scopy_(n, &resid[1], &c__1, &workd[*n + 1], &c__1);
	ipntr[1] = *n + 1;
	ipntr[2] = 1;
	*ido = 2;
	goto L9000;
    } else if (*(unsigned char *)bmat == 'I') {
	scopy_(n, &resid[1], &c__1, &workd[1], &c__1);
    }

L40:

    if (*(unsigned char *)bmat == 'G') {
	second_(&t3);
	timing_1.tmvbx += t3 - t2;
    }

    if (*(unsigned char *)bmat == 'G') {
	rnorm_buf__ = sdot_(n, &resid[1], &c__1, &workd[1], &c__1);
	mpi_allreduce__(&rnorm_buf__, rnorm, &c__1, &mpipriv_1.mpi_real__, &
		mpipriv_1.mpi_sum__, comm, ierr);
	*rnorm = sqrt((dabs(*rnorm)));
    } else if (*(unsigned char *)bmat == 'I') {
	*rnorm = psnorm2_(comm, n, &resid[1], &c__1);
    }

/*     %--------------------------------------% */
/*     | Check for further orthogonalization. | */
/*     %--------------------------------------% */

    if (msglvl > 2) {
	psvout_(comm, &debug_1.logfil, &c__1, &rnorm0, &debug_1.ndigit, "_ge"
		"tv0: re-orthonalization ; rnorm0 is", (ftnlen)38);
	psvout_(comm, &debug_1.logfil, &c__1, rnorm, &debug_1.ndigit, "_getv"
		"0: re-orthonalization ; rnorm is", (ftnlen)37);
    }

    if (*rnorm > rnorm0 * .717f) {
	goto L50;
    }

    ++iter;
    if (iter <= 1) {

/*        %-----------------------------------% */
/*        | Perform iterative refinement step | */
/*        %-----------------------------------% */

	rnorm0 = *rnorm;
	goto L30;
    } else {

/*        %------------------------------------% */
/*        | Iterative refinement step "failed" | */
/*        %------------------------------------% */

	i__1 = *n;
	for (jj = 1; jj <= i__1; ++jj) {
	    resid[jj] = 0.f;
/* L45: */
	}
	*rnorm = 0.f;
	*ierr = -1;
    }

L50:

    if (msglvl > 0) {
	psvout_(comm, &debug_1.logfil, &c__1, rnorm, &debug_1.ndigit, "_getv"
		"0: B-norm of initial / restarted starting vector", (ftnlen)53)
		;
    }
    if (msglvl > 2) {
	psvout_(comm, &debug_1.logfil, n, &resid[1], &debug_1.ndigit, "_getv"
		"0: initial / restarted starting vector", (ftnlen)43);
    }
    *ido = 99;

    second_(&t1);
    timing_1.tgetv0 += t1 - t0;

L9000:
    return 0;

/*     %----------------% */
/*     | End of psgetv0 | */
/*     %----------------% */

} /* psgetv0_ */

