/* pzgetv0.f -- translated by f2c (version 20050501).
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

static integer c__9 = 9;
static integer c__1 = 1;
static doublecomplex c_b29 = {1.,0.};
static doublecomplex c_b31 = {0.,0.};
static doublecomplex c_b34 = {-1.,-0.};

/* \BeginDoc */

/* \Name: pzgetv0 */

/* Message Passing Layer: MPI */

/* \Description: */
/*  Generate a random initial residual vector for the Arnoldi process. */
/*  Force the residual vector to be in the range of the operator OP. */

/* \Usage: */
/*  call pzgetv0 */
/*     ( COMM, IDO, BMAT, ITRY, INITV, N, J, V, LDV, RESID, RNORM, */
/*       IPNTR, WORKD, WORKL, IERR ) */

/* \Arguments */
/*  COMM    MPI  Communicator for the processor grid.  (INPUT) */

/*  IDO     Integer.  (INPUT/OUTPUT) */
/*          Reverse communication flag.  IDO must be zero on the first */
/*          call to pzgetv0 . */
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
/*          ITRY counts the number of times that pzgetv0  is called. */
/*          It should be set to 1 on the initial call to pzgetv0 . */

/*  INITV   Logical variable.  (INPUT) */
/*          .TRUE.  => the initial residual vector is given in RESID. */
/*          .FALSE. => generate a random initial residual vector. */

/*  N       Integer.  (INPUT) */
/*          Dimension of the problem. */

/*  J       Integer.  (INPUT) */
/*          Index of the residual vector to be generated, with respect to */
/*          the Arnoldi process.  J > 1 in case of a "restart". */

/*  V       Complex*16  N by J array.  (INPUT) */
/*          The first J-1 columns of V contain the current Arnoldi basis */
/*          if this is a "restart". */

/*  LDV     Integer.  (INPUT) */
/*          Leading dimension of V exactly as declared in the calling */
/*          program. */

/*  RESID   Complex*16  array of length N.  (INPUT/OUTPUT) */
/*          Initial residual vector to be generated.  If RESID is */
/*          provided, force RESID into the range of the operator OP. */

/*  RNORM   Double precision  scalar.  (OUTPUT) */
/*          B-norm of the generated residual. */

/*  IPNTR   Integer array of length 3.  (OUTPUT) */

/*  WORKD   Complex*16  work array of length 2*N.  (REVERSE COMMUNICATION). */
/*          On exit, WORK(1:N) = B*RESID to be used in SSAITR. */

/*  WORKL   Complex*16  work space used for Gram Schmidt orthogonalization */

/*  IERR    Integer.  (OUTPUT) */
/*          =  0: Normal exit. */
/*          = -1: Cannot generate a nontrivial restarted residual vector */
/*                in the range of the operator OP. */

/* \EndDoc */

/* ----------------------------------------------------------------------- */

/* \BeginLib */

/* \Local variables: */
/*     xxxxxx  Complex*16 */

/* \References: */
/*  1. D.C. Sorensen, "Implicit Application of Polynomial Filters in */
/*     a k-Step Arnoldi Method", SIAM J. Matr. Anal. Apps., 13 (1992), */
/*     pp 357-385. */

/* \Routines called: */
/*     second   ARPACK utility routine for timing. */
/*     pzvout    Parallel ARPACK utility routine that prints vectors. */
/*     pzlarnv   Parallel wrapper for LAPACK routine zlarnv  (generates a random vector). */
/*     zgemv     Level 2 BLAS routine for matrix vector multiplication. */
/*     zcopy     Level 1 BLAS that copies one vector to another. */
/*     zdotc     Level 1 BLAS that computes the scalar product of two vectors. */
/*     pdznorm2  Parallel version of Level 1 BLAS that computes the norm of a vector. */

/* \Author */
/*     Danny Sorensen               Phuong Vu */
/*     Richard Lehoucq              CRPC / Rice University */
/*     Dept. of Computational &     Houston, Texas */
/*     Applied Mathematics */
/*     Rice University */
/*     Houston, Texas */

/* \Parallel Modifications */
/*     Kristi Maschhoff */

/* \Revision history: */
/*     Starting Point: Complex Code FILE: getv0.F   SID: 2.1 */

/* \SCCS Information: */
/* FILE: getv0.F   SID: 1.7   DATE OF SID: 04/12/01 */

/* \EndLib */

/* ----------------------------------------------------------------------- */

/* Subroutine */ int pzgetv0_(integer *comm, integer *ido, char *bmat, 
	integer *itry, logical *initv, integer *n, integer *j, doublecomplex *
	v, integer *ldv, doublecomplex *resid, doublereal *rnorm, integer *
	ipntr, doublecomplex *workd, doublecomplex *workl, integer *ierr, 
	ftnlen bmat_len)
{
    /* Initialized data */

    static logical inits = TRUE_;

    /* System generated locals */
    integer v_dim1, v_offset, i__1, i__2;
    doublereal d__1, d__2;
    doublecomplex z__1;

    /* Builtin functions */
    integer s_wsle(cilist *), do_lio(integer *, integer *, char *, ftnlen), 
	    e_wsle(void);
    double d_imag(doublecomplex *), sqrt(doublereal);

    /* Local variables */
    extern doublereal pdznorm2_(integer *, integer *, doublecomplex *, 
	    integer *);
    static doublecomplex cnorm_buf__;
    static real t0, t1, t2, t3;
    static integer jj, igen, myid, iter;
    static logical orth;
    static integer iseed[4], idist;
    static doublecomplex cnorm;
    extern /* Double Complex */ VOID zdotc_(doublecomplex *, integer *, 
	    doublecomplex *, integer *, doublecomplex *, integer *);
    static logical first;
    extern /* Subroutine */ int zgemv_(char *, integer *, integer *, 
	    doublecomplex *, doublecomplex *, integer *, doublecomplex *, 
	    integer *, doublecomplex *, doublecomplex *, integer *, ftnlen), 
	    mpi_allreduce__(doublecomplex *, doublecomplex *, integer *, 
	    integer *, integer *, integer *, integer *), zcopy_(integer *, 
	    doublecomplex *, integer *, doublecomplex *, integer *), 
	    mpi_comm_rank__(integer *, integer *, integer *);
    extern doublereal dlapy2_(doublereal *, doublereal *);
    static doublecomplex cnorm2;
    static doublereal rnorm0;
    static integer msglvl;
    extern /* Subroutine */ int second_(real *), pdvout_(integer *, integer *,
	     integer *, doublereal *, integer *, char *, ftnlen), pzvout_(
	    integer *, integer *, integer *, doublecomplex *, integer *, char 
	    *, ftnlen), pzlarnv_(integer *, integer *, integer *, integer *, 
	    doublecomplex *);

    /* Fortran I/O blocks */
    static cilist io___4 = { 0, 0, 0, 0, 0 };




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

/*        %-----------------------------------% */
/*        | Generate a seed on each processor | */
/*        | using process id (myid).          | */
/*        | Note: the seed must be between 1  | */
/*        | and 4095.  iseed(4) must be odd.  | */
/*        %-----------------------------------% */

	mpi_comm_rank__(comm, &myid, ierr);
	igen = (myid << 1) + 1001;
	if (igen > 4095) {
	    s_wsle(&io___4);
	    do_lio(&c__9, &c__1, "Error in p_getv0: seed exceeds 4095!", (
		    ftnlen)36);
	    e_wsle();
	}

	iseed[0] = igen / 1000;
	igen %= 1000;
	iseed[1] = igen / 100;
	igen %= 100;
	iseed[2] = igen / 10;
	iseed[3] = igen % 10;

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
	    pzlarnv_(comm, &idist, iseed, n, &resid[1]);
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
	    zcopy_(n, &resid[1], &c__1, &workd[1], &c__1);
	    *ido = -1;
	    goto L9000;
	}
    }

/*     %----------------------------------------% */
/*     | Back from computing B*(initial-vector) | */
/*     %----------------------------------------% */

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
	zcopy_(n, &workd[*n + 1], &c__1, &resid[1], &c__1);
	ipntr[1] = *n + 1;
	ipntr[2] = 1;
	*ido = 2;
	goto L9000;
    } else if (*(unsigned char *)bmat == 'I') {
	zcopy_(n, &resid[1], &c__1, &workd[1], &c__1);
    }

L20:

    if (*(unsigned char *)bmat == 'G') {
	second_(&t3);
	timing_1.tmvbx += t3 - t2;
    }

    first = FALSE_;
    if (*(unsigned char *)bmat == 'G') {
	zdotc_(&z__1, n, &resid[1], &c__1, &workd[1], &c__1);
	cnorm_buf__.r = z__1.r, cnorm_buf__.i = z__1.i;
	mpi_allreduce__(&cnorm_buf__, &cnorm, &c__1, &
		mpipriv_1.mpi_double_complex__, &mpipriv_1.mpi_sum__, comm, 
		ierr);
	d__1 = cnorm.r;
	d__2 = d_imag(&cnorm);
	rnorm0 = sqrt(dlapy2_(&d__1, &d__2));
    } else if (*(unsigned char *)bmat == 'I') {
	rnorm0 = pdznorm2_(comm, n, &resid[1], &c__1);
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
/*     | Parlett`s book, page 107 and in Gragg and Reichel TOMS paper. | */
/*     %---------------------------------------------------------------% */

    orth = TRUE_;
L30:

    i__1 = *j - 1;
    zgemv_("C", n, &i__1, &c_b29, &v[v_offset], ldv, &workd[1], &c__1, &c_b31,
	     &workl[*j + 1], &c__1, (ftnlen)1);
    i__1 = *j - 1;
    mpi_allreduce__(&workl[*j + 1], &workl[1], &i__1, &
	    mpipriv_1.mpi_double_complex__, &mpipriv_1.mpi_sum__, comm, ierr);
    i__1 = *j - 1;
    zgemv_("N", n, &i__1, &c_b34, &v[v_offset], ldv, &workl[1], &c__1, &c_b29,
	     &resid[1], &c__1, (ftnlen)1);

/*     %----------------------------------------------------------% */
/*     | Compute the B-norm of the orthogonalized starting vector | */
/*     %----------------------------------------------------------% */

    second_(&t2);
    if (*(unsigned char *)bmat == 'G') {
	++timing_1.nbx;
	zcopy_(n, &resid[1], &c__1, &workd[*n + 1], &c__1);
	ipntr[1] = *n + 1;
	ipntr[2] = 1;
	*ido = 2;
	goto L9000;
    } else if (*(unsigned char *)bmat == 'I') {
	zcopy_(n, &resid[1], &c__1, &workd[1], &c__1);
    }

L40:

    if (*(unsigned char *)bmat == 'G') {
	second_(&t3);
	timing_1.tmvbx += t3 - t2;
    }

    if (*(unsigned char *)bmat == 'G') {
	zdotc_(&z__1, n, &resid[1], &c__1, &workd[1], &c__1);
	cnorm_buf__.r = z__1.r, cnorm_buf__.i = z__1.i;
	mpi_allreduce__(&cnorm_buf__, &cnorm, &c__1, &
		mpipriv_1.mpi_double_complex__, &mpipriv_1.mpi_sum__, comm, 
		ierr);
	d__1 = cnorm.r;
	d__2 = d_imag(&cnorm);
	*rnorm = sqrt(dlapy2_(&d__1, &d__2));
    } else if (*(unsigned char *)bmat == 'I') {
	*rnorm = pdznorm2_(comm, n, &resid[1], &c__1);
    }

/*     %--------------------------------------% */
/*     | Check for further orthogonalization. | */
/*     %--------------------------------------% */

    if (msglvl > 2) {
	pdvout_(comm, &debug_1.logfil, &c__1, &rnorm0, &debug_1.ndigit, "_ge"
		"tv0: re-orthonalization ; rnorm0 is", (ftnlen)38);
	pdvout_(comm, &debug_1.logfil, &c__1, rnorm, &debug_1.ndigit, "_getv"
		"0: re-orthonalization ; rnorm is", (ftnlen)37);
    }

    if (*rnorm > rnorm0 * .717f) {
	goto L50;
    }

    ++iter;
    if (iter <= 5) {

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
	    i__2 = jj;
	    resid[i__2].r = 0., resid[i__2].i = 0.;
/* L45: */
	}
	*rnorm = 0.;
	*ierr = -1;
    }

L50:

    if (msglvl > 0) {
	z__1.r = *rnorm, z__1.i = 0.;
	cnorm2.r = z__1.r, cnorm2.i = z__1.i;
	pzvout_(comm, &debug_1.logfil, &c__1, &cnorm2, &debug_1.ndigit, "_ge"
		"tv0: B-norm of initial / restarted starting vector", (ftnlen)
		53);
    }
    if (msglvl > 2) {
	pzvout_(comm, &debug_1.logfil, n, &resid[1], &debug_1.ndigit, "_getv"
		"0: initial / restarted starting vector", (ftnlen)43);
    }
    *ido = 99;

    second_(&t1);
    timing_1.tgetv0 += t1 - t0;

L9000:
    return 0;

/*     %----------------% */
/*     | End of pzgetv0  | */
/*     %----------------% */

} /* pzgetv0_ */

