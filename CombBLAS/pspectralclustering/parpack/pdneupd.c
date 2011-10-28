/* pdneupd.f -- translated by f2c (version 20050501).
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

static doublereal c_b3 = .66666666666666663;
static integer c__1 = 1;
static doublereal c_b37 = 0.;
static doublereal c_b38 = 1.;
static logical c_true = TRUE_;
static doublereal c_b64 = -1.;

/* \BeginDoc */

/* \Name: pdneupd */

/* Message Passing Layer: MPI */

/* \Description: */

/*  This subroutine returns the converged approximations to eigenvalues */
/*  of A*z = lambda*B*z and (optionally): */

/*      (1) The corresponding approximate eigenvectors; */

/*      (2) An orthonormal basis for the associated approximate */
/*          invariant subspace; */

/*      (3) Both. */

/*  There is negligible additional cost to obtain eigenvectors.  An orthonormal */
/*  basis is always computed.  There is an additional storage cost of n*nev */
/*  if both are requested (in this case a separate array Z must be supplied). */

/*  The approximate eigenvalues and eigenvectors of  A*z = lambda*B*z */
/*  are derived from approximate eigenvalues and eigenvectors of */
/*  of the linear operator OP prescribed by the MODE selection in the */
/*  call to PDNAUPD .  PDNAUPD  must be called before this routine is called. */
/*  These approximate eigenvalues and vectors are commonly called Ritz */
/*  values and Ritz vectors respectively.  They are referred to as such */
/*  in the comments that follow.  The computed orthonormal basis for the */
/*  invariant subspace corresponding to these Ritz values is referred to as a */
/*  Schur basis. */

/*  See documentation in the header of the subroutine PDNAUPD  for */
/*  definition of OP as well as other terms and the relation of computed */
/*  Ritz values and Ritz vectors of OP with respect to the given problem */
/*  A*z = lambda*B*z.  For a brief description, see definitions of */
/*  IPARAM(7), MODE and WHICH in the documentation of PDNAUPD . */

/* \Usage: */
/*  call pdneupd */
/*     ( COMM, RVEC, HOWMNY, SELECT, DR, DI, Z, LDZ, SIGMAR, SIGMAI, */
/*       WORKEV, BMAT, N, WHICH, NEV, TOL, RESID, NCV, V, LDV, IPARAM, */
/*       IPNTR, WORKD, WORKL, LWORKL, INFO ) */

/* \Arguments */
/*  COMM    MPI  Communicator for the processor grid.  (INPUT) */

/*  RVEC    LOGICAL  (INPUT) */
/*          Specifies whether a basis for the invariant subspace corresponding */
/*          to the converged Ritz value approximations for the eigenproblem */
/*          A*z = lambda*B*z is computed. */

/*             RVEC = .FALSE.     Compute Ritz values only. */

/*             RVEC = .TRUE.      Compute the Ritz vectors or Schur vectors. */
/*                                See Remarks below. */

/*  HOWMNY  Character*1  (INPUT) */
/*          Specifies the form of the basis for the invariant subspace */
/*          corresponding to the converged Ritz values that is to be computed. */

/*          = 'A': Compute NEV Ritz vectors; */
/*          = 'P': Compute NEV Schur vectors; */
/*          = 'S': compute some of the Ritz vectors, specified */
/*                 by the logical array SELECT. */

/*  SELECT  Logical array of dimension NCV.  (INPUT) */
/*          If HOWMNY = 'S', SELECT specifies the Ritz vectors to be */
/*          computed. To select the Ritz vector corresponding to a */
/*          Ritz value (DR(j), DI(j)), SELECT(j) must be set to .TRUE.. */
/*          If HOWMNY = 'A' or 'P', SELECT is used as internal workspace. */

/*  DR      Double precision  array of dimension NEV+1.  (OUTPUT) */
/*          If IPARAM(7) = 1,2 or 3 and SIGMAI=0.0  then on exit: DR contains */
/*          the real part of the Ritz  approximations to the eigenvalues of */
/*          A*z = lambda*B*z. */
/*          If IPARAM(7) = 3, 4 and SIGMAI is not equal to zero, then on exit: */
/*          DR contains the real part of the Ritz values of OP computed by */
/*          PDNAUPD . A further computation must be performed by the user */
/*          to transform the Ritz values computed for OP by PDNAUPD  to those */
/*          of the original system A*z = lambda*B*z. See remark 3 below. */

/*  DI      Double precision  array of dimension NEV+1.  (OUTPUT) */
/*          On exit, DI contains the imaginary part of the Ritz value */
/*          approximations to the eigenvalues of A*z = lambda*B*z associated */
/*          with DR. */

/*          NOTE: When Ritz values are complex, they will come in complex */
/*                conjugate pairs.  If eigenvectors are requested, the */
/*                corresponding Ritz vectors will also come in conjugate */
/*                pairs and the real and imaginary parts of these are */
/*                represented in two consecutive columns of the array Z */
/*                (see below). */

/*  Z       Double precision  N by NEV+1 array if RVEC = .TRUE. and HOWMNY = 'A'. (OUTPUT) */
/*          On exit, if RVEC = .TRUE. and HOWMNY = 'A', then the columns of */
/*          Z represent approximate eigenvectors (Ritz vectors) corresponding */
/*          to the NCONV=IPARAM(5) Ritz values for eigensystem */
/*          A*z = lambda*B*z. */

/*          The complex Ritz vector associated with the Ritz value */
/*          with positive imaginary part is stored in two consecutive */
/*          columns.  The first column holds the real part of the Ritz */
/*          vector and the second column holds the imaginary part.  The */
/*          Ritz vector associated with the Ritz value with negative */
/*          imaginary part is simply the complex conjugate of the Ritz vector */
/*          associated with the positive imaginary part. */

/*          If  RVEC = .FALSE. or HOWMNY = 'P', then Z is not referenced. */

/*          NOTE: If if RVEC = .TRUE. and a Schur basis is not required, */
/*          the array Z may be set equal to first NEV+1 columns of the Arnoldi */
/*          basis array V computed by PDNAUPD .  In this case the Arnoldi basis */
/*          will be destroyed and overwritten with the eigenvector basis. */

/*  LDZ     Integer.  (INPUT) */
/*          The leading dimension of the array Z.  If Ritz vectors are */
/*          desired, then  LDZ >= max( 1, N ).  In any case,  LDZ >= 1. */

/*  SIGMAR  Double precision   (INPUT) */
/*          If IPARAM(7) = 3 or 4, represents the real part of the shift. */
/*          Not referenced if IPARAM(7) = 1 or 2. */

/*  SIGMAI  Double precision   (INPUT) */
/*          If IPARAM(7) = 3 or 4, represents the imaginary part of the shift. */
/*          Not referenced if IPARAM(7) = 1 or 2. See remark 3 below. */

/*  WORKEV  Double precision  work array of dimension 3*NCV.  (WORKSPACE) */

/*  **** The remaining arguments MUST be the same as for the   **** */
/*  **** call to PDNAUPD  that was just completed.               **** */

/*  NOTE: The remaining arguments */

/*           BMAT, N, WHICH, NEV, TOL, RESID, NCV, V, LDV, IPARAM, IPNTR, */
/*           WORKD, WORKL, LWORKL, INFO */

/*         must be passed directly to PDNEUPD  following the last call */
/*         to PDNAUPD .  These arguments MUST NOT BE MODIFIED between */
/*         the the last call to PDNAUPD  and the call to PDNEUPD . */

/*  Three of these parameters (V, WORKL, INFO) are also output parameters: */

/*  V       Double precision  N by NCV array.  (INPUT/OUTPUT) */

/*          Upon INPUT: the NCV columns of V contain the Arnoldi basis */
/*                      vectors for OP as constructed by PDNAUPD  . */

/*          Upon OUTPUT: If RVEC = .TRUE. the first NCONV=IPARAM(5) columns */
/*                       contain approximate Schur vectors that span the */
/*                       desired invariant subspace.  See Remark 2 below. */

/*          NOTE: If the array Z has been set equal to first NEV+1 columns */
/*          of the array V and RVEC=.TRUE. and HOWMNY= 'A', then the */
/*          Arnoldi basis held by V has been overwritten by the desired */
/*          Ritz vectors.  If a separate array Z has been passed then */
/*          the first NCONV=IPARAM(5) columns of V will contain approximate */
/*          Schur vectors that span the desired invariant subspace. */

/*  WORKL   Double precision  work array of length LWORKL.  (OUTPUT/WORKSPACE) */
/*          WORKL(1:ncv*ncv+3*ncv) contains information obtained in */
/*          PDNAUPD .  They are not changed by PDNEUPD . */
/*          WORKL(ncv*ncv+3*ncv+1:3*ncv*ncv+6*ncv) holds the */
/*          real and imaginary part of the untransformed Ritz values, */
/*          the upper quasi-triangular matrix for H, and the */
/*          associated matrix representation of the invariant subspace for H. */

/*          Note: IPNTR(9:13) contains the pointers into WORKL for addresses */
/*          of the above information computed by PDNEUPD . */
/*          ------------------------------------------------------------- */
/*          IPNTR(9):  pointer to the real part of the NCV RITZ values of the */
/*                     original system. */
/*          IPNTR(10): pointer to the imaginary part of the NCV RITZ values of */
/*                     the original system. */
/*          IPNTR(11): pointer to the NCV corresponding error bounds. */
/*          IPNTR(12): pointer to the NCV by NCV upper quasi-triangular */
/*                     Schur matrix for H. */
/*          IPNTR(13): pointer to the NCV by NCV matrix of eigenvectors */
/*                     of the upper Hessenberg matrix H. Only referenced by */
/*                     PDNEUPD  if RVEC = .TRUE. See Remark 2 below. */
/*          ------------------------------------------------------------- */

/*  INFO    Integer.  (OUTPUT) */
/*          Error flag on output. */

/*          =  0: Normal exit. */

/*          =  1: The Schur form computed by LAPACK routine dlahqr */
/*                could not be reordered by LAPACK routine dtrsen . */
/*                Re-enter subroutine pdneupd  with IPARAM(5)=NCV and */
/*                increase the size of the arrays DR and DI to have */
/*                dimension at least dimension NCV and allocate at least NCV */
/*                columns for Z. NOTE: Not necessary if Z and V share */
/*                the same space. Please notify the authors if this error */
/*                occurs. */

/*          = -1: N must be positive. */
/*          = -2: NEV must be positive. */
/*          = -3: NCV-NEV >= 2 and less than or equal to N. */
/*          = -5: WHICH must be one of 'LM', 'SM', 'LR', 'SR', 'LI', 'SI' */
/*          = -6: BMAT must be one of 'I' or 'G'. */
/*          = -7: Length of private work WORKL array is not sufficient. */
/*          = -8: Error return from calculation of a real Schur form. */
/*                Informational error from LAPACK routine dlahqr . */
/*          = -9: Error return from calculation of eigenvectors. */
/*                Informational error from LAPACK routine dtrevc . */
/*          = -10: IPARAM(7) must be 1,2,3,4. */
/*          = -11: IPARAM(7) = 1 and BMAT = 'G' are incompatible. */
/*          = -12: HOWMNY = 'S' not yet implemented */
/*          = -13: HOWMNY must be one of 'A' or 'P' if RVEC = .true. */
/*          = -14: PDNAUPD  did not find any eigenvalues to sufficient */
/*                 accuracy. */
/*          = -15: PDNEUPD  got a different count of the number of converged */
/*                 Ritz values than PDNAUPD  got.  This indicates the user */
/*                 probably made an error in passing data from PDNAUPD  to */
/*                 PDNEUPD  or that the data was modified before entering */
/*                 PDNEUPD . */

/* \BeginLib */

/* \References: */
/*  1. D.C. Sorensen, "Implicit Application of Polynomial Filters in */
/*     a k-Step Arnoldi Method", SIAM J. Matr. Anal. Apps., 13 (1992), */
/*     pp 357-385. */
/*  2. R.B. Lehoucq, "Analysis and Implementation of an Implicitly */
/*     Restarted Arnoldi Iteration", Rice University Technical Report */
/*     TR95-13, Department of Computational and Applied Mathematics. */
/*  3. B.N. Parlett & Y. Saad, "Complex Shift and Invert Strategies for */
/*     Real Matrices", Linear Algebra and its Applications, vol 88/89, */
/*     pp 575-595, (1987). */

/* \Routines called: */
/*     pivout  Parallel ARPACK utility routine that prints integers. */
/*     pdmout   Parallel ARPACK utility routine that prints matrices */
/*     pdvout   Parallel ARPACK utility routine that prints vectors. */
/*     dgeqr2   LAPACK routine that computes the QR factorization of */
/*             a matrix. */
/*     dlacpy   LAPACK matrix copy routine. */
/*     dlahqr   LAPACK routine to compute the real Schur form of an */
/*             upper Hessenberg matrix. */
/*     pdlamch  ScaLAPACK routine that determines machine constants. */
/*     dlapy2   LAPACK routine to compute sqrt(x**2+y**2) carefully. */
/*     dlaset   LAPACK matrix initialization routine. */
/*     dorm2r   LAPACK routine that applies an orthogonal matrix in */
/*             factored form. */
/*     dtrevc   LAPACK routine to compute the eigenvectors of a matrix */
/*             in upper quasi-triangular form. */
/*     dtrsen   LAPACK routine that re-orders the Schur form. */
/*     dtrmm    Level 3 BLAS matrix times an upper triangular matrix. */
/*     dger     Level 2 BLAS rank one update to a matrix. */
/*     dnrm2    Level 1 BLAS that computes the norm of a vector. */
/*     dscal    Level 1 BLAS that scales a vector. */
/*     dcopy    Level 1 BLAS that copies one vector to another . */

/* \Remarks */

/*  1. Currently only HOWMNY = 'A' and 'P' are implemented. */

/*     Let X` denote the transpose of X. */

/*  2. Schur vectors are an orthogonal representation for the basis of */
/*     Ritz vectors. Thus, their numerical properties are often superior. */
/*     If RVEC = .TRUE. then the relationship */
/*             A * V(:,1:IPARAM(5)) = V(:,1:IPARAM(5)) * T, and */
/*     V(:,1:IPARAM(5))` * V(:,1:IPARAM(5)) = I are approximately satisfied. */
/*     Here T is the leading submatrix of order IPARAM(5) of the real */
/*     upper quasi-triangular matrix stored workl(ipntr(12)). That is, */
/*     T is block upper triangular with 1-by-1 and 2-by-2 diagonal blocks; */
/*     each 2-by-2 diagonal block has its diagonal elements equal and its */
/*     off-diagonal elements of opposite sign.  Corresponding to each 2-by-2 */
/*     diagonal block is a complex conjugate pair of Ritz values. The real */
/*     Ritz values are stored on the diagonal of T. */

/*  3. If IPARAM(7) = 3 or 4 and SIGMAI is not equal zero, then the user must */
/*     form the IPARAM(5) Rayleigh quotients in order to transform the Ritz */
/*     values computed by PDNAUPD  for OP to those of A*z = lambda*B*z. */
/*     Set RVEC = .true. and HOWMNY = 'A', and */
/*     compute */
/*           Z(:,I)` * A * Z(:,I) if DI(I) = 0. */
/*     If DI(I) is not equal to zero and DI(I+1) = - D(I), */
/*     then the desired real and imaginary parts of the Ritz value are */
/*           Z(:,I)` * A * Z(:,I) +  Z(:,I+1)` * A * Z(:,I+1), */
/*           Z(:,I)` * A * Z(:,I+1) -  Z(:,I+1)` * A * Z(:,I), respectively. */
/*     Another possibility is to set RVEC = .true. and HOWMNY = 'P' and */
/*     compute V(:,1:IPARAM(5))` * A * V(:,1:IPARAM(5)) and then an upper */
/*     quasi-triangular matrix of order IPARAM(5) is computed. See remark */
/*     2 above. */

/* \Authors */
/*     Danny Sorensen               Phuong Vu */
/*     Richard Lehoucq              CRPC / Rice University */
/*     Chao Yang                    Houston, Texas */
/*     Dept. of Computational & */
/*     Applied Mathematics */
/*     Rice University */
/*     Houston, Texas */

/* \Parallel Modifications */
/*     Kristi Maschhoff */

/* \Revision history: */
/*     Starting Point: Serial Code FILE: neupd.F   SID: 2.3 */

/* \SCCS Information: */
/* FILE: neupd.F   SID: 1.8   DATE OF SID: 04/10/01 */

/* \EndLib */

/* ----------------------------------------------------------------------- */
/* Subroutine */ int pdneupd_(integer *comm, logical *rvec, char *howmny, 
	logical *select, doublereal *dr, doublereal *di, doublereal *z__, 
	integer *ldz, doublereal *sigmar, doublereal *sigmai, doublereal *
	workev, char *bmat, integer *n, char *which, integer *nev, doublereal 
	*tol, doublereal *resid, integer *ncv, doublereal *v, integer *ldv, 
	integer *iparam, integer *ipntr, doublereal *workd, doublereal *workl,
	 integer *lworkl, integer *info, ftnlen howmny_len, ftnlen bmat_len, 
	ftnlen which_len)
{
    /* System generated locals */
    integer v_dim1, v_offset, z_dim1, z_offset, i__1;
    doublereal d__1, d__2;

    /* Builtin functions */
    double pow_dd(doublereal *, doublereal *);
    integer s_cmp(char *, char *, ftnlen, ftnlen);
    /* Subroutine */ int s_copy(char *, char *, ftnlen, ftnlen);

    /* Local variables */
    static integer j, k, ih, jj, np;
    static doublereal vl[1]	/* was [1][1] */;
    static integer ibd, ldh, ldq, iri;
    static doublereal sep;
    static integer irr, wri, wrr;
    extern /* Subroutine */ int dger_(integer *, integer *, doublereal *, 
	    doublereal *, integer *, doublereal *, integer *, doublereal *, 
	    integer *);
    static integer mode;
    static doublereal eps23;
    static integer ierr;
    static doublereal temp;
    static integer iwev;
    static char type__[6];
    extern doublereal dnrm2_(integer *, doublereal *, integer *);
    static doublereal temp1;
    extern /* Subroutine */ int dscal_(integer *, doublereal *, doublereal *, 
	    integer *);
    static integer ihbds, iconj;
    extern /* Subroutine */ int dgemv_(char *, integer *, integer *, 
	    doublereal *, doublereal *, integer *, doublereal *, integer *, 
	    doublereal *, doublereal *, integer *, ftnlen);
    static doublereal conds;
    static logical reord;
    extern /* Subroutine */ int dcopy_(integer *, doublereal *, integer *, 
	    doublereal *, integer *);
    static integer nconv;
    extern /* Subroutine */ int dtrmm_(char *, char *, char *, char *, 
	    integer *, integer *, doublereal *, doublereal *, integer *, 
	    doublereal *, integer *, ftnlen, ftnlen, ftnlen, ftnlen);
    static integer iwork[1];
    static doublereal rnorm;
    static integer ritzi, ritzr;
    extern /* Subroutine */ int dgeqr2_(integer *, integer *, doublereal *, 
	    integer *, doublereal *, doublereal *, integer *);
    extern doublereal dlapy2_(doublereal *, doublereal *);
    extern /* Subroutine */ int dorm2r_(char *, char *, integer *, integer *, 
	    integer *, doublereal *, integer *, doublereal *, doublereal *, 
	    integer *, doublereal *, integer *, ftnlen, ftnlen);
    static integer iheigi, iheigr, bounds, invsub, iuptri, msglvl, outncv, 
	    ishift, numcnv;
    extern /* Subroutine */ int dlacpy_(char *, integer *, integer *, 
	    doublereal *, integer *, doublereal *, integer *, ftnlen), 
	    dlahqr_(logical *, logical *, integer *, integer *, integer *, 
	    doublereal *, integer *, doublereal *, doublereal *, integer *, 
	    integer *, doublereal *, integer *, integer *), dlaset_(char *, 
	    integer *, integer *, doublereal *, doublereal *, doublereal *, 
	    integer *, ftnlen), pdmout_(integer *, integer *, integer *, 
	    integer *, doublereal *, integer *, integer *, char *, ftnlen), 
	    dtrevc_(char *, char *, logical *, integer *, doublereal *, 
	    integer *, doublereal *, integer *, doublereal *, integer *, 
	    integer *, integer *, doublereal *, integer *, ftnlen, ftnlen), 
	    dtrsen_(char *, char *, logical *, integer *, doublereal *, 
	    integer *, doublereal *, integer *, doublereal *, doublereal *, 
	    integer *, doublereal *, doublereal *, doublereal *, integer *, 
	    integer *, integer *, integer *, ftnlen, ftnlen), pdvout_(integer 
	    *, integer *, integer *, doublereal *, integer *, char *, ftnlen),
	     pivout_(integer *, integer *, integer *, integer *, integer *, 
	    char *, ftnlen);
    extern doublereal pdlamch_(integer *, char *, ftnlen);
    extern /* Subroutine */ int pdngets_(integer *, integer *, char *, 
	    integer *, integer *, doublereal *, doublereal *, doublereal *, 
	    doublereal *, doublereal *, ftnlen);


/*     %--------------------% */
/*     | MPI  Communicator | */
/*     %--------------------% */


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


/*     %---------------% */
/*     | Local Scalars | */
/*     %---------------% */


/*     %----------------------% */
/*     | External Subroutines | */
/*     %----------------------% */


/*     %--------------------% */
/*     | External Functions | */
/*     %--------------------% */


/*     %---------------------% */
/*     | Intrinsic Functions | */
/*     %---------------------% */


/*     %-----------------------% */
/*     | Executable Statements | */
/*     %-----------------------% */

/*     %------------------------% */
/*     | Set default parameters | */
/*     %------------------------% */

    /* Parameter adjustments */
    z_dim1 = *ldz;
    z_offset = 1 + z_dim1;
    z__ -= z_offset;
    --workd;
    --resid;
    --di;
    --dr;
    --workev;
    --select;
    v_dim1 = *ldv;
    v_offset = 1 + v_dim1;
    v -= v_offset;
    --iparam;
    --ipntr;
    --workl;

    /* Function Body */
    msglvl = debug_1.mneupd;
    mode = iparam[7];
    nconv = iparam[5];
    *info = 0;

/*     %---------------------------------% */
/*     | Get machine dependent constant. | */
/*     %---------------------------------% */

    eps23 = pdlamch_(comm, "Epsilon-Machine", (ftnlen)15);
    eps23 = pow_dd(&eps23, &c_b3);

/*     %--------------% */
/*     | Quick return | */
/*     %--------------% */

    ierr = 0;

    if (nconv <= 0) {
	ierr = -14;
    } else if (*n <= 0) {
	ierr = -1;
    } else if (*nev <= 0) {
	ierr = -2;
    } else if (*ncv <= *nev + 1) {
	ierr = -3;
    } else if (s_cmp(which, "LM", (ftnlen)2, (ftnlen)2) != 0 && s_cmp(which, 
	    "SM", (ftnlen)2, (ftnlen)2) != 0 && s_cmp(which, "LR", (ftnlen)2, 
	    (ftnlen)2) != 0 && s_cmp(which, "SR", (ftnlen)2, (ftnlen)2) != 0 
	    && s_cmp(which, "LI", (ftnlen)2, (ftnlen)2) != 0 && s_cmp(which, 
	    "SI", (ftnlen)2, (ftnlen)2) != 0) {
	ierr = -5;
    } else if (*(unsigned char *)bmat != 'I' && *(unsigned char *)bmat != 'G')
	     {
	ierr = -6;
    } else /* if(complicated condition) */ {
/* Computing 2nd power */
	i__1 = *ncv;
	if (*lworkl < i__1 * i__1 * 3 + *ncv * 6) {
	    ierr = -7;
	} else if (*(unsigned char *)howmny != 'A' && *(unsigned char *)
		howmny != 'P' && *(unsigned char *)howmny != 'S' && *rvec) {
	    ierr = -13;
	} else if (*(unsigned char *)howmny == 'S') {
	    ierr = -12;
	}
    }

    if (mode == 1 || mode == 2) {
	s_copy(type__, "REGULR", (ftnlen)6, (ftnlen)6);
    } else if (mode == 3 && *sigmai == 0.) {
	s_copy(type__, "SHIFTI", (ftnlen)6, (ftnlen)6);
    } else if (mode == 3) {
	s_copy(type__, "REALPT", (ftnlen)6, (ftnlen)6);
    } else if (mode == 4) {
	s_copy(type__, "IMAGPT", (ftnlen)6, (ftnlen)6);
    } else {
	ierr = -10;
    }
    if (mode == 1 && *(unsigned char *)bmat == 'G') {
	ierr = -11;
    }

/*     %------------% */
/*     | Error Exit | */
/*     %------------% */

    if (ierr != 0) {
	*info = ierr;
	goto L9000;
    }

/*     %--------------------------------------------------------% */
/*     | Pointer into WORKL for address of H, RITZ, BOUNDS, Q   | */
/*     | etc... and the remaining workspace.                    | */
/*     | Also update pointer to be used on output.              | */
/*     | Memory is laid out as follows:                         | */
/*     | workl(1:ncv*ncv) := generated Hessenberg matrix        | */
/*     | workl(ncv*ncv+1:ncv*ncv+2*ncv) := real and imaginary   | */
/*     |                                   parts of ritz values | */
/*     | workl(ncv*ncv+2*ncv+1:ncv*ncv+3*ncv) := error bounds   | */
/*     %--------------------------------------------------------% */

/*     %-----------------------------------------------------------% */
/*     | The following is used and set by PDNEUPD .                 | */
/*     | workl(ncv*ncv+3*ncv+1:ncv*ncv+4*ncv) := The untransformed | */
/*     |                             real part of the Ritz values. | */
/*     | workl(ncv*ncv+4*ncv+1:ncv*ncv+5*ncv) := The untransformed | */
/*     |                        imaginary part of the Ritz values. | */
/*     | workl(ncv*ncv+5*ncv+1:ncv*ncv+6*ncv) := The untransformed | */
/*     |                           error bounds of the Ritz values | */
/*     | workl(ncv*ncv+6*ncv+1:2*ncv*ncv+6*ncv) := Holds the upper | */
/*     |                             quasi-triangular matrix for H | */
/*     | workl(2*ncv*ncv+6*ncv+1: 3*ncv*ncv+6*ncv) := Holds the    | */
/*     |       associated matrix representation of the invariant   | */
/*     |       subspace for H.                                     | */
/*     | GRAND total of NCV * ( 3 * NCV + 6 ) locations.           | */
/*     %-----------------------------------------------------------% */

    ih = ipntr[5];
    ritzr = ipntr[6];
    ritzi = ipntr[7];
    bounds = ipntr[8];
    ldh = *ncv;
    ldq = *ncv;
    iheigr = bounds + ldh;
    iheigi = iheigr + ldh;
    ihbds = iheigi + ldh;
    iuptri = ihbds + ldh;
    invsub = iuptri + ldh * *ncv;
    ipntr[9] = iheigr;
    ipntr[10] = iheigi;
    ipntr[11] = ihbds;
    ipntr[12] = iuptri;
    ipntr[13] = invsub;
    wrr = 1;
    wri = *ncv + 1;
    iwev = wri + *ncv;

/*     %-----------------------------------------% */
/*     | irr points to the REAL part of the Ritz | */
/*     |     values computed by _neigh before    | */
/*     |     exiting _naup2.                     | */
/*     | iri points to the IMAGINARY part of the | */
/*     |     Ritz values computed by _neigh      | */
/*     |     before exiting _naup2.              | */
/*     | ibd points to the Ritz estimates        | */
/*     |     computed by _neigh before exiting   | */
/*     |     _naup2.                             | */
/*     %-----------------------------------------% */

    irr = ipntr[14] + *ncv * *ncv;
    iri = irr + *ncv;
    ibd = iri + *ncv;

/*     %------------------------------------% */
/*     | RNORM is B-norm of the RESID(1:N). | */
/*     %------------------------------------% */

    rnorm = workl[ih + 2];
    workl[ih + 2] = 0.;

    if (msglvl > 2) {
	pdvout_(comm, &debug_1.logfil, ncv, &workl[irr], &debug_1.ndigit, 
		"_neupd: Real part of Ritz values passed in from _NAUPD.", (
		ftnlen)55);
	pdvout_(comm, &debug_1.logfil, ncv, &workl[iri], &debug_1.ndigit, 
		"_neupd: Imag part of Ritz values passed in from _NAUPD.", (
		ftnlen)55);
	pdvout_(comm, &debug_1.logfil, ncv, &workl[ibd], &debug_1.ndigit, 
		"_neupd: Ritz estimates passed in from _NAUPD.", (ftnlen)45);
    }

    if (*rvec) {

	reord = FALSE_;

/*        %---------------------------------------------------% */
/*        | Use the temporary bounds array to store indices   | */
/*        | These will be used to mark the select array later | */
/*        %---------------------------------------------------% */

	i__1 = *ncv;
	for (j = 1; j <= i__1; ++j) {
	    workl[bounds + j - 1] = (doublereal) j;
	    select[j] = FALSE_;
/* L10: */
	}

/*        %-------------------------------------% */
/*        | Select the wanted Ritz values.      | */
/*        | Sort the Ritz values so that the    | */
/*        | wanted ones appear at the tailing   | */
/*        | NEV positions of workl(irr) and     | */
/*        | workl(iri).  Move the corresponding | */
/*        | error estimates in workl(bound)     | */
/*        | accordingly.                        | */
/*        %-------------------------------------% */

	np = *ncv - *nev;
	ishift = 0;
	pdngets_(comm, &ishift, which, nev, &np, &workl[irr], &workl[iri], &
		workl[bounds], &workl[1], &workl[np + 1], (ftnlen)2);

	if (msglvl > 2) {
	    pdvout_(comm, &debug_1.logfil, ncv, &workl[irr], &debug_1.ndigit, 
		    "_neupd: Real part of Ritz values after calling _NGETS.", 
		    (ftnlen)54);
	    pdvout_(comm, &debug_1.logfil, ncv, &workl[iri], &debug_1.ndigit, 
		    "_neupd: Imag part of Ritz values after calling _NGETS.", 
		    (ftnlen)54);
	    pdvout_(comm, &debug_1.logfil, ncv, &workl[bounds], &
		    debug_1.ndigit, "_neupd: Ritz value indices after callin"
		    "g _NGETS.", (ftnlen)48);
	}

/*        %-----------------------------------------------------% */
/*        | Record indices of the converged wanted Ritz values  | */
/*        | Mark the select array for possible reordering       | */
/*        %-----------------------------------------------------% */

	numcnv = 0;
	i__1 = *ncv;
	for (j = 1; j <= i__1; ++j) {
/* Computing MAX */
	    d__1 = eps23, d__2 = dlapy2_(&workl[irr + *ncv - j], &workl[iri + 
		    *ncv - j]);
	    temp1 = max(d__1,d__2);
	    jj = (integer) workl[bounds + *ncv - j];
	    if (numcnv < nconv && workl[ibd + jj - 1] <= *tol * temp1) {
		select[jj] = TRUE_;
		++numcnv;
		if (jj > *nev) {
		    reord = TRUE_;
		}
	    }
/* L11: */
	}

/*        %-----------------------------------------------------------% */
/*        | Check the count (numcnv) of converged Ritz values with    | */
/*        | the number (nconv) reported by dnaupd.  If these two      | */
/*        | are different then there has probably been an error       | */
/*        | caused by incorrect passing of the dnaupd data.           | */
/*        %-----------------------------------------------------------% */

	if (msglvl > 2) {
	    pivout_(comm, &debug_1.logfil, &c__1, &numcnv, &debug_1.ndigit, 
		    "_neupd: Number of specified eigenvalues", (ftnlen)39);
	    pivout_(comm, &debug_1.logfil, &c__1, &nconv, &debug_1.ndigit, 
		    "_neupd: Number of \"converged\" eigenvalues", (ftnlen)41)
		    ;
	}

	if (numcnv != nconv) {
	    *info = -15;
	    goto L9000;
	}

/*        %-----------------------------------------------------------% */
/*        | Call LAPACK routine dlahqr  to compute the real Schur form | */
/*        | of the upper Hessenberg matrix returned by PDNAUPD .       | */
/*        | Make a copy of the upper Hessenberg matrix.               | */
/*        | Initialize the Schur vector matrix Q to the identity.     | */
/*        %-----------------------------------------------------------% */

	i__1 = ldh * *ncv;
	dcopy_(&i__1, &workl[ih], &c__1, &workl[iuptri], &c__1);
	dlaset_("All", ncv, ncv, &c_b37, &c_b38, &workl[invsub], &ldq, (
		ftnlen)3);
	dlahqr_(&c_true, &c_true, ncv, &c__1, ncv, &workl[iuptri], &ldh, &
		workl[iheigr], &workl[iheigi], &c__1, ncv, &workl[invsub], &
		ldq, &ierr);
	dcopy_(ncv, &workl[invsub + *ncv - 1], &ldq, &workl[ihbds], &c__1);

	if (ierr != 0) {
	    *info = -8;
	    goto L9000;
	}

	if (msglvl > 1) {
	    pdvout_(comm, &debug_1.logfil, ncv, &workl[iheigr], &
		    debug_1.ndigit, "_neupd: Real part of the eigenvalues of"
		    " H", (ftnlen)41);
	    pdvout_(comm, &debug_1.logfil, ncv, &workl[iheigi], &
		    debug_1.ndigit, "_neupd: Imaginary part of the Eigenvalu"
		    "es of H", (ftnlen)46);
	    pdvout_(comm, &debug_1.logfil, ncv, &workl[ihbds], &
		    debug_1.ndigit, "_neupd: Last row of the Schur vector ma"
		    "trix", (ftnlen)43);
	    if (msglvl > 3) {
		pdmout_(comm, &debug_1.logfil, ncv, ncv, &workl[iuptri], &ldh,
			 &debug_1.ndigit, "_neupd: The upper quasi-triangula"
			"r matrix ", (ftnlen)42);
	    }
	}

	if (reord) {

/*           %-----------------------------------------------------% */
/*           | Reorder the computed upper quasi-triangular matrix. | */
/*           %-----------------------------------------------------% */

	    dtrsen_("None", "V", &select[1], ncv, &workl[iuptri], &ldh, &
		    workl[invsub], &ldq, &workl[iheigr], &workl[iheigi], &
		    nconv, &conds, &sep, &workl[ihbds], ncv, iwork, &c__1, &
		    ierr, (ftnlen)4, (ftnlen)1);

	    if (ierr == 1) {
		*info = 1;
		goto L9000;
	    }

	    if (msglvl > 2) {
		pdvout_(comm, &debug_1.logfil, ncv, &workl[iheigr], &
			debug_1.ndigit, "_neupd: Real part of the eigenvalue"
			"s of H--reordered", (ftnlen)52);
		pdvout_(comm, &debug_1.logfil, ncv, &workl[iheigi], &
			debug_1.ndigit, "_neupd: Imag part of the eigenvalue"
			"s of H--reordered", (ftnlen)52);
		if (msglvl > 3) {
		    pdmout_(comm, &debug_1.logfil, ncv, ncv, &workl[iuptri], &
			    ldq, &debug_1.ndigit, "_neupd: Quasi-triangular "
			    "matrix after re-ordering", (ftnlen)49);
		}
	    }
	}

/*        %---------------------------------------% */
/*        | Copy the last row of the Schur vector | */
/*        | into workl(ihbds).  This will be used | */
/*        | to compute the Ritz estimates of      | */
/*        | converged Ritz values.                | */
/*        %---------------------------------------% */

	dcopy_(ncv, &workl[invsub + *ncv - 1], &ldq, &workl[ihbds], &c__1);

/*        %----------------------------------------------------% */
/*        | Place the computed eigenvalues of H into DR and DI | */
/*        | if a spectral transformation was not used.         | */
/*        %----------------------------------------------------% */

	if (s_cmp(type__, "REGULR", (ftnlen)6, (ftnlen)6) == 0) {
	    dcopy_(&nconv, &workl[iheigr], &c__1, &dr[1], &c__1);
	    dcopy_(&nconv, &workl[iheigi], &c__1, &di[1], &c__1);
	}

/*        %----------------------------------------------------------% */
/*        | Compute the QR factorization of the matrix representing  | */
/*        | the wanted invariant subspace located in the first NCONV | */
/*        | columns of workl(invsub,ldq).                            | */
/*        %----------------------------------------------------------% */

	dgeqr2_(ncv, &nconv, &workl[invsub], &ldq, &workev[1], &workev[*ncv + 
		1], &ierr);

/*        %---------------------------------------------------------% */
/*        | * Postmultiply V by Q using dorm2r .                     | */
/*        | * Copy the first NCONV columns of VQ into Z.            | */
/*        | * Postmultiply Z by R.                                  | */
/*        | The N by NCONV matrix Z is now a matrix representation  | */
/*        | of the approximate invariant subspace associated with   | */
/*        | the Ritz values in workl(iheigr) and workl(iheigi)      | */
/*        | The first NCONV columns of V are now approximate Schur  | */
/*        | vectors associated with the real upper quasi-triangular | */
/*        | matrix of order NCONV in workl(iuptri)                  | */
/*        %---------------------------------------------------------% */

	dorm2r_("Right", "Notranspose", n, ncv, &nconv, &workl[invsub], &ldq, 
		&workev[1], &v[v_offset], ldv, &workd[*n + 1], &ierr, (ftnlen)
		5, (ftnlen)11);
	dlacpy_("All", n, &nconv, &v[v_offset], ldv, &z__[z_offset], ldz, (
		ftnlen)3);

	i__1 = nconv;
	for (j = 1; j <= i__1; ++j) {

/*           %---------------------------------------------------% */
/*           | Perform both a column and row scaling if the      | */
/*           | diagonal element of workl(invsub,ldq) is negative | */
/*           | I'm lazy and don't take advantage of the upper    | */
/*           | quasi-triangular form of workl(iuptri,ldq)        | */
/*           | Note that since Q is orthogonal, R is a diagonal  | */
/*           | matrix consisting of plus or minus ones           | */
/*           %---------------------------------------------------% */

	    if (workl[invsub + (j - 1) * ldq + j - 1] < 0.) {
		dscal_(&nconv, &c_b64, &workl[iuptri + j - 1], &ldq);
		dscal_(&nconv, &c_b64, &workl[iuptri + (j - 1) * ldq], &c__1);
	    }

/* L20: */
	}

	if (*(unsigned char *)howmny == 'A') {

/*           %--------------------------------------------% */
/*           | Compute the NCONV wanted eigenvectors of T | */
/*           | located in workl(iuptri,ldq).              | */
/*           %--------------------------------------------% */

	    i__1 = *ncv;
	    for (j = 1; j <= i__1; ++j) {
		if (j <= nconv) {
		    select[j] = TRUE_;
		} else {
		    select[j] = FALSE_;
		}
/* L30: */
	    }

	    dtrevc_("Right", "Select", &select[1], ncv, &workl[iuptri], &ldq, 
		    vl, &c__1, &workl[invsub], &ldq, ncv, &outncv, &workev[1],
		     &ierr, (ftnlen)5, (ftnlen)6);

	    if (ierr != 0) {
		*info = -9;
		goto L9000;
	    }

/*           %------------------------------------------------% */
/*           | Scale the returning eigenvectors so that their | */
/*           | Euclidean norms are all one. LAPACK subroutine | */
/*           | dtrevc  returns each eigenvector normalized so  | */
/*           | that the element of largest magnitude has      | */
/*           | magnitude 1;                                   | */
/*           %------------------------------------------------% */

	    iconj = 0;
	    i__1 = nconv;
	    for (j = 1; j <= i__1; ++j) {

		if (workl[iheigi + j - 1] == 0.) {

/*                 %----------------------% */
/*                 | real eigenvalue case | */
/*                 %----------------------% */

		    temp = dnrm2_(ncv, &workl[invsub + (j - 1) * ldq], &c__1);
		    d__1 = 1. / temp;
		    dscal_(ncv, &d__1, &workl[invsub + (j - 1) * ldq], &c__1);
		} else {

/*                 %-------------------------------------------% */
/*                 | Complex conjugate pair case. Note that    | */
/*                 | since the real and imaginary part of      | */
/*                 | the eigenvector are stored in consecutive | */
/*                 | columns, we further normalize by the      | */
/*                 | square root of two.                       | */
/*                 %-------------------------------------------% */

		    if (iconj == 0) {
			d__1 = dnrm2_(ncv, &workl[invsub + (j - 1) * ldq], &
				c__1);
			d__2 = dnrm2_(ncv, &workl[invsub + j * ldq], &c__1);
			temp = dlapy2_(&d__1, &d__2);
			d__1 = 1. / temp;
			dscal_(ncv, &d__1, &workl[invsub + (j - 1) * ldq], &
				c__1);
			d__1 = 1. / temp;
			dscal_(ncv, &d__1, &workl[invsub + j * ldq], &c__1);
			iconj = 1;
		    } else {
			iconj = 0;
		    }

		}

/* L40: */
	    }

	    dgemv_("T", ncv, &nconv, &c_b38, &workl[invsub], &ldq, &workl[
		    ihbds], &c__1, &c_b37, &workev[1], &c__1, (ftnlen)1);

	    iconj = 0;
	    i__1 = nconv;
	    for (j = 1; j <= i__1; ++j) {
		if (workl[iheigi + j - 1] != 0.) {

/*                 %-------------------------------------------% */
/*                 | Complex conjugate pair case. Note that    | */
/*                 | since the real and imaginary part of      | */
/*                 | the eigenvector are stored in consecutive | */
/*                 %-------------------------------------------% */

		    if (iconj == 0) {
			workev[j] = dlapy2_(&workev[j], &workev[j + 1]);
			workev[j + 1] = workev[j];
			iconj = 1;
		    } else {
			iconj = 0;
		    }
		}
/* L45: */
	    }

	    if (msglvl > 2) {
		pdvout_(comm, &debug_1.logfil, ncv, &workl[ihbds], &
			debug_1.ndigit, "_neupd: Last row of the eigenvector"
			" matrix for T", (ftnlen)48);
		if (msglvl > 3) {
		    pdmout_(comm, &debug_1.logfil, ncv, ncv, &workl[invsub], &
			    ldq, &debug_1.ndigit, "_neupd: The eigenvector m"
			    "atrix for T", (ftnlen)36);
		}
	    }


/*           %---------------------------------------% */
/*           | Copy Ritz estimates into workl(ihbds) | */
/*           %---------------------------------------% */

	    dcopy_(&nconv, &workev[1], &c__1, &workl[ihbds], &c__1);

/*           %---------------------------------------------------------% */
/*           | Compute the QR factorization of the eigenvector matrix  | */
/*           | associated with leading portion of T in the first NCONV | */
/*           | columns of workl(invsub,ldq).                           | */
/*           %---------------------------------------------------------% */

	    dgeqr2_(ncv, &nconv, &workl[invsub], &ldq, &workev[1], &workev[*
		    ncv + 1], &ierr);

/*           %----------------------------------------------% */
/*           | * Postmultiply Z by Q.                       | */
/*           | * Postmultiply Z by R.                       | */
/*           | The N by NCONV matrix Z is now contains the  | */
/*           | Ritz vectors associated with the Ritz values | */
/*           | in workl(iheigr) and workl(iheigi).          | */
/*           %----------------------------------------------% */

	    dorm2r_("Right", "Notranspose", n, ncv, &nconv, &workl[invsub], &
		    ldq, &workev[1], &z__[z_offset], ldz, &workd[*n + 1], &
		    ierr, (ftnlen)5, (ftnlen)11);

	    dtrmm_("Right", "Upper", "No transpose", "Non-unit", n, &nconv, &
		    c_b38, &workl[invsub], &ldq, &z__[z_offset], ldz, (ftnlen)
		    5, (ftnlen)5, (ftnlen)12, (ftnlen)8);

	}

    } else {

/*        %------------------------------------------------------% */
/*        | An approximate invariant subspace is not needed.     | */
/*        | Place the Ritz values computed PDNAUPD  into DR and DI | */
/*        %------------------------------------------------------% */

	dcopy_(&nconv, &workl[ritzr], &c__1, &dr[1], &c__1);
	dcopy_(&nconv, &workl[ritzi], &c__1, &di[1], &c__1);
	dcopy_(&nconv, &workl[ritzr], &c__1, &workl[iheigr], &c__1);
	dcopy_(&nconv, &workl[ritzi], &c__1, &workl[iheigi], &c__1);
	dcopy_(&nconv, &workl[bounds], &c__1, &workl[ihbds], &c__1);
    }

/*     %------------------------------------------------% */
/*     | Transform the Ritz values and possibly vectors | */
/*     | and corresponding error bounds of OP to those  | */
/*     | of A*x = lambda*B*x.                           | */
/*     %------------------------------------------------% */

    if (s_cmp(type__, "REGULR", (ftnlen)6, (ftnlen)6) == 0) {

	if (*rvec) {
	    dscal_(ncv, &rnorm, &workl[ihbds], &c__1);
	}

    } else {

/*        %---------------------------------------% */
/*        |   A spectral transformation was used. | */
/*        | * Determine the Ritz estimates of the | */
/*        |   Ritz values in the original system. | */
/*        %---------------------------------------% */

	if (s_cmp(type__, "SHIFTI", (ftnlen)6, (ftnlen)6) == 0) {

	    if (*rvec) {
		dscal_(ncv, &rnorm, &workl[ihbds], &c__1);
	    }
	    i__1 = *ncv;
	    for (k = 1; k <= i__1; ++k) {
		temp = dlapy2_(&workl[iheigr + k - 1], &workl[iheigi + k - 1])
			;
		workl[ihbds + k - 1] = (d__1 = workl[ihbds + k - 1], abs(d__1)
			) / temp / temp;
/* L50: */
	    }

	} else if (s_cmp(type__, "REALPT", (ftnlen)6, (ftnlen)6) == 0) {

	    i__1 = *ncv;
	    for (k = 1; k <= i__1; ++k) {
/* L60: */
	    }

	} else if (s_cmp(type__, "IMAGPT", (ftnlen)6, (ftnlen)6) == 0) {

	    i__1 = *ncv;
	    for (k = 1; k <= i__1; ++k) {
/* L70: */
	    }

	}

/*        %-----------------------------------------------------------% */
/*        | *  Transform the Ritz values back to the original system. | */
/*        |    For TYPE = 'SHIFTI' the transformation is              | */
/*        |             lambda = 1/theta + sigma                      | */
/*        |    For TYPE = 'REALPT' or 'IMAGPT' the user must from     | */
/*        |    Rayleigh quotients or a projection. See remark 3 above.| */
/*        | NOTES:                                                    | */
/*        | *The Ritz vectors are not affected by the transformation. | */
/*        %-----------------------------------------------------------% */

	if (s_cmp(type__, "SHIFTI", (ftnlen)6, (ftnlen)6) == 0) {

	    i__1 = *ncv;
	    for (k = 1; k <= i__1; ++k) {
		temp = dlapy2_(&workl[iheigr + k - 1], &workl[iheigi + k - 1])
			;
		workl[iheigr + k - 1] = workl[iheigr + k - 1] / temp / temp + 
			*sigmar;
		workl[iheigi + k - 1] = -workl[iheigi + k - 1] / temp / temp 
			+ *sigmai;
/* L80: */
	    }

	    dcopy_(&nconv, &workl[iheigr], &c__1, &dr[1], &c__1);
	    dcopy_(&nconv, &workl[iheigi], &c__1, &di[1], &c__1);

	} else if (s_cmp(type__, "REALPT", (ftnlen)6, (ftnlen)6) == 0 || 
		s_cmp(type__, "IMAGPT", (ftnlen)6, (ftnlen)6) == 0) {

	    dcopy_(&nconv, &workl[iheigr], &c__1, &dr[1], &c__1);
	    dcopy_(&nconv, &workl[iheigi], &c__1, &di[1], &c__1);

	}

	if (s_cmp(type__, "SHIFTI", (ftnlen)6, (ftnlen)6) == 0 && msglvl > 1) 
		{
	    pdvout_(comm, &debug_1.logfil, &nconv, &dr[1], &debug_1.ndigit, 
		    "_neupd: Untransformed real part of the Ritz valuess.", (
		    ftnlen)52);
	    pdvout_(comm, &debug_1.logfil, &nconv, &di[1], &debug_1.ndigit, 
		    "_neupd: Untransformed imag part of the Ritz valuess.", (
		    ftnlen)52);
	    pdvout_(comm, &debug_1.logfil, &nconv, &workl[ihbds], &
		    debug_1.ndigit, "_neupd: Ritz estimates of untransformed"
		    " Ritz values.", (ftnlen)52);
	} else if (s_cmp(type__, "REGULR", (ftnlen)6, (ftnlen)6) == 0 && 
		msglvl > 1) {
	    pdvout_(comm, &debug_1.logfil, &nconv, &dr[1], &debug_1.ndigit, 
		    "_neupd: Real parts of converged Ritz values.", (ftnlen)
		    44);
	    pdvout_(comm, &debug_1.logfil, &nconv, &di[1], &debug_1.ndigit, 
		    "_neupd: Imag parts of converged Ritz values.", (ftnlen)
		    44);
	    pdvout_(comm, &debug_1.logfil, &nconv, &workl[ihbds], &
		    debug_1.ndigit, "_neupd: Associated Ritz estimates.", (
		    ftnlen)34);
	}

    }

/*     %-------------------------------------------------% */
/*     | Eigenvector Purification step. Formally perform | */
/*     | one of inverse subspace iteration. Only used    | */
/*     | for MODE = 2.                                   | */
/*     %-------------------------------------------------% */

    if (*rvec && *(unsigned char *)howmny == 'A' && s_cmp(type__, "SHIFTI", (
	    ftnlen)6, (ftnlen)6) == 0) {

/*        %------------------------------------------------% */
/*        | Purify the computed Ritz vectors by adding a   | */
/*        | little bit of the residual vector:             | */
/*        |                      T                         | */
/*        |          resid(:)*( e    s ) / theta           | */
/*        |                      NCV                       | */
/*        | where H s = s theta. Remember that when theta  | */
/*        | has nonzero imaginary part, the corresponding  | */
/*        | Ritz vector is stored across two columns of Z. | */
/*        %------------------------------------------------% */

	iconj = 0;
	i__1 = nconv;
	for (j = 1; j <= i__1; ++j) {
	    if (workl[iheigi + j - 1] == 0.) {
		workev[j] = workl[invsub + (j - 1) * ldq + *ncv - 1] / workl[
			iheigr + j - 1];
	    } else if (iconj == 0) {
		temp = dlapy2_(&workl[iheigr + j - 1], &workl[iheigi + j - 1])
			;
		workev[j] = (workl[invsub + (j - 1) * ldq + *ncv - 1] * workl[
			iheigr + j - 1] + workl[invsub + j * ldq + *ncv - 1] *
			 workl[iheigi + j - 1]) / temp / temp;
		workev[j + 1] = (workl[invsub + j * ldq + *ncv - 1] * workl[
			iheigr + j - 1] - workl[invsub + (j - 1) * ldq + *ncv 
			- 1] * workl[iheigi + j - 1]) / temp / temp;
		iconj = 1;
	    } else {
		iconj = 0;
	    }
/* L110: */
	}

/*        %---------------------------------------% */
/*        | Perform a rank one update to Z and    | */
/*        | purify all the Ritz vectors together. | */
/*        %---------------------------------------% */

	dger_(n, &nconv, &c_b38, &resid[1], &c__1, &workev[1], &c__1, &z__[
		z_offset], ldz);

    }

L9000:

    return 0;

/*     %----------------% */
/*     | End of PDNEUPD  | */
/*     %----------------% */

} /* pdneupd_ */

