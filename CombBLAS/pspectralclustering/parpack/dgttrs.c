/* dgttrs.f -- translated by f2c (version 20050501).
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

/* Subroutine */ int dgttrs_(char *trans, integer *n, integer *nrhs, 
	doublereal *dl, doublereal *d__, doublereal *du, doublereal *du2, 
	integer *ipiv, doublereal *b, integer *ldb, integer *info, ftnlen 
	trans_len)
{
    /* System generated locals */
    integer b_dim1, b_offset, i__1, i__2;

    /* Local variables */
    static integer i__, j;
    static doublereal temp;
    extern logical lsame_(char *, char *, ftnlen, ftnlen);
    extern /* Subroutine */ int xerbla_(char *, integer *, ftnlen);
    static logical notran;


/*  -- LAPACK routine (version 2.0) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd., */
/*     Courant Institute, Argonne National Lab, and Rice University */
/*     September 30, 1994 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  DGTTRS solves one of the systems of equations */
/*     A*X = B  or  A'*X = B, */
/*  with a tridiagonal matrix A using the LU factorization computed */
/*  by DGTTRF. */

/*  Arguments */
/*  ========= */

/*  TRANS   (input) CHARACTER */
/*          Specifies the form of the system of equations: */
/*          = 'N':  A * X = B  (No transpose) */
/*          = 'T':  A'* X = B  (Transpose) */
/*          = 'C':  A'* X = B  (Conjugate transpose = Transpose) */

/*  N       (input) INTEGER */
/*          The order of the matrix A.  N >= 0. */

/*  NRHS    (input) INTEGER */
/*          The number of right hand sides, i.e., the number of columns */
/*          of the matrix B.  NRHS >= 0. */

/*  DL      (input) DOUBLE PRECISION array, dimension (N-1) */
/*          The (n-1) multipliers that define the matrix L from the */
/*          LU factorization of A. */

/*  D       (input) DOUBLE PRECISION array, dimension (N) */
/*          The n diagonal elements of the upper triangular matrix U from */
/*          the LU factorization of A. */

/*  DU      (input) DOUBLE PRECISION array, dimension (N-1) */
/*          The (n-1) elements of the first superdiagonal of U. */

/*  DU2     (input) DOUBLE PRECISION array, dimension (N-2) */
/*          The (n-2) elements of the second superdiagonal of U. */

/*  IPIV    (input) INTEGER array, dimension (N) */
/*          The pivot indices; for 1 <= i <= n, row i of the matrix was */
/*          interchanged with row IPIV(i).  IPIV(i) will always be either */
/*          i or i+1; IPIV(i) = i indicates a row interchange was not */
/*          required. */

/*  B       (input/output) DOUBLE PRECISION array, dimension (LDB,NRHS) */
/*          On entry, the right hand side matrix B. */
/*          On exit, B is overwritten by the solution matrix X. */

/*  LDB     (input) INTEGER */
/*          The leading dimension of the array B.  LDB >= max(1,N). */

/*  INFO    (output) INTEGER */
/*          = 0:  successful exit */
/*          < 0:  if INFO = -i, the i-th argument had an illegal value */

/*  ===================================================================== */

/*     .. Local Scalars .. */
/*     .. */
/*     .. External Functions .. */
/*     .. */
/*     .. External Subroutines .. */
/*     .. */
/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. Executable Statements .. */

    /* Parameter adjustments */
    --dl;
    --d__;
    --du;
    --du2;
    --ipiv;
    b_dim1 = *ldb;
    b_offset = 1 + b_dim1;
    b -= b_offset;

    /* Function Body */
    *info = 0;
    notran = lsame_(trans, "N", (ftnlen)1, (ftnlen)1);
    if (! notran && ! lsame_(trans, "T", (ftnlen)1, (ftnlen)1) && ! lsame_(
	    trans, "C", (ftnlen)1, (ftnlen)1)) {
	*info = -1;
    } else if (*n < 0) {
	*info = -2;
    } else if (*nrhs < 0) {
	*info = -3;
    } else if (*ldb < max(*n,1)) {
	*info = -10;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("DGTTRS", &i__1, (ftnlen)6);
	return 0;
    }

/*     Quick return if possible */

    if (*n == 0 || *nrhs == 0) {
	return 0;
    }

    if (notran) {

/*        Solve A*X = B using the LU factorization of A, */
/*        overwriting each right hand side vector with its solution. */

	i__1 = *nrhs;
	for (j = 1; j <= i__1; ++j) {

/*           Solve L*x = b. */

	    i__2 = *n - 1;
	    for (i__ = 1; i__ <= i__2; ++i__) {
		if (ipiv[i__] == i__) {
		    b[i__ + 1 + j * b_dim1] -= dl[i__] * b[i__ + j * b_dim1];
		} else {
		    temp = b[i__ + j * b_dim1];
		    b[i__ + j * b_dim1] = b[i__ + 1 + j * b_dim1];
		    b[i__ + 1 + j * b_dim1] = temp - dl[i__] * b[i__ + j * 
			    b_dim1];
		}
/* L10: */
	    }

/*           Solve U*x = b. */

	    b[*n + j * b_dim1] /= d__[*n];
	    if (*n > 1) {
		b[*n - 1 + j * b_dim1] = (b[*n - 1 + j * b_dim1] - du[*n - 1] 
			* b[*n + j * b_dim1]) / d__[*n - 1];
	    }
	    for (i__ = *n - 2; i__ >= 1; --i__) {
		b[i__ + j * b_dim1] = (b[i__ + j * b_dim1] - du[i__] * b[i__ 
			+ 1 + j * b_dim1] - du2[i__] * b[i__ + 2 + j * b_dim1]
			) / d__[i__];
/* L20: */
	    }
/* L30: */
	}
    } else {

/*        Solve A' * X = B. */

	i__1 = *nrhs;
	for (j = 1; j <= i__1; ++j) {

/*           Solve U'*x = b. */

	    b[j * b_dim1 + 1] /= d__[1];
	    if (*n > 1) {
		b[j * b_dim1 + 2] = (b[j * b_dim1 + 2] - du[1] * b[j * b_dim1 
			+ 1]) / d__[2];
	    }
	    i__2 = *n;
	    for (i__ = 3; i__ <= i__2; ++i__) {
		b[i__ + j * b_dim1] = (b[i__ + j * b_dim1] - du[i__ - 1] * b[
			i__ - 1 + j * b_dim1] - du2[i__ - 2] * b[i__ - 2 + j *
			 b_dim1]) / d__[i__];
/* L40: */
	    }

/*           Solve L'*x = b. */

	    for (i__ = *n - 1; i__ >= 1; --i__) {
		if (ipiv[i__] == i__) {
		    b[i__ + j * b_dim1] -= dl[i__] * b[i__ + 1 + j * b_dim1];
		} else {
		    temp = b[i__ + 1 + j * b_dim1];
		    b[i__ + 1 + j * b_dim1] = b[i__ + j * b_dim1] - dl[i__] * 
			    temp;
		    b[i__ + j * b_dim1] = temp;
		}
/* L50: */
	    }
/* L60: */
	}
    }

/*     End of DGTTRS */

    return 0;
} /* dgttrs_ */

