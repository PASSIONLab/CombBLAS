/* spttrf.f -- translated by f2c (version 20050501).
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

/* Subroutine */ int spttrf_(integer *n, real *d__, real *e, integer *info)
{
    /* System generated locals */
    integer i__1;

    /* Local variables */
    static integer i__;
    static real di, ei;
    extern /* Subroutine */ int xerbla_(char *, integer *, ftnlen);


/*  -- LAPACK routine (version 2.0) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd., */
/*     Courant Institute, Argonne National Lab, and Rice University */
/*     March 31, 1993 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  SPTTRF computes the factorization of a real symmetric positive */
/*  definite tridiagonal matrix A. */

/*  If the subdiagonal elements of A are supplied in the array E, the */
/*  factorization has the form A = L*D*L**T, where D is diagonal and L */
/*  is unit lower bidiagonal; if the superdiagonal elements of A are */
/*  supplied, it has the form A = U**T*D*U, where U is unit upper */
/*  bidiagonal.  (The two forms are equivalent if A is real.) */

/*  Arguments */
/*  ========= */

/*  N       (input) INTEGER */
/*          The order of the matrix A.  N >= 0. */

/*  D       (input/output) REAL array, dimension (N) */
/*          On entry, the n diagonal elements of the tridiagonal matrix */
/*          A.  On exit, the n diagonal elements of the diagonal matrix */
/*          D from the L*D*L**T factorization of A. */

/*  E       (input/output) REAL array, dimension (N-1) */
/*          On entry, the (n-1) off-diagonal elements of the tridiagonal */
/*          matrix A. */
/*          On exit, the (n-1) off-diagonal elements of the unit */
/*          bidiagonal factor L or U from the factorization of A. */

/*  INFO    (output) INTEGER */
/*          = 0:  successful exit */
/*          < 0:  if INFO = -i, the i-th argument had an illegal value */
/*          > 0:  if INFO = i, the leading minor of order i is not */
/*                positive definite; if i < N, the factorization could */
/*                not be completed, while if i = N, the factorization was */
/*                completed, but D(N) = 0. */

/*  ===================================================================== */

/*     .. Parameters .. */
/*     .. */
/*     .. Local Scalars .. */
/*     .. */
/*     .. External Subroutines .. */
/*     .. */
/*     .. Executable Statements .. */

/*     Test the input parameters. */

    /* Parameter adjustments */
    --e;
    --d__;

    /* Function Body */
    *info = 0;
    if (*n < 0) {
	*info = -1;
	i__1 = -(*info);
	xerbla_("SPTTRF", &i__1, (ftnlen)6);
	return 0;
    }

/*     Quick return if possible */

    if (*n == 0) {
	return 0;
    }

/*     Compute the L*D*L' (or U'*D*U) factorization of A. */

    i__1 = *n - 1;
    for (i__ = 1; i__ <= i__1; ++i__) {

/*        Drop out of the loop if d(i) <= 0: the matrix is not positive */
/*        definite. */

	di = d__[i__];
	if (di <= 0.f) {
	    goto L20;
	}

/*        Solve for e(i) and d(i+1). */

	ei = e[i__];
	e[i__] = ei / di;
	d__[i__ + 1] -= e[i__] * ei;
/* L10: */
    }

/*     Check d(n) for positive definiteness. */

    i__ = *n;
    if (d__[i__] > 0.f) {
	goto L30;
    }

L20:
    *info = i__;

L30:
    return 0;

/*     End of SPTTRF */

} /* spttrf_ */

