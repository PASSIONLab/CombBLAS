/* dgttrf.f -- translated by f2c (version 20050501).
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

/* Subroutine */ int dgttrf_(integer *n, doublereal *dl, doublereal *d__, 
	doublereal *du, doublereal *du2, integer *ipiv, integer *info)
{
    /* System generated locals */
    integer i__1;
    doublereal d__1, d__2;

    /* Local variables */
    static integer i__;
    static doublereal fact, temp;
    extern /* Subroutine */ int xerbla_(char *, integer *, ftnlen);


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

/*  DGTTRF computes an LU factorization of a real tridiagonal matrix A */
/*  using elimination with partial pivoting and row interchanges. */

/*  The factorization has the form */
/*     A = L * U */
/*  where L is a product of permutation and unit lower bidiagonal */
/*  matrices and U is upper triangular with nonzeros in only the main */
/*  diagonal and first two superdiagonals. */

/*  Arguments */
/*  ========= */

/*  N       (input) INTEGER */
/*          The order of the matrix A.  N >= 0. */

/*  DL      (input/output) DOUBLE PRECISION array, dimension (N-1) */
/*          On entry, DL must contain the (n-1) subdiagonal elements of */
/*          A. */
/*          On exit, DL is overwritten by the (n-1) multipliers that */
/*          define the matrix L from the LU factorization of A. */

/*  D       (input/output) DOUBLE PRECISION array, dimension (N) */
/*          On entry, D must contain the diagonal elements of A. */
/*          On exit, D is overwritten by the n diagonal elements of the */
/*          upper triangular matrix U from the LU factorization of A. */

/*  DU      (input/output) DOUBLE PRECISION array, dimension (N-1) */
/*          On entry, DU must contain the (n-1) superdiagonal elements */
/*          of A. */
/*          On exit, DU is overwritten by the (n-1) elements of the first */
/*          superdiagonal of U. */

/*  DU2     (output) DOUBLE PRECISION array, dimension (N-2) */
/*          On exit, DU2 is overwritten by the (n-2) elements of the */
/*          second superdiagonal of U. */

/*  IPIV    (output) INTEGER array, dimension (N) */
/*          The pivot indices; for 1 <= i <= n, row i of the matrix was */
/*          interchanged with row IPIV(i).  IPIV(i) will always be either */
/*          i or i+1; IPIV(i) = i indicates a row interchange was not */
/*          required. */

/*  INFO    (output) INTEGER */
/*          = 0:  successful exit */
/*          < 0:  if INFO = -i, the i-th argument had an illegal value */
/*          > 0:  if INFO = i, U(i,i) is exactly zero. The factorization */
/*                has been completed, but the factor U is exactly */
/*                singular, and division by zero will occur if it is used */
/*                to solve a system of equations. */

/*  ===================================================================== */

/*     .. Local Scalars .. */
/*     .. */
/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. External Subroutines .. */
/*     .. */
/*     .. Parameters .. */
/*     .. */
/*     .. Executable Statements .. */

    /* Parameter adjustments */
    --ipiv;
    --du2;
    --du;
    --d__;
    --dl;

    /* Function Body */
    *info = 0;
    if (*n < 0) {
	*info = -1;
	i__1 = -(*info);
	xerbla_("DGTTRF", &i__1, (ftnlen)6);
	return 0;
    }

/*     Quick return if possible */

    if (*n == 0) {
	return 0;
    }

/*     Initialize IPIV(i) = i */

    i__1 = *n;
    for (i__ = 1; i__ <= i__1; ++i__) {
	ipiv[i__] = i__;
/* L10: */
    }

    i__1 = *n - 1;
    for (i__ = 1; i__ <= i__1; ++i__) {
	if (dl[i__] == 0.) {

/*           Subdiagonal is zero, no elimination is required. */

	    if (d__[i__] == 0. && *info == 0) {
		*info = i__;
	    }
	    if (i__ < *n - 1) {
		du2[i__] = 0.;
	    }
	} else if ((d__1 = d__[i__], abs(d__1)) >= (d__2 = dl[i__], abs(d__2))
		) {

/*           No row interchange required, eliminate DL(I) */

	    fact = dl[i__] / d__[i__];
	    dl[i__] = fact;
	    d__[i__ + 1] -= fact * du[i__];
	    if (i__ < *n - 1) {
		du2[i__] = 0.;
	    }
	} else {

/*           Interchange rows I and I+1, eliminate DL(I) */

	    fact = d__[i__] / dl[i__];
	    d__[i__] = dl[i__];
	    dl[i__] = fact;
	    temp = du[i__];
	    du[i__] = d__[i__ + 1];
	    d__[i__ + 1] = temp - fact * d__[i__ + 1];
	    if (i__ < *n - 1) {
		du2[i__] = du[i__ + 1];
		du[i__ + 1] = -fact * du[i__ + 1];
	    }
	    ++ipiv[i__];
	}
/* L20: */
    }
    if (d__[*n] == 0. && *info == 0) {
	*info = *n;
	return 0;
    }

    return 0;

/*     End of DGTTRF */

} /* dgttrf_ */

