/* spttrs.f -- translated by f2c (version 20050501).
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

/* Subroutine */ int spttrs_(integer *n, integer *nrhs, real *d__, real *e, 
	real *b, integer *ldb, integer *info)
{
    /* System generated locals */
    integer b_dim1, b_offset, i__1, i__2;

    /* Local variables */
    static integer i__, j;
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

/*  SPTTRS solves a system of linear equations A * X = B with a */
/*  symmetric positive definite tridiagonal matrix A using the */
/*  factorization A = L*D*L**T or A = U**T*D*U computed by SPTTRF. */
/*  (The two forms are equivalent if A is real.) */

/*  Arguments */
/*  ========= */

/*  N       (input) INTEGER */
/*          The order of the tridiagonal matrix A.  N >= 0. */

/*  NRHS    (input) INTEGER */
/*          The number of right hand sides, i.e., the number of columns */
/*          of the matrix B.  NRHS >= 0. */

/*  D       (input) REAL array, dimension (N) */
/*          The n diagonal elements of the diagonal matrix D from the */
/*          factorization computed by SPTTRF. */

/*  E       (input) REAL array, dimension (N-1) */
/*          The (n-1) off-diagonal elements of the unit bidiagonal factor */
/*          U or L from the factorization computed by SPTTRF. */

/*  B       (input/output) REAL array, dimension (LDB,NRHS) */
/*          On entry, the right hand side matrix B. */
/*          On exit, the solution matrix X. */

/*  LDB     (input) INTEGER */
/*          The leading dimension of the array B.  LDB >= max(1,N). */

/*  INFO    (output) INTEGER */
/*          = 0:  successful exit */
/*          < 0:  if INFO = -i, the i-th argument had an illegal value */

/*  ===================================================================== */

/*     .. Local Scalars .. */
/*     .. */
/*     .. External Subroutines .. */
/*     .. */
/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. Executable Statements .. */

/*     Test the input arguments. */

    /* Parameter adjustments */
    --d__;
    --e;
    b_dim1 = *ldb;
    b_offset = 1 + b_dim1;
    b -= b_offset;

    /* Function Body */
    *info = 0;
    if (*n < 0) {
	*info = -1;
    } else if (*nrhs < 0) {
	*info = -2;
    } else if (*ldb < max(1,*n)) {
	*info = -6;
    }
    if (*info != 0) {
	i__1 = -(*info);
	xerbla_("SPTTRS", &i__1, (ftnlen)6);
	return 0;
    }

/*     Quick return if possible */

    if (*n == 0) {
	return 0;
    }

/*     Solve A * X = B using the factorization A = L*D*L', */
/*     overwriting each right hand side vector with its solution. */

    i__1 = *nrhs;
    for (j = 1; j <= i__1; ++j) {

/*        Solve L * x = b. */

	i__2 = *n;
	for (i__ = 2; i__ <= i__2; ++i__) {
	    b[i__ + j * b_dim1] -= b[i__ - 1 + j * b_dim1] * e[i__ - 1];
/* L10: */
	}

/*        Solve D * L' * x = b. */

	b[*n + j * b_dim1] /= d__[*n];
	for (i__ = *n - 1; i__ >= 1; --i__) {
	    b[i__ + j * b_dim1] = b[i__ + j * b_dim1] / d__[i__] - b[i__ + 1 
		    + j * b_dim1] * e[i__];
/* L20: */
	}
/* L30: */
    }

    return 0;

/*     End of SPTTRS */

} /* spttrs_ */

