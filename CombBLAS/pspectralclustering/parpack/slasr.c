/* slasr.f -- translated by f2c (version 20050501).
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

/* Subroutine */ int slasr_(char *side, char *pivot, char *direct, integer *m,
	 integer *n, real *c__, real *s, real *a, integer *lda, ftnlen 
	side_len, ftnlen pivot_len, ftnlen direct_len)
{
    /* System generated locals */
    integer a_dim1, a_offset, i__1, i__2;

    /* Local variables */
    static integer i__, j, info;
    static real temp;
    extern logical lsame_(char *, char *, ftnlen, ftnlen);
    static real ctemp, stemp;
    extern /* Subroutine */ int xerbla_(char *, integer *, ftnlen);


/*  -- LAPACK auxiliary routine (version 2.0) -- */
/*     Univ. of Tennessee, Univ. of California Berkeley, NAG Ltd., */
/*     Courant Institute, Argonne National Lab, and Rice University */
/*     October 31, 1992 */

/*     .. Scalar Arguments .. */
/*     .. */
/*     .. Array Arguments .. */
/*     .. */

/*  Purpose */
/*  ======= */

/*  SLASR   performs the transformation */

/*     A := P*A,   when SIDE = 'L' or 'l'  (  Left-hand side ) */

/*     A := A*P',  when SIDE = 'R' or 'r'  ( Right-hand side ) */

/*  where A is an m by n real matrix and P is an orthogonal matrix, */
/*  consisting of a sequence of plane rotations determined by the */
/*  parameters PIVOT and DIRECT as follows ( z = m when SIDE = 'L' or 'l' */
/*  and z = n when SIDE = 'R' or 'r' ): */

/*  When  DIRECT = 'F' or 'f'  ( Forward sequence ) then */

/*     P = P( z - 1 )*...*P( 2 )*P( 1 ), */

/*  and when DIRECT = 'B' or 'b'  ( Backward sequence ) then */

/*     P = P( 1 )*P( 2 )*...*P( z - 1 ), */

/*  where  P( k ) is a plane rotation matrix for the following planes: */

/*     when  PIVOT = 'V' or 'v'  ( Variable pivot ), */
/*        the plane ( k, k + 1 ) */

/*     when  PIVOT = 'T' or 't'  ( Top pivot ), */
/*        the plane ( 1, k + 1 ) */

/*     when  PIVOT = 'B' or 'b'  ( Bottom pivot ), */
/*        the plane ( k, z ) */

/*  c( k ) and s( k )  must contain the  cosine and sine that define the */
/*  matrix  P( k ).  The two by two plane rotation part of the matrix */
/*  P( k ), R( k ), is assumed to be of the form */

/*     R( k ) = (  c( k )  s( k ) ). */
/*              ( -s( k )  c( k ) ) */

/*  This version vectorises across rows of the array A when SIDE = 'L'. */

/*  Arguments */
/*  ========= */

/*  SIDE    (input) CHARACTER*1 */
/*          Specifies whether the plane rotation matrix P is applied to */
/*          A on the left or the right. */
/*          = 'L':  Left, compute A := P*A */
/*          = 'R':  Right, compute A:= A*P' */

/*  DIRECT  (input) CHARACTER*1 */
/*          Specifies whether P is a forward or backward sequence of */
/*          plane rotations. */
/*          = 'F':  Forward, P = P( z - 1 )*...*P( 2 )*P( 1 ) */
/*          = 'B':  Backward, P = P( 1 )*P( 2 )*...*P( z - 1 ) */

/*  PIVOT   (input) CHARACTER*1 */
/*          Specifies the plane for which P(k) is a plane rotation */
/*          matrix. */
/*          = 'V':  Variable pivot, the plane (k,k+1) */
/*          = 'T':  Top pivot, the plane (1,k+1) */
/*          = 'B':  Bottom pivot, the plane (k,z) */

/*  M       (input) INTEGER */
/*          The number of rows of the matrix A.  If m <= 1, an immediate */
/*          return is effected. */

/*  N       (input) INTEGER */
/*          The number of columns of the matrix A.  If n <= 1, an */
/*          immediate return is effected. */

/*  C, S    (input) REAL arrays, dimension */
/*                  (M-1) if SIDE = 'L' */
/*                  (N-1) if SIDE = 'R' */
/*          c(k) and s(k) contain the cosine and sine that define the */
/*          matrix P(k).  The two by two plane rotation part of the */
/*          matrix P(k), R(k), is assumed to be of the form */
/*          R( k ) = (  c( k )  s( k ) ). */
/*                   ( -s( k )  c( k ) ) */

/*  A       (input/output) REAL array, dimension (LDA,N) */
/*          The m by n matrix A.  On exit, A is overwritten by P*A if */
/*          SIDE = 'R' or by A*P' if SIDE = 'L'. */

/*  LDA     (input) INTEGER */
/*          The leading dimension of the array A.  LDA >= max(1,M). */

/*  ===================================================================== */

/*     .. Parameters .. */
/*     .. */
/*     .. Local Scalars .. */
/*     .. */
/*     .. External Functions .. */
/*     .. */
/*     .. External Subroutines .. */
/*     .. */
/*     .. Intrinsic Functions .. */
/*     .. */
/*     .. Executable Statements .. */

/*     Test the input parameters */

    /* Parameter adjustments */
    --c__;
    --s;
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;

    /* Function Body */
    info = 0;
    if (! (lsame_(side, "L", (ftnlen)1, (ftnlen)1) || lsame_(side, "R", (
	    ftnlen)1, (ftnlen)1))) {
	info = 1;
    } else if (! (lsame_(pivot, "V", (ftnlen)1, (ftnlen)1) || lsame_(pivot, 
	    "T", (ftnlen)1, (ftnlen)1) || lsame_(pivot, "B", (ftnlen)1, (
	    ftnlen)1))) {
	info = 2;
    } else if (! (lsame_(direct, "F", (ftnlen)1, (ftnlen)1) || lsame_(direct, 
	    "B", (ftnlen)1, (ftnlen)1))) {
	info = 3;
    } else if (*m < 0) {
	info = 4;
    } else if (*n < 0) {
	info = 5;
    } else if (*lda < max(1,*m)) {
	info = 9;
    }
    if (info != 0) {
	xerbla_("SLASR ", &info, (ftnlen)6);
	return 0;
    }

/*     Quick return if possible */

    if (*m == 0 || *n == 0) {
	return 0;
    }
    if (lsame_(side, "L", (ftnlen)1, (ftnlen)1)) {

/*        Form  P * A */

	if (lsame_(pivot, "V", (ftnlen)1, (ftnlen)1)) {
	    if (lsame_(direct, "F", (ftnlen)1, (ftnlen)1)) {
		i__1 = *m - 1;
		for (j = 1; j <= i__1; ++j) {
		    ctemp = c__[j];
		    stemp = s[j];
		    if (ctemp != 1.f || stemp != 0.f) {
			i__2 = *n;
			for (i__ = 1; i__ <= i__2; ++i__) {
			    temp = a[j + 1 + i__ * a_dim1];
			    a[j + 1 + i__ * a_dim1] = ctemp * temp - stemp * 
				    a[j + i__ * a_dim1];
			    a[j + i__ * a_dim1] = stemp * temp + ctemp * a[j 
				    + i__ * a_dim1];
/* L10: */
			}
		    }
/* L20: */
		}
	    } else if (lsame_(direct, "B", (ftnlen)1, (ftnlen)1)) {
		for (j = *m - 1; j >= 1; --j) {
		    ctemp = c__[j];
		    stemp = s[j];
		    if (ctemp != 1.f || stemp != 0.f) {
			i__1 = *n;
			for (i__ = 1; i__ <= i__1; ++i__) {
			    temp = a[j + 1 + i__ * a_dim1];
			    a[j + 1 + i__ * a_dim1] = ctemp * temp - stemp * 
				    a[j + i__ * a_dim1];
			    a[j + i__ * a_dim1] = stemp * temp + ctemp * a[j 
				    + i__ * a_dim1];
/* L30: */
			}
		    }
/* L40: */
		}
	    }
	} else if (lsame_(pivot, "T", (ftnlen)1, (ftnlen)1)) {
	    if (lsame_(direct, "F", (ftnlen)1, (ftnlen)1)) {
		i__1 = *m;
		for (j = 2; j <= i__1; ++j) {
		    ctemp = c__[j - 1];
		    stemp = s[j - 1];
		    if (ctemp != 1.f || stemp != 0.f) {
			i__2 = *n;
			for (i__ = 1; i__ <= i__2; ++i__) {
			    temp = a[j + i__ * a_dim1];
			    a[j + i__ * a_dim1] = ctemp * temp - stemp * a[
				    i__ * a_dim1 + 1];
			    a[i__ * a_dim1 + 1] = stemp * temp + ctemp * a[
				    i__ * a_dim1 + 1];
/* L50: */
			}
		    }
/* L60: */
		}
	    } else if (lsame_(direct, "B", (ftnlen)1, (ftnlen)1)) {
		for (j = *m; j >= 2; --j) {
		    ctemp = c__[j - 1];
		    stemp = s[j - 1];
		    if (ctemp != 1.f || stemp != 0.f) {
			i__1 = *n;
			for (i__ = 1; i__ <= i__1; ++i__) {
			    temp = a[j + i__ * a_dim1];
			    a[j + i__ * a_dim1] = ctemp * temp - stemp * a[
				    i__ * a_dim1 + 1];
			    a[i__ * a_dim1 + 1] = stemp * temp + ctemp * a[
				    i__ * a_dim1 + 1];
/* L70: */
			}
		    }
/* L80: */
		}
	    }
	} else if (lsame_(pivot, "B", (ftnlen)1, (ftnlen)1)) {
	    if (lsame_(direct, "F", (ftnlen)1, (ftnlen)1)) {
		i__1 = *m - 1;
		for (j = 1; j <= i__1; ++j) {
		    ctemp = c__[j];
		    stemp = s[j];
		    if (ctemp != 1.f || stemp != 0.f) {
			i__2 = *n;
			for (i__ = 1; i__ <= i__2; ++i__) {
			    temp = a[j + i__ * a_dim1];
			    a[j + i__ * a_dim1] = stemp * a[*m + i__ * a_dim1]
				     + ctemp * temp;
			    a[*m + i__ * a_dim1] = ctemp * a[*m + i__ * 
				    a_dim1] - stemp * temp;
/* L90: */
			}
		    }
/* L100: */
		}
	    } else if (lsame_(direct, "B", (ftnlen)1, (ftnlen)1)) {
		for (j = *m - 1; j >= 1; --j) {
		    ctemp = c__[j];
		    stemp = s[j];
		    if (ctemp != 1.f || stemp != 0.f) {
			i__1 = *n;
			for (i__ = 1; i__ <= i__1; ++i__) {
			    temp = a[j + i__ * a_dim1];
			    a[j + i__ * a_dim1] = stemp * a[*m + i__ * a_dim1]
				     + ctemp * temp;
			    a[*m + i__ * a_dim1] = ctemp * a[*m + i__ * 
				    a_dim1] - stemp * temp;
/* L110: */
			}
		    }
/* L120: */
		}
	    }
	}
    } else if (lsame_(side, "R", (ftnlen)1, (ftnlen)1)) {

/*        Form A * P' */

	if (lsame_(pivot, "V", (ftnlen)1, (ftnlen)1)) {
	    if (lsame_(direct, "F", (ftnlen)1, (ftnlen)1)) {
		i__1 = *n - 1;
		for (j = 1; j <= i__1; ++j) {
		    ctemp = c__[j];
		    stemp = s[j];
		    if (ctemp != 1.f || stemp != 0.f) {
			i__2 = *m;
			for (i__ = 1; i__ <= i__2; ++i__) {
			    temp = a[i__ + (j + 1) * a_dim1];
			    a[i__ + (j + 1) * a_dim1] = ctemp * temp - stemp *
				     a[i__ + j * a_dim1];
			    a[i__ + j * a_dim1] = stemp * temp + ctemp * a[
				    i__ + j * a_dim1];
/* L130: */
			}
		    }
/* L140: */
		}
	    } else if (lsame_(direct, "B", (ftnlen)1, (ftnlen)1)) {
		for (j = *n - 1; j >= 1; --j) {
		    ctemp = c__[j];
		    stemp = s[j];
		    if (ctemp != 1.f || stemp != 0.f) {
			i__1 = *m;
			for (i__ = 1; i__ <= i__1; ++i__) {
			    temp = a[i__ + (j + 1) * a_dim1];
			    a[i__ + (j + 1) * a_dim1] = ctemp * temp - stemp *
				     a[i__ + j * a_dim1];
			    a[i__ + j * a_dim1] = stemp * temp + ctemp * a[
				    i__ + j * a_dim1];
/* L150: */
			}
		    }
/* L160: */
		}
	    }
	} else if (lsame_(pivot, "T", (ftnlen)1, (ftnlen)1)) {
	    if (lsame_(direct, "F", (ftnlen)1, (ftnlen)1)) {
		i__1 = *n;
		for (j = 2; j <= i__1; ++j) {
		    ctemp = c__[j - 1];
		    stemp = s[j - 1];
		    if (ctemp != 1.f || stemp != 0.f) {
			i__2 = *m;
			for (i__ = 1; i__ <= i__2; ++i__) {
			    temp = a[i__ + j * a_dim1];
			    a[i__ + j * a_dim1] = ctemp * temp - stemp * a[
				    i__ + a_dim1];
			    a[i__ + a_dim1] = stemp * temp + ctemp * a[i__ + 
				    a_dim1];
/* L170: */
			}
		    }
/* L180: */
		}
	    } else if (lsame_(direct, "B", (ftnlen)1, (ftnlen)1)) {
		for (j = *n; j >= 2; --j) {
		    ctemp = c__[j - 1];
		    stemp = s[j - 1];
		    if (ctemp != 1.f || stemp != 0.f) {
			i__1 = *m;
			for (i__ = 1; i__ <= i__1; ++i__) {
			    temp = a[i__ + j * a_dim1];
			    a[i__ + j * a_dim1] = ctemp * temp - stemp * a[
				    i__ + a_dim1];
			    a[i__ + a_dim1] = stemp * temp + ctemp * a[i__ + 
				    a_dim1];
/* L190: */
			}
		    }
/* L200: */
		}
	    }
	} else if (lsame_(pivot, "B", (ftnlen)1, (ftnlen)1)) {
	    if (lsame_(direct, "F", (ftnlen)1, (ftnlen)1)) {
		i__1 = *n - 1;
		for (j = 1; j <= i__1; ++j) {
		    ctemp = c__[j];
		    stemp = s[j];
		    if (ctemp != 1.f || stemp != 0.f) {
			i__2 = *m;
			for (i__ = 1; i__ <= i__2; ++i__) {
			    temp = a[i__ + j * a_dim1];
			    a[i__ + j * a_dim1] = stemp * a[i__ + *n * a_dim1]
				     + ctemp * temp;
			    a[i__ + *n * a_dim1] = ctemp * a[i__ + *n * 
				    a_dim1] - stemp * temp;
/* L210: */
			}
		    }
/* L220: */
		}
	    } else if (lsame_(direct, "B", (ftnlen)1, (ftnlen)1)) {
		for (j = *n - 1; j >= 1; --j) {
		    ctemp = c__[j];
		    stemp = s[j];
		    if (ctemp != 1.f || stemp != 0.f) {
			i__1 = *m;
			for (i__ = 1; i__ <= i__1; ++i__) {
			    temp = a[i__ + j * a_dim1];
			    a[i__ + j * a_dim1] = stemp * a[i__ + *n * a_dim1]
				     + ctemp * temp;
			    a[i__ + *n * a_dim1] = ctemp * a[i__ + *n * 
				    a_dim1] - stemp * temp;
/* L230: */
			}
		    }
/* L240: */
		}
	    }
	}
    }

    return 0;

/*     End of SLASR */

} /* slasr_ */

