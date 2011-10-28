/* pdlamch.f -- translated by f2c (version 20050501).
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

/* Table of constant values */

static integer c__1 = 1;

doublereal pdlamch_(integer *ictxt, char *cmach, ftnlen cmach_len)
{
    /* System generated locals */
    doublereal ret_val;

    /* Local variables */
    static doublereal temp, temp1;
    extern logical lsame_(char *, char *, ftnlen, ftnlen);
    static integer idumm;
    extern /* Subroutine */ int mpi_allreduce__(doublereal *, doublereal *, 
	    integer *, integer *, integer *, integer *, integer *);
    extern doublereal dlamch_(char *, ftnlen);


/*  -- ScaLAPACK auxilliary routine (version 1.0) -- */
/*     University of Tennessee, Knoxville, Oak Ridge National Laboratory, */
/*     and University of California, Berkeley. */
/*     February 28, 1995 */

/*     .. Scalar Arguments .. */
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
/*     .. */

/*  Purpose */
/*  ======= */

/*  PDLAMCH determines double precision machine parameters. */

/*  Arguments */
/*  ========= */

/*  ICTXT   (global input) INTEGER */
/*          The BLACS context handle in which the computation takes */
/*          place. */

/*  CMACH   (global input) CHARACTER*1 */
/*          Specifies the value to be returned by PDLAMCH: */
/*          = 'E' or 'e',   PDLAMCH := eps */
/*          = 'S' or 's ,   PDLAMCH := sfmin */
/*          = 'B' or 'b',   PDLAMCH := base */
/*          = 'P' or 'p',   PDLAMCH := eps*base */
/*          = 'N' or 'n',   PDLAMCH := t */
/*          = 'R' or 'r',   PDLAMCH := rnd */
/*          = 'M' or 'm',   PDLAMCH := emin */
/*          = 'U' or 'u',   PDLAMCH := rmin */
/*          = 'L' or 'l',   PDLAMCH := emax */
/*          = 'O' or 'o',   PDLAMCH := rmax */

/*          where */

/*          eps   = relative machine precision */
/*          sfmin = safe minimum, such that 1/sfmin does not overflow */
/*          base  = base of the machine */
/*          prec  = eps*base */
/*          t     = number of (base) digits in the mantissa */
/*          rnd   = 1.0 when rounding occurs in addition, 0.0 otherwise */
/*          emin  = minimum exponent before (gradual) underflow */
/*          rmin  = underflow threshold - base**(emin-1) */
/*          emax  = largest exponent before overflow */
/*          rmax  = overflow threshold  - (base**emax)*(1-eps) */

/*  ===================================================================== */

/*     .. Local Scalars .. */
/*     .. */
/*     .. External Subroutines .. */
/*      EXTERNAL           DGAMN2D, DGAMX2D */
/*     .. */
/*     .. External Functions .. */
/*     .. */
/*     .. Executable Statements .. */

    temp1 = dlamch_(cmach, (ftnlen)1);

    if (lsame_(cmach, "E", (ftnlen)1, (ftnlen)1) || lsame_(cmach, "S", (
	    ftnlen)1, (ftnlen)1) || lsame_(cmach, "M", (ftnlen)1, (ftnlen)1) 
	    || lsame_(cmach, "U", (ftnlen)1, (ftnlen)1)) {
	mpi_allreduce__(&temp1, &temp, &c__1, &
		mpipriv_1.mpi_double_precision__, &mpipriv_1.mpi_max__, ictxt,
		 &idumm);
/*         CALL DGAMX2D( ICTXT, 'All', ' ', 1, 1, TEMP, 1, IDUMM, */
/*     $                 IDUMM, 1, -1, IDUMM ) */
    } else if (lsame_(cmach, "L", (ftnlen)1, (ftnlen)1) || lsame_(cmach, 
	    "O", (ftnlen)1, (ftnlen)1)) {
	mpi_allreduce__(&temp1, &temp, &c__1, &
		mpipriv_1.mpi_double_precision__, &mpipriv_1.mpi_min__, ictxt,
		 &idumm);
/*         CALL DGAMN2D( ICTXT, 'All', ' ', 1, 1, TEMP, 1, IDUMM, */
/*     $                 IDUMM, 1, -1, IDUMM ) */
    } else {
	temp = temp1;
    }

    ret_val = temp;

/*     End of PDLAMCH */

    return ret_val;
} /* pdlamch_ */

