/* pdnorm2.f -- translated by f2c (version 20050501).
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
static doublereal c_b3 = 2.;

/* \BeginDoc */

/* \Name: pdnorm2 */

/* Message Passing Layer: MPI */

/* \Description: */

/* \Usage: */
/*  call pdnorm2 ( COMM, N, X, INC ) */

/* \Arguments */
/*  COMM    MPI Communicator for the processor grid.  (INPUT) */

/* \SCCS Information: */
/* FILE: norm2.F   SID: 1.2   DATE OF SID: 2/22/96 */

/* ----------------------------------------------------------------------- */

doublereal pdnorm2_(integer *comm, integer *n, doublereal *x, integer *inc)
{
    /* System generated locals */
    doublereal ret_val, d__1;

    /* Builtin functions */
    double pow_dd(doublereal *, doublereal *), sqrt(doublereal);

    /* Local variables */
    static doublereal buf, max__;
    static integer ierr;
    extern doublereal dnrm2_(integer *, doublereal *, integer *);
    extern /* Subroutine */ int mpi_allreduce__(doublereal *, doublereal *, 
	    integer *, integer *, integer *, integer *, integer *);



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

/*     %------------------% */
/*     | Scalar Arguments | */
/*     %------------------% */


/*     %-----------------% */
/*     | Array Arguments | */
/*     %-----------------% */


/*     %---------------% */
/*     | Local Scalars | */
/*     %---------------% */


/*     %---------------------% */
/*     | Intrinsic Functions | */
/*     %---------------------% */


/*     %--------------------% */
/*     | External Functions | */
/*     %--------------------% */


/*     %-----------------------% */
/*     | Executable Statements | */
/*     %-----------------------% */

    /* Parameter adjustments */
    --x;

    /* Function Body */
    ret_val = dnrm2_(n, &x[1], inc);

    buf = ret_val;
    mpi_allreduce__(&buf, &max__, &c__1, &mpipriv_1.mpi_double_precision__, &
	    mpipriv_1.mpi_max__, comm, &ierr);
    if (max__ == 0.) {
	ret_val = 0.;
    } else {
	d__1 = ret_val / max__;
	buf = pow_dd(&d__1, &c_b3);
	mpi_allreduce__(&buf, &ret_val, &c__1, &
		mpipriv_1.mpi_double_precision__, &mpipriv_1.mpi_sum__, comm, 
		&ierr);
	ret_val = max__ * sqrt((abs(ret_val)));
    }

/*     %----------------% */
/*     | End of pdnorm2 | */
/*     %----------------% */

    return ret_val;
} /* pdnorm2_ */

