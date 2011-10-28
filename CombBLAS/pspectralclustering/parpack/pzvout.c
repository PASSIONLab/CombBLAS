/* pzvout.f -- translated by f2c (version 20050501).
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
static integer c__2 = 2;

/*  Routine:    PZVOUT - Parallel Version of ARPACK utility routine ZVOUT */

/*  Purpose:    Complex*16 vector output routine. */

/*  Usage:      CALL PZVOUT (COMM, LOUT, N, CX, IDIGIT, IFMT) */

/*  Arguments */
/*     COMM   - MPI Communicator for the processor grid */
/*     N      - Length of array CX.  (Input) */
/*     CX     - Complex*16 array to be printed.  (Input) */
/*     IFMT   - Format to be used in printing array CX.  (Input) */
/*     IDIGIT - Print up to IABS(IDIGIT) decimal digits per number.  (In) */
/*              If IDIGIT .LT. 0, printing is done with 72 columns. */
/*              If IDIGIT .GT. 0, printing is done with 132 columns. */

/* FILE: vout.F   SID: 1.1   DATE OF SID: 3/19/97   RELEASE: 1 */

/* ----------------------------------------------------------------------- */

/* Subroutine */ int pzvout_(integer *comm, integer *lout, integer *n, 
	doublecomplex *cx, integer *idigit, char *ifmt, ftnlen ifmt_len)
{
    /* Format strings */
    static char fmt_9999[] = "(/1x,a/1x,a)";
    static char fmt_9998[] = "(1x,i4,\002 - \002,i4,\002:\002,1x,1p,2(\002"
	    "(\002,d10.3,\002,\002,d10.3,\002)  \002))";
    static char fmt_9997[] = "(1x,i4,\002 - \002,i4,\002:\002,1x,1p,1(\002"
	    "(\002,d10.3,\002,\002,d10.3,\002)  \002))";
    static char fmt_9988[] = "(1x,i4,\002 - \002,i4,\002:\002,1x,1p,2(\002"
	    "(\002,d12.5,\002,\002,d12.5,\002)  \002))";
    static char fmt_9987[] = "(1x,i4,\002 - \002,i4,\002:\002,1x,1p,1(\002"
	    "(\002,d12.5,\002,\002,d12.5,\002)  \002))";
    static char fmt_9978[] = "(1x,i4,\002 - \002,i4,\002:\002,1x,1p,2(\002"
	    "(\002,d14.7,\002,\002,d14.7,\002)  \002))";
    static char fmt_9977[] = "(1x,i4,\002 - \002,i4,\002:\002,1x,1p,1(\002"
	    "(\002,d14.7,\002,\002,d14.7,\002)  \002))";
    static char fmt_9968[] = "(1x,i4,\002 - \002,i4,\002:\002,1x,1p,1(\002"
	    "(\002,d20.13,\002,\002,d20.13,\002)  \002))";
    static char fmt_9958[] = "(1x,i4,\002 - \002,i4,\002:\002,1x,1p,4(\002"
	    "(\002,d10.3,\002,\002,d10.3,\002)  \002))";
    static char fmt_9957[] = "(1x,i4,\002 - \002,i4,\002:\002,1x,1p,3(\002"
	    "(\002,d10.3,\002,\002,d10.3,\002)  \002))";
    static char fmt_9956[] = "(1x,i4,\002 - \002,i4,\002:\002,1x,1p,2(\002"
	    "(\002,d10.3,\002,\002,d10.3,\002)  \002))";
    static char fmt_9955[] = "(1x,i4,\002 - \002,i4,\002:\002,1x,1p,1(\002"
	    "(\002,d10.3,\002,\002,d10.3,\002)  \002))";
    static char fmt_9948[] = "(1x,i4,\002 - \002,i4,\002:\002,1x,1p,3(\002"
	    "(\002,d12.5,\002,\002,d12.5,\002)  \002))";
    static char fmt_9947[] = "(1x,i4,\002 - \002,i4,\002:\002,1x,1p,2(\002"
	    "(\002,d12.5,\002,\002,d12.5,\002)  \002))";
    static char fmt_9946[] = "(1x,i4,\002 - \002,i4,\002:\002,1x,1p,1(\002"
	    "(\002,d12.5,\002,\002,d12.5,\002)  \002))";
    static char fmt_9938[] = "(1x,i4,\002 - \002,i4,\002:\002,1x,1p,3(\002"
	    "(\002,d14.7,\002,\002,d14.7,\002)  \002))";
    static char fmt_9937[] = "(1x,i4,\002 - \002,i4,\002:\002,1x,1p,2(\002"
	    "(\002,d14.7,\002,\002,d14.7,\002)  \002))";
    static char fmt_9936[] = "(1x,i4,\002 - \002,i4,\002:\002,1x,1p,1(\002"
	    "(\002,d14.7,\002,\002,d14.7,\002)  \002))";
    static char fmt_9928[] = "(1x,i4,\002 - \002,i4,\002:\002,1x,1p,2(\002"
	    "(\002,d20.13,\002,\002,d20.13,\002)  \002))";
    static char fmt_9927[] = "(1x,i4,\002 - \002,i4,\002:\002,1x,1p,1(\002"
	    "(\002,d20.13,\002,\002,d20.13,\002)  \002))";
    static char fmt_9994[] = "(1x,\002 \002)";

    /* System generated locals */
    integer i__1, i__2, i__3;

    /* Builtin functions */
    integer i_len(char *, ftnlen), s_wsfe(cilist *), do_fio(integer *, char *,
	     ftnlen), e_wsfe(void);

    /* Local variables */
    static integer i__, k1, k2, lll;
    static char line[80];
    static integer ierr, myid;
    extern /* Subroutine */ int mpi_comm_rank__(integer *, integer *, integer 
	    *);
    static integer ndigit;

    /* Fortran I/O blocks */
    static cilist io___6 = { 0, 0, 0, fmt_9999, 0 };
    static cilist io___10 = { 0, 0, 0, fmt_9998, 0 };
    static cilist io___11 = { 0, 0, 0, fmt_9997, 0 };
    static cilist io___12 = { 0, 0, 0, fmt_9988, 0 };
    static cilist io___13 = { 0, 0, 0, fmt_9987, 0 };
    static cilist io___14 = { 0, 0, 0, fmt_9978, 0 };
    static cilist io___15 = { 0, 0, 0, fmt_9977, 0 };
    static cilist io___16 = { 0, 0, 0, fmt_9968, 0 };
    static cilist io___17 = { 0, 0, 0, fmt_9958, 0 };
    static cilist io___18 = { 0, 0, 0, fmt_9957, 0 };
    static cilist io___19 = { 0, 0, 0, fmt_9956, 0 };
    static cilist io___20 = { 0, 0, 0, fmt_9955, 0 };
    static cilist io___21 = { 0, 0, 0, fmt_9948, 0 };
    static cilist io___22 = { 0, 0, 0, fmt_9947, 0 };
    static cilist io___23 = { 0, 0, 0, fmt_9946, 0 };
    static cilist io___24 = { 0, 0, 0, fmt_9938, 0 };
    static cilist io___25 = { 0, 0, 0, fmt_9937, 0 };
    static cilist io___26 = { 0, 0, 0, fmt_9936, 0 };
    static cilist io___27 = { 0, 0, 0, fmt_9928, 0 };
    static cilist io___28 = { 0, 0, 0, fmt_9927, 0 };
    static cilist io___29 = { 0, 0, 0, fmt_9994, 0 };


/*     ... */

/*     .. MPI VARIABLES AND FUNCTIONS .. */
/*     .. Variable Declaration .. */
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

/*     ... SPECIFICATIONS FOR ARGUMENTS */
/*     ... */
/*     ... SPECIFICATIONS FOR LOCAL VARIABLES */
/*     ... */
/*     ... FIRST EXECUTABLE STATEMENT */


/*     Determine processor configuration */

    /* Parameter adjustments */
    --cx;

    /* Function Body */
    mpi_comm_rank__(comm, &myid, &ierr);

/*     .. Only Processor 0 will write to file LOUT .. */

    if (myid == 0) {

/* Computing MIN */
	i__1 = i_len(ifmt, ifmt_len);
	lll = min(i__1,80);
	i__1 = lll;
	for (i__ = 1; i__ <= i__1; ++i__) {
	    *(unsigned char *)&line[i__ - 1] = '-';
/* L10: */
	}

	for (i__ = lll + 1; i__ <= 80; ++i__) {
	    *(unsigned char *)&line[i__ - 1] = ' ';
/* L20: */
	}

	io___6.ciunit = *lout;
	s_wsfe(&io___6);
	do_fio(&c__1, ifmt, ifmt_len);
	do_fio(&c__1, line, lll);
	e_wsfe();

	if (*n <= 0) {
	    return 0;
	}
	ndigit = *idigit;
	if (*idigit == 0) {
	    ndigit = 4;
	}

/* ======================================================================= */
/*             CODE FOR OUTPUT USING 72 COLUMNS FORMAT */
/* ======================================================================= */

	if (*idigit < 0) {
	    ndigit = -(*idigit);
	    if (ndigit <= 4) {
		i__1 = *n;
		for (k1 = 1; k1 <= i__1; k1 += 2) {
/* Computing MIN */
		    i__2 = *n, i__3 = k1 + 1;
		    k2 = min(i__2,i__3);
		    if (k1 != *n) {
			io___10.ciunit = *lout;
			s_wsfe(&io___10);
			do_fio(&c__1, (char *)&k1, (ftnlen)sizeof(integer));
			do_fio(&c__1, (char *)&k2, (ftnlen)sizeof(integer));
			i__2 = k2;
			for (i__ = k1; i__ <= i__2; ++i__) {
			    do_fio(&c__2, (char *)&cx[i__], (ftnlen)sizeof(
				    doublereal));
			}
			e_wsfe();
		    } else {
			io___11.ciunit = *lout;
			s_wsfe(&io___11);
			do_fio(&c__1, (char *)&k1, (ftnlen)sizeof(integer));
			do_fio(&c__1, (char *)&k2, (ftnlen)sizeof(integer));
			i__2 = k2;
			for (i__ = k1; i__ <= i__2; ++i__) {
			    do_fio(&c__2, (char *)&cx[i__], (ftnlen)sizeof(
				    doublereal));
			}
			e_wsfe();
		    }
/* L30: */
		}
	    } else if (ndigit <= 6) {
		i__1 = *n;
		for (k1 = 1; k1 <= i__1; k1 += 2) {
/* Computing MIN */
		    i__2 = *n, i__3 = k1 + 1;
		    k2 = min(i__2,i__3);
		    if (k1 != *n) {
			io___12.ciunit = *lout;
			s_wsfe(&io___12);
			do_fio(&c__1, (char *)&k1, (ftnlen)sizeof(integer));
			do_fio(&c__1, (char *)&k2, (ftnlen)sizeof(integer));
			i__2 = k2;
			for (i__ = k1; i__ <= i__2; ++i__) {
			    do_fio(&c__2, (char *)&cx[i__], (ftnlen)sizeof(
				    doublereal));
			}
			e_wsfe();
		    } else {
			io___13.ciunit = *lout;
			s_wsfe(&io___13);
			do_fio(&c__1, (char *)&k1, (ftnlen)sizeof(integer));
			do_fio(&c__1, (char *)&k2, (ftnlen)sizeof(integer));
			i__2 = k2;
			for (i__ = k1; i__ <= i__2; ++i__) {
			    do_fio(&c__2, (char *)&cx[i__], (ftnlen)sizeof(
				    doublereal));
			}
			e_wsfe();
		    }
/* L40: */
		}
	    } else if (ndigit <= 8) {
		i__1 = *n;
		for (k1 = 1; k1 <= i__1; k1 += 2) {
/* Computing MIN */
		    i__2 = *n, i__3 = k1 + 1;
		    k2 = min(i__2,i__3);
		    if (k1 != *n) {
			io___14.ciunit = *lout;
			s_wsfe(&io___14);
			do_fio(&c__1, (char *)&k1, (ftnlen)sizeof(integer));
			do_fio(&c__1, (char *)&k2, (ftnlen)sizeof(integer));
			i__2 = k2;
			for (i__ = k1; i__ <= i__2; ++i__) {
			    do_fio(&c__2, (char *)&cx[i__], (ftnlen)sizeof(
				    doublereal));
			}
			e_wsfe();
		    } else {
			io___15.ciunit = *lout;
			s_wsfe(&io___15);
			do_fio(&c__1, (char *)&k1, (ftnlen)sizeof(integer));
			do_fio(&c__1, (char *)&k2, (ftnlen)sizeof(integer));
			i__2 = k2;
			for (i__ = k1; i__ <= i__2; ++i__) {
			    do_fio(&c__2, (char *)&cx[i__], (ftnlen)sizeof(
				    doublereal));
			}
			e_wsfe();
		    }
/* L50: */
		}
	    } else {
		i__1 = *n;
		for (k1 = 1; k1 <= i__1; ++k1) {
		    io___16.ciunit = *lout;
		    s_wsfe(&io___16);
		    do_fio(&c__1, (char *)&k1, (ftnlen)sizeof(integer));
		    do_fio(&c__1, (char *)&k1, (ftnlen)sizeof(integer));
		    do_fio(&c__2, (char *)&cx[i__], (ftnlen)sizeof(doublereal)
			    );
		    e_wsfe();
/* L60: */
		}
	    }

/* ======================================================================= */
/*             CODE FOR OUTPUT USING 132 COLUMNS FORMAT */
/* ======================================================================= */

	} else {
	    if (ndigit <= 4) {
		i__1 = *n;
		for (k1 = 1; k1 <= i__1; k1 += 4) {
/* Computing MIN */
		    i__2 = *n, i__3 = k1 + 3;
		    k2 = min(i__2,i__3);
		    if (k1 + 3 <= *n) {
			io___17.ciunit = *lout;
			s_wsfe(&io___17);
			do_fio(&c__1, (char *)&k1, (ftnlen)sizeof(integer));
			do_fio(&c__1, (char *)&k2, (ftnlen)sizeof(integer));
			i__2 = k2;
			for (i__ = k1; i__ <= i__2; ++i__) {
			    do_fio(&c__2, (char *)&cx[i__], (ftnlen)sizeof(
				    doublereal));
			}
			e_wsfe();
		    } else if (k1 + 3 - *n == 1) {
			io___18.ciunit = *lout;
			s_wsfe(&io___18);
			do_fio(&c__1, (char *)&k1, (ftnlen)sizeof(integer));
			do_fio(&c__1, (char *)&k2, (ftnlen)sizeof(integer));
			i__2 = k2;
			for (i__ = k1; i__ <= i__2; ++i__) {
			    do_fio(&c__2, (char *)&cx[i__], (ftnlen)sizeof(
				    doublereal));
			}
			e_wsfe();
		    } else if (k1 + 3 - *n == 2) {
			io___19.ciunit = *lout;
			s_wsfe(&io___19);
			do_fio(&c__1, (char *)&k1, (ftnlen)sizeof(integer));
			do_fio(&c__1, (char *)&k2, (ftnlen)sizeof(integer));
			i__2 = k2;
			for (i__ = k1; i__ <= i__2; ++i__) {
			    do_fio(&c__2, (char *)&cx[i__], (ftnlen)sizeof(
				    doublereal));
			}
			e_wsfe();
		    } else if (k1 + 3 - *n == 1) {
			io___20.ciunit = *lout;
			s_wsfe(&io___20);
			do_fio(&c__1, (char *)&k1, (ftnlen)sizeof(integer));
			do_fio(&c__1, (char *)&k2, (ftnlen)sizeof(integer));
			i__2 = k2;
			for (i__ = k1; i__ <= i__2; ++i__) {
			    do_fio(&c__2, (char *)&cx[i__], (ftnlen)sizeof(
				    doublereal));
			}
			e_wsfe();
		    }
/* L70: */
		}
	    } else if (ndigit <= 6) {
		i__1 = *n;
		for (k1 = 1; k1 <= i__1; k1 += 3) {
/* Computing MIN */
		    i__2 = *n, i__3 = k1 + 2;
		    k2 = min(i__2,i__3);
		    if (k1 + 2 <= *n) {
			io___21.ciunit = *lout;
			s_wsfe(&io___21);
			do_fio(&c__1, (char *)&k1, (ftnlen)sizeof(integer));
			do_fio(&c__1, (char *)&k2, (ftnlen)sizeof(integer));
			i__2 = k2;
			for (i__ = k1; i__ <= i__2; ++i__) {
			    do_fio(&c__2, (char *)&cx[i__], (ftnlen)sizeof(
				    doublereal));
			}
			e_wsfe();
		    } else if (k1 + 2 - *n == 1) {
			io___22.ciunit = *lout;
			s_wsfe(&io___22);
			do_fio(&c__1, (char *)&k1, (ftnlen)sizeof(integer));
			do_fio(&c__1, (char *)&k2, (ftnlen)sizeof(integer));
			i__2 = k2;
			for (i__ = k1; i__ <= i__2; ++i__) {
			    do_fio(&c__2, (char *)&cx[i__], (ftnlen)sizeof(
				    doublereal));
			}
			e_wsfe();
		    } else if (k1 + 2 - *n == 2) {
			io___23.ciunit = *lout;
			s_wsfe(&io___23);
			do_fio(&c__1, (char *)&k1, (ftnlen)sizeof(integer));
			do_fio(&c__1, (char *)&k2, (ftnlen)sizeof(integer));
			i__2 = k2;
			for (i__ = k1; i__ <= i__2; ++i__) {
			    do_fio(&c__2, (char *)&cx[i__], (ftnlen)sizeof(
				    doublereal));
			}
			e_wsfe();
		    }
/* L80: */
		}
	    } else if (ndigit <= 8) {
		i__1 = *n;
		for (k1 = 1; k1 <= i__1; k1 += 3) {
/* Computing MIN */
		    i__2 = *n, i__3 = k1 + 2;
		    k2 = min(i__2,i__3);
		    if (k1 + 2 <= *n) {
			io___24.ciunit = *lout;
			s_wsfe(&io___24);
			do_fio(&c__1, (char *)&k1, (ftnlen)sizeof(integer));
			do_fio(&c__1, (char *)&k2, (ftnlen)sizeof(integer));
			i__2 = k2;
			for (i__ = k1; i__ <= i__2; ++i__) {
			    do_fio(&c__2, (char *)&cx[i__], (ftnlen)sizeof(
				    doublereal));
			}
			e_wsfe();
		    } else if (k1 + 2 - *n == 1) {
			io___25.ciunit = *lout;
			s_wsfe(&io___25);
			do_fio(&c__1, (char *)&k1, (ftnlen)sizeof(integer));
			do_fio(&c__1, (char *)&k2, (ftnlen)sizeof(integer));
			i__2 = k2;
			for (i__ = k1; i__ <= i__2; ++i__) {
			    do_fio(&c__2, (char *)&cx[i__], (ftnlen)sizeof(
				    doublereal));
			}
			e_wsfe();
		    } else if (k1 + 2 - *n == 2) {
			io___26.ciunit = *lout;
			s_wsfe(&io___26);
			do_fio(&c__1, (char *)&k1, (ftnlen)sizeof(integer));
			do_fio(&c__1, (char *)&k2, (ftnlen)sizeof(integer));
			i__2 = k2;
			for (i__ = k1; i__ <= i__2; ++i__) {
			    do_fio(&c__2, (char *)&cx[i__], (ftnlen)sizeof(
				    doublereal));
			}
			e_wsfe();
		    }
/* L90: */
		}
	    } else {
		i__1 = *n;
		for (k1 = 1; k1 <= i__1; k1 += 2) {
/* Computing MIN */
		    i__2 = *n, i__3 = k1 + 1;
		    k2 = min(i__2,i__3);
		    if (k1 + 2 <= *n) {
			io___27.ciunit = *lout;
			s_wsfe(&io___27);
			do_fio(&c__1, (char *)&k1, (ftnlen)sizeof(integer));
			do_fio(&c__1, (char *)&k2, (ftnlen)sizeof(integer));
			i__2 = k2;
			for (i__ = k1; i__ <= i__2; ++i__) {
			    do_fio(&c__2, (char *)&cx[i__], (ftnlen)sizeof(
				    doublereal));
			}
			e_wsfe();
		    } else if (k1 + 2 - *n == 1) {
			io___28.ciunit = *lout;
			s_wsfe(&io___28);
			do_fio(&c__1, (char *)&k1, (ftnlen)sizeof(integer));
			do_fio(&c__1, (char *)&k2, (ftnlen)sizeof(integer));
			i__2 = k2;
			for (i__ = k1; i__ <= i__2; ++i__) {
			    do_fio(&c__2, (char *)&cx[i__], (ftnlen)sizeof(
				    doublereal));
			}
			e_wsfe();
		    }
/* L100: */
		}
	    }
	}
	io___29.ciunit = *lout;
	s_wsfe(&io___29);
	e_wsfe();


    }
    return 0;

/* ======================================================================= */
/*                   FORMAT FOR 72 COLUMNS */
/* ======================================================================= */

/*                 DISPLAY 4 SIGNIFICANT DIGITS */


/*                 DISPLAY 6 SIGNIFICANT DIGITS */


/*                 DISPLAY 8 SIGNIFICANT DIGITS */


/*                 DISPLAY 13 SIGNIFICANT DIGITS */


/* ========================================================================= */
/*                   FORMAT FOR 132 COLUMNS */
/* ========================================================================= */

/*                 DISPLAY 4 SIGNIFICANT DIGITS */


/*                 DISPLAY 6 SIGNIFICANT DIGITS */


/*                 DISPLAY 8 SIGNIFICANT DIGITS */


/*                 DISPLAY 13 SIGNIFICANT DIGITS */




} /* pzvout_ */

