#define PAPI_VERSION_NUMBER(maj,min,rev,inc) (((maj)<<24) | ((min)<<16) | ((rev)<<8) | (inc))
#define PAPI_VERSION_MAJOR(x)   	(((x)>>24)    & 0xff)
#define PAPI_VERSION_MINOR(x)		(((x)>>16)    & 0xff)
#define PAPI_VERSION_REVISION(x)	(((x)>>8)     & 0xff)
#define PAPI_VERSION_INCREMENT(x)((x)          & 0xff)
#define PAPI_VERSION  			PAPI_VERSION_NUMBER(4,2,0,0)
#define PAPI_VER_CURRENT 		(PAPI_VERSION & 0xffff0000)
#define IS_NATIVE( EventCode ) ( ( EventCode & PAPI_NATIVE_MASK ) && !(EventCode & PAPI_PRESET_MASK) )
#define IS_PRESET( EventCode ) ( ( EventCode & PAPI_PRESET_MASK ) && !(EventCode & PAPI_NATIVE_MASK) )
#define IS_USER_DEFINED( EventCode ) ( ( EventCode & PAPI_PRESET_MASK ) && (EventCode & PAPI_NATIVE_MASK) )
#define PAPI_OK          0     /**< No error */
#define PAPI_EINVAL     -1     /**< Invalid argument */
#define PAPI_ENOMEM     -2     /**< Insufficient memory */
#define PAPI_ESYS       -3     /**< A System/C library call failed */
#define PAPI_ESBSTR     -4     /**< Not supported by substrate */
#define PAPI_ECLOST     -5     /**< Access to the counters was lost or interrupted */
#define PAPI_EBUG       -6     /**< Internal error, please send mail to the developers */
#define PAPI_ENOEVNT    -7     /**< Event does not exist */
#define PAPI_ECNFLCT    -8     /**< Event exists, but cannot be counted due to counter resource limitations */
#define PAPI_ENOTRUN    -9     /**< EventSet is currently not running */
#define PAPI_EISRUN     -10    /**< EventSet is currently counting */
#define PAPI_ENOEVST    -11    /**< No such EventSet Available */
#define PAPI_ENOTPRESET -12    /**< Event in argument is not a valid preset */
#define PAPI_ENOCNTR    -13    /**< Hardware does not support performance counters */
#define PAPI_EMISC      -14    /**< Unknown error code */
#define PAPI_EPERM      -15    /**< Permission level does not permit operation */
#define PAPI_ENOINIT    -16    /**< PAPI hasn't been initialized yet */
#define PAPI_ENOCMP     -17    /**< Component Index isn't set */
#define PAPI_ENOSUPP    -18    /**< Not supported */
#define PAPI_ENOIMPL    -19    /**< Not implemented */
#define PAPI_EBUF       -20    /**< Buffer size exceeded */
#define PAPI_EINVAL_DOM -21    /**< EventSet domain is not supported for the operation */
#define PAPI_EATTR		-22    /**< Invalid or missing event attributes */
#define PAPI_ECOUNT		-23    /**< Too many events or attributes */
#define PAPI_ECOMBO		-24    /**< Bad combination of features */
#define PAPI_NUM_ERRORS	 25    /**< Number of error messages specified in this API */
#define PAPI_LOW_LEVEL_INITED 	1       /* Low level has called library init */
#define PAPI_HIGH_LEVEL_INITED 	2       /* High level has called library init */
#define PAPI_THREAD_LEVEL_INITED 4      /* Threads have been inited */
#define PAPI_NULL       -1      /**<A nonexistent hardware event used as a placeholder */
#define PAPI_DOM_USER    0x1    /**< User context counted */
#define PAPI_DOM_MIN     PAPI_DOM_USER
#define PAPI_DOM_KERNEL	 0x2    /**< Kernel/OS context counted */
#define PAPI_DOM_OTHER	 0x4    /**< Exception/transient mode (like user TLB misses ) */
#define PAPI_DOM_SUPERVISOR 0x8 /**< Supervisor/hypervisor context counted */
#define PAPI_DOM_ALL	 (PAPI_DOM_USER|PAPI_DOM_KERNEL|PAPI_DOM_OTHER|PAPI_DOM_SUPERVISOR) /**< All contexts counted */
#define PAPI_DOM_MAX     PAPI_DOM_ALL
#define PAPI_DOM_HWSPEC  0x80000000     /**< Flag that indicates we are not reading CPU like stuff.*/
#define PAPI_HIGH_LEVEL_TLS     0x2
#define PAPI_TLS_HIGH_LEVEL     PAPI_HIGH_LEVEL_TLS
#define PAPI_USR1_LOCK          	0x0    /**< User controlled locks */
#define PAPI_USR2_LOCK          	0x1    /**< User controlled locks */
#define PAPI_NUM_LOCK           	0x2    /**< Used with setting up array */
#define PAPI_LOCK_USR1          	PAPI_USR1_LOCK
#define PAPI_LOCK_USR2          	PAPI_USR2_LOCK
#define PAPI_MPX_DEF_DEG 32			   /* Maximum number of counters we can mpx */
#define PAPI_VENDOR_UNKNOWN 0
#define PAPI_VENDOR_INTEL   1
#define PAPI_VENDOR_AMD     2
#define PAPI_VENDOR_IBM     3
#define PAPI_VENDOR_CRAY    4
#define PAPI_VENDOR_SUN     5
#define PAPI_VENDOR_FREESCALE 6
#define PAPI_VENDOR_ARM     7
#define PAPI_VENDOR_MIPS    8
#define PAPI_GRN_THR     0x1    /**< PAPI counters for each individual thread */
#define PAPI_GRN_MIN     PAPI_GRN_THR
#define PAPI_GRN_PROC    0x2    /**< PAPI counters for each individual process */
#define PAPI_GRN_PROCG   0x4    /**< PAPI counters for each individual process group */
#define PAPI_GRN_SYS     0x8    /**< PAPI counters for the current CPU, are you bound? */
#define PAPI_GRN_SYS_CPU 0x10   /**< PAPI counters for all CPU's individually */
#define PAPI_GRN_MAX     PAPI_GRN_SYS_CPU
#define PAPI_PER_CPU     1			   /*Counts are accumulated on a per cpu basis */
#define PAPI_PER_NODE    2			   /*Counts are accumulated on a per node or */
#define PAPI_SYSTEM	 3				   /*Counts are accumulated for events occuring in */
#define PAPI_PER_THR     0			   /*Counts are accumulated on a per kernel thread basis */
#define PAPI_PER_PROC    1			   /*Counts are accumulated on a per process basis */
#define PAPI_ONESHOT	 1			   /*Option to the overflow handler 2b called once */
#define PAPI_RANDOMIZE	 2			   /*Option to have the threshold of the overflow*/
#define PAPI_STOPPED      0x01  /**< EventSet stopped */
#define PAPI_RUNNING      0x02  /**< EventSet running */
#define PAPI_PAUSED       0x04  /**< EventSet temp. disabled by the library */
#define PAPI_NOT_INIT     0x08  /**< EventSet defined, but not initialized */
#define PAPI_OVERFLOWING  0x10  /**< EventSet has overflowing enabled */
#define PAPI_PROFILING    0x20  /**< EventSet has profiling enabled */
#define PAPI_MULTIPLEXING 0x40  /**< EventSet has multiplexing enabled */
#define PAPI_ATTACHED	  0x80  /**< EventSet is attached to another thread/process */
#define PAPI_CPU_ATTACHED 0x100 /**< EventSet is attached to a specific cpu (not counting thread of execution) */
#define PAPI_QUIET       0      /**< Option to turn off automatic reporting of return codes < 0 to stderr. */
#define PAPI_VERB_ECONT  1      /**< Option to automatically report any return codes < 0 to stderr and continue. */
#define PAPI_VERB_ESTOP  2      /**< Option to automatically report any return codes < 0 to stderr and exit. */
#define PAPI_PROFIL_POSIX     0x0        /**< Default type of profiling, similar to 'man profil'. */
#define PAPI_PROFIL_RANDOM    0x1        /**< Drop a random 25% of the samples. */
#define PAPI_PROFIL_WEIGHTED  0x2        /**< Weight the samples by their value. */
#define PAPI_PROFIL_COMPRESS  0x4        /**< Ignore samples if hash buckets get big. */
#define PAPI_PROFIL_BUCKET_16 0x8        /**< Use 16 bit buckets to accumulate profile info (default) */
#define PAPI_PROFIL_BUCKET_32 0x10       /**< Use 32 bit buckets to accumulate profile info */
#define PAPI_PROFIL_BUCKET_64 0x20       /**< Use 64 bit buckets to accumulate profile info */
#define PAPI_PROFIL_FORCE_SW  0x40       /**< Force Software overflow in profiling */
#define PAPI_PROFIL_DATA_EAR  0x80       /**< Use data address register profiling */
#define PAPI_PROFIL_INST_EAR  0x100      /**< Use instruction address register profiling */
#define PAPI_PROFIL_BUCKETS   (PAPI_PROFIL_BUCKET_16 | PAPI_PROFIL_BUCKET_32 | PAPI_PROFIL_BUCKET_64)
#define PAPI_OVERFLOW_FORCE_SW 0x40	/**< Force using Software */
#define PAPI_OVERFLOW_HARDWARE 0x80	/**< Using Hardware */
#define PAPI_MULTIPLEX_DEFAULT	0x0	/**< Use whatever method is available, prefer kernel of course. */
#define PAPI_MULTIPLEX_FORCE_SW 0x1	/**< Force PAPI multiplexing instead of kernel */
#define PAPI_INHERIT_ALL  1     /**< The flag to this to inherit all children's counters */
#define PAPI_INHERIT_NONE 0     /**< The flag to this to inherit none of the children's counters */
#define PAPI_DETACH			1		/**< Detach */
#define PAPI_DEBUG          2       /**< Option to turn on  debugging features of the PAPI library */
#define PAPI_MULTIPLEX 		3       /**< Turn on/off or multiplexing for an eventset */
#define PAPI_DEFDOM  		4       /**< Domain for all new eventsets. Takes non-NULL option pointer. */
#define PAPI_DOMAIN  		5       /**< Domain for an eventset */
#define PAPI_DEFGRN  		6       /**< Granularity for all new eventsets */
#define PAPI_GRANUL  		7       /**< Granularity for an eventset */
#define PAPI_DEF_MPX_NS     8       /**< Multiplexing/overflowing interval in ns, same as PAPI_DEF_ITIMER_NS */
#define PAPI_EDGE_DETECT    9       /**< Count cycles of events if supported <not implemented> */
#define PAPI_INVERT         10		/**< Invert count detect if supported <not implemented> */
#define PAPI_MAX_MPX_CTRS	11      /**< Maximum number of counters we can multiplex */
#define PAPI_PROFIL  		12      /**< Option to turn on the overflow/profil reporting software <not implemented> */
#define PAPI_PRELOAD 		13      /**< Option to find out the environment variable that can preload libraries */
#define PAPI_CLOCKRATE  	14      /**< Clock rate in MHz */
#define PAPI_MAX_HWCTRS 	15      /**< Number of physical hardware counters */
#define PAPI_HWINFO  		16      /**< Hardware information */
#define PAPI_EXEINFO  		17      /**< Executable information */
#define PAPI_MAX_CPUS 		18      /**< Number of ncpus we can talk to from here */
#define PAPI_ATTACH			19      /**< Attach to a another tid/pid instead of ourself */
#define PAPI_SHLIBINFO      20      /**< Shared Library information */
#define PAPI_LIB_VERSION    21      /**< Option to find out the complete version number of the PAPI library */
#define PAPI_COMPONENTINFO  22      /**< Find out what the component substrate supports */
#define PAPI_DATA_ADDRESS   23      /**< Option to set data address range restriction */
#define PAPI_INSTR_ADDRESS  24      /**< Option to set instruction address range restriction */
#define PAPI_DEF_ITIMER		25		/**< Option to set the type of itimer used in both software multiplexing, overflowing and profiling */
#define PAPI_DEF_ITIMER_NS	26		/**< Multiplexing/overflowing interval in ns, same as PAPI_DEF_MPX_NS */
#define PAPI_CPU_ATTACH		27      /**< Specify a cpu number the event set should be tied to */
#define PAPI_INHERIT		28      /**< Option to set counter inheritance flag */
#define PAPI_USER_EVENTS_FILE 29	/**< Option to set file from where to parse user defined events */
#define PAPI_INIT_SLOTS    64     /*Number of initialized slots in*/
#define PAPI_MIN_STR_LEN        64      /* For small strings, like names & stuff */
#define PAPI_MAX_STR_LEN       128      /* For average run-of-the-mill strings */
#define PAPI_2MAX_STR_LEN      256      /* For somewhat longer run-of-the-mill strings */
#define PAPI_HUGE_STR_LEN     1024      /* This should be defined in terms of a system parameter */
#define PAPI_DERIVED           0x1      /* Flag to indicate that the event is derived */
#define PAPI_ENUM_ALL PAPI_ENUM_EVENTS
#define PAPI_PRESET_BIT_MSC		(1 << PAPI_PRESET_ENUM_MSC)	/* Miscellaneous preset event bit */
#define PAPI_PRESET_BIT_INS		(1 << PAPI_PRESET_ENUM_INS)	/* Instruction related preset event bit */
#define PAPI_PRESET_BIT_IDL		(1 << PAPI_PRESET_ENUM_IDL)	/* Stalled or Idle preset event bit */
#define PAPI_PRESET_BIT_BR		(1 << PAPI_PRESET_ENUM_BR)	/* branch related preset events */
#define PAPI_PRESET_BIT_CND		(1 << PAPI_PRESET_ENUM_CND)	/* conditional preset events */
#define PAPI_PRESET_BIT_MEM		(1 << PAPI_PRESET_ENUM_MEM)	/* memory related preset events */
#define PAPI_PRESET_BIT_CACH	(1 << PAPI_PRESET_ENUM_CACH)	/* cache related preset events */
#define PAPI_PRESET_BIT_L1		(1 << PAPI_PRESET_ENUM_L1)	/* L1 cache related preset events */
#define PAPI_PRESET_BIT_L2		(1 << PAPI_PRESET_ENUM_L2)	/* L2 cache related preset events */
#define PAPI_PRESET_BIT_L3		(1 << PAPI_PRESET_ENUM_L3)	/* L3 cache related preset events */
#define PAPI_PRESET_BIT_TLB		(1 << PAPI_PRESET_ENUM_TLB)	/* Translation Lookaside Buffer events */
#define PAPI_PRESET_BIT_FP		(1 << PAPI_PRESET_ENUM_FP)	/* Floating Point related preset events */
#define PAPI_NTV_GROUP_AND_MASK		0x00FF0000	/* bits occupied by group number */
#define PAPI_NTV_GROUP_SHIFT		16			/* bit shift to encode group number */
#define long_long long long
#define u_long_long unsigned long long
#define PAPI_MH_TYPE_EMPTY    0x0
#define PAPI_MH_TYPE_INST     0x1
#define PAPI_MH_TYPE_DATA     0x2
#define PAPI_MH_TYPE_VECTOR   0x4
#define PAPI_MH_TYPE_TRACE    0x8
#define PAPI_MH_TYPE_UNIFIED  (PAPI_MH_TYPE_INST|PAPI_MH_TYPE_DATA)
#define PAPI_MH_CACHE_TYPE(a) (a & 0xf)
#define PAPI_MH_TYPE_WT       0x00	   /* write-through cache */
#define PAPI_MH_TYPE_WB       0x10	   /* write-back cache */
#define PAPI_MH_CACHE_WRITE_POLICY(a) (a & 0xf0)
#define PAPI_MH_TYPE_UNKNOWN  0x000
#define PAPI_MH_TYPE_LRU      0x100
#define PAPI_MH_TYPE_PSEUDO_LRU 0x200
#define PAPI_MH_CACHE_REPLACEMENT_POLICY(a) (a & 0xf00)
#define PAPI_MH_TYPE_TLB       0x1000  /* tlb, not memory cache */
#define PAPI_MH_TYPE_PREF      0x2000  /* prefetch buffer */
#define PAPI_MH_MAX_LEVELS    6		   /* # descriptors for each TLB or cache level */
#define PAPI_MAX_MEM_HIERARCHY_LEVELS 	  4
#define PAPIF_DMEM_VMPEAK     1
#define PAPIF_DMEM_VMSIZE     2
#define PAPIF_DMEM_RESIDENT   3
#define PAPIF_DMEM_HIGH_WATER 4
#define PAPIF_DMEM_SHARED     5
#define PAPIF_DMEM_TEXT       6
#define PAPIF_DMEM_LIBRARY    7
#define PAPIF_DMEM_HEAP       8
#define PAPIF_DMEM_LOCKED     9
#define PAPIF_DMEM_STACK      10
#define PAPIF_DMEM_PAGESIZE   11
#define PAPIF_DMEM_PTE        12
#define PAPIF_DMEM_MAXVAL     12
//#define PAPI_MAX_INFO_TERMS  19		   /* should match PAPI_MAX_COUNTER_TERMS defined in papi_internal.h */
//#define PAPI_MAX_INFO_TERMS 12


#include <iostream>
using namespace std;
int main(void) {
//cout << "PAPI_VERSION_NUMBER(maj,min,rev,inc) = " << PAPI_VERSION_NUMBER(maj,min,rev,inc) << endl;
//cout << "PAPI_VERSION_MAJOR(x) = " << PAPI_VERSION_MAJOR(x) << endl;
//cout << "PAPI_VERSION_MINOR(x)		(((x)>>16) = " << PAPI_VERSION_MINOR(x)		(((x)>>16) << endl;
//cout << "PAPI_VERSION_REVISION(x)	(((x)>>8) = " << PAPI_VERSION_REVISION(x)	(((x)>>8) << endl;
//cout << "PAPI_VERSION_INCREMENT(x)((x) = " << PAPI_VERSION_INCREMENT(x)((x) << endl;
cout << "PAPI_VERSION = " << PAPI_VERSION << endl;
cout << "PAPI_VER_CURRENT = " << PAPI_VER_CURRENT << endl;
//cout << "IS_NATIVE( = " << IS_NATIVE( << endl;
//cout << "IS_PRESET( = " << IS_PRESET( << endl;
//cout << "IS_USER_DEFINED( = " << IS_USER_DEFINED( << endl;
cout << "PAPI_OK = " << PAPI_OK << endl;
cout << "PAPI_EINVAL = " << PAPI_EINVAL << endl;
cout << "PAPI_ENOMEM = " << PAPI_ENOMEM << endl;
cout << "PAPI_ESYS = " << PAPI_ESYS << endl;
cout << "PAPI_ESBSTR = " << PAPI_ESBSTR << endl;
cout << "PAPI_ECLOST = " << PAPI_ECLOST << endl;
cout << "PAPI_EBUG = " << PAPI_EBUG << endl;
cout << "PAPI_ENOEVNT = " << PAPI_ENOEVNT << endl;
cout << "PAPI_ECNFLCT = " << PAPI_ECNFLCT << endl;
cout << "PAPI_ENOTRUN = " << PAPI_ENOTRUN << endl;
cout << "PAPI_EISRUN = " << PAPI_EISRUN << endl;
cout << "PAPI_ENOEVST = " << PAPI_ENOEVST << endl;
cout << "PAPI_ENOTPRESET = " << PAPI_ENOTPRESET << endl;
cout << "PAPI_ENOCNTR = " << PAPI_ENOCNTR << endl;
cout << "PAPI_EMISC = " << PAPI_EMISC << endl;
cout << "PAPI_EPERM = " << PAPI_EPERM << endl;
cout << "PAPI_ENOINIT = " << PAPI_ENOINIT << endl;
cout << "PAPI_ENOCMP = " << PAPI_ENOCMP << endl;
cout << "PAPI_ENOSUPP = " << PAPI_ENOSUPP << endl;
cout << "PAPI_ENOIMPL = " << PAPI_ENOIMPL << endl;
cout << "PAPI_EBUF = " << PAPI_EBUF << endl;
cout << "PAPI_EINVAL_DOM = " << PAPI_EINVAL_DOM << endl;
cout << "PAPI_EATTR	= " << PAPI_EATTR << endl;
cout << "PAPI_ECOUNT = " << PAPI_ECOUNT << endl;
cout << "PAPI_ECOMBO = " << PAPI_ECOMBO << endl;
cout << "PAPI_NUM_ERRORS	 = " << PAPI_NUM_ERRORS	 << endl;
cout << "PAPI_LOW_LEVEL_INITED = " << PAPI_LOW_LEVEL_INITED << endl;
cout << "PAPI_HIGH_LEVEL_INITED = " << PAPI_HIGH_LEVEL_INITED << endl;
cout << "PAPI_THREAD_LEVEL_INITED = " << PAPI_THREAD_LEVEL_INITED << endl;
cout << "PAPI_NULL = " << PAPI_NULL << endl;
cout << "PAPI_DOM_USER = " << PAPI_DOM_USER << endl;
cout << "PAPI_DOM_MIN = " << PAPI_DOM_MIN << endl;
cout << "PAPI_DOM_KERNEL	 = " << PAPI_DOM_KERNEL	 << endl;
cout << "PAPI_DOM_OTHER	 = " << PAPI_DOM_OTHER	 << endl;
cout << "PAPI_DOM_SUPERVISOR = " << PAPI_DOM_SUPERVISOR << endl;
cout << "PAPI_DOM_ALL	 = " << PAPI_DOM_ALL	 << endl;
cout << "PAPI_DOM_MAX = " << PAPI_DOM_MAX << endl;
cout << "PAPI_DOM_HWSPEC = " << PAPI_DOM_HWSPEC << endl;
cout << "PAPI_HIGH_LEVEL_TLS = " << PAPI_HIGH_LEVEL_TLS << endl;
cout << "PAPI_TLS_HIGH_LEVEL = " << PAPI_TLS_HIGH_LEVEL << endl;
cout << "PAPI_USR1_LOCK = " << PAPI_USR1_LOCK << endl;
cout << "PAPI_USR2_LOCK = " << PAPI_USR2_LOCK << endl;
cout << "PAPI_NUM_LOCK = " << PAPI_NUM_LOCK << endl;
cout << "PAPI_LOCK_USR1 = " << PAPI_LOCK_USR1 << endl;
cout << "PAPI_LOCK_USR2 = " << PAPI_LOCK_USR2 << endl;
cout << "PAPI_MPX_DEF_DEG = " << PAPI_MPX_DEF_DEG << endl;
cout << "PAPI_VENDOR_UNKNOWN = " << PAPI_VENDOR_UNKNOWN << endl;
cout << "PAPI_VENDOR_INTEL = " << PAPI_VENDOR_INTEL << endl;
cout << "PAPI_VENDOR_AMD = " << PAPI_VENDOR_AMD << endl;
cout << "PAPI_VENDOR_IBM = " << PAPI_VENDOR_IBM << endl;
cout << "PAPI_VENDOR_CRAY = " << PAPI_VENDOR_CRAY << endl;
cout << "PAPI_VENDOR_SUN = " << PAPI_VENDOR_SUN << endl;
cout << "PAPI_VENDOR_FREESCALE = " << PAPI_VENDOR_FREESCALE << endl;
cout << "PAPI_VENDOR_ARM = " << PAPI_VENDOR_ARM << endl;
cout << "PAPI_VENDOR_MIPS = " << PAPI_VENDOR_MIPS << endl;
cout << "PAPI_GRN_THR = " << PAPI_GRN_THR << endl;
cout << "PAPI_GRN_MIN = " << PAPI_GRN_MIN << endl;
cout << "PAPI_GRN_PROC = " << PAPI_GRN_PROC << endl;
cout << "PAPI_GRN_PROCG = " << PAPI_GRN_PROCG << endl;
cout << "PAPI_GRN_SYS = " << PAPI_GRN_SYS << endl;
cout << "PAPI_GRN_SYS_CPU = " << PAPI_GRN_SYS_CPU << endl;
cout << "PAPI_GRN_MAX = " << PAPI_GRN_MAX << endl;
cout << "PAPI_PER_CPU = " << PAPI_PER_CPU << endl;
cout << "PAPI_PER_NODE = " << PAPI_PER_NODE << endl;
cout << "PAPI_SYSTEM	 = " << PAPI_SYSTEM	 << endl;
cout << "PAPI_PER_THR = " << PAPI_PER_THR << endl;
cout << "PAPI_PER_PROC = " << PAPI_PER_PROC << endl;
cout << "PAPI_ONESHOT	 = " << PAPI_ONESHOT	 << endl;
cout << "PAPI_RANDOMIZE	 = " << PAPI_RANDOMIZE	 << endl;
cout << "PAPI_STOPPED = " << PAPI_STOPPED << endl;
cout << "PAPI_RUNNING = " << PAPI_RUNNING << endl;
cout << "PAPI_PAUSED = " << PAPI_PAUSED << endl;
cout << "PAPI_NOT_INIT = " << PAPI_NOT_INIT << endl;
cout << "PAPI_OVERFLOWING = " << PAPI_OVERFLOWING << endl;
cout << "PAPI_PROFILING = " << PAPI_PROFILING << endl;
cout << "PAPI_MULTIPLEXING = " << PAPI_MULTIPLEXING << endl;
cout << "PAPI_ATTACHED	 = " << PAPI_ATTACHED	 << endl;
cout << "PAPI_CPU_ATTACHED = " << PAPI_CPU_ATTACHED << endl;
cout << "PAPI_QUIET = " << PAPI_QUIET << endl;
cout << "PAPI_VERB_ECONT = " << PAPI_VERB_ECONT << endl;
cout << "PAPI_VERB_ESTOP = " << PAPI_VERB_ESTOP << endl;
cout << "PAPI_PROFIL_POSIX = " << PAPI_PROFIL_POSIX << endl;
cout << "PAPI_PROFIL_RANDOM = " << PAPI_PROFIL_RANDOM << endl;
cout << "PAPI_PROFIL_WEIGHTED = " << PAPI_PROFIL_WEIGHTED << endl;
cout << "PAPI_PROFIL_COMPRESS = " << PAPI_PROFIL_COMPRESS << endl;
cout << "PAPI_PROFIL_BUCKET_16 = " << PAPI_PROFIL_BUCKET_16 << endl;
cout << "PAPI_PROFIL_BUCKET_32 = " << PAPI_PROFIL_BUCKET_32 << endl;
cout << "PAPI_PROFIL_BUCKET_64 = " << PAPI_PROFIL_BUCKET_64 << endl;
cout << "PAPI_PROFIL_FORCE_SW = " << PAPI_PROFIL_FORCE_SW << endl;
cout << "PAPI_PROFIL_DATA_EAR = " << PAPI_PROFIL_DATA_EAR << endl;
cout << "PAPI_PROFIL_INST_EAR = " << PAPI_PROFIL_INST_EAR << endl;
cout << "PAPI_PROFIL_BUCKETS = " << PAPI_PROFIL_BUCKETS << endl;
cout << "PAPI_OVERFLOW_FORCE_SW = " << PAPI_OVERFLOW_FORCE_SW << endl;
cout << "PAPI_OVERFLOW_HARDWARE = " << PAPI_OVERFLOW_HARDWARE << endl;
//cout << "PAPI_MULTIPLEX_DEFAULT	0x0	/**< = " << PAPI_MULTIPLEX_DEFAULT	0x0	/**< << endl;
cout << "PAPI_MULTIPLEX_FORCE_SW = " << PAPI_MULTIPLEX_FORCE_SW << endl;
cout << "PAPI_INHERIT_ALL = " << PAPI_INHERIT_ALL << endl;
cout << "PAPI_INHERIT_NONE = " << PAPI_INHERIT_NONE << endl;
//cout << "PAPI_DETACH			1		/**< = " << PAPI_DETACH			1		/**< << endl;
cout << "PAPI_DEBUG = " << PAPI_DEBUG << endl;
cout << "PAPI_MULTIPLEX = " << PAPI_MULTIPLEX << endl;
cout << "PAPI_DEFDOM = " << PAPI_DEFDOM << endl;
cout << "PAPI_DOMAIN = " << PAPI_DOMAIN << endl;
cout << "PAPI_DEFGRN = " << PAPI_DEFGRN << endl;
cout << "PAPI_GRANUL = " << PAPI_GRANUL << endl;
cout << "PAPI_DEF_MPX_NS = " << PAPI_DEF_MPX_NS << endl;
cout << "PAPI_EDGE_DETECT = " << PAPI_EDGE_DETECT << endl;
cout << "PAPI_INVERT = " << PAPI_INVERT << endl;
cout << "PAPI_MAX_MPX_CTRS = " << PAPI_MAX_MPX_CTRS	<< endl;
cout << "PAPI_PROFIL = " << PAPI_PROFIL << endl;
cout << "PAPI_PRELOAD = " << PAPI_PRELOAD << endl;
cout << "PAPI_CLOCKRATE = " << PAPI_CLOCKRATE << endl;
cout << "PAPI_MAX_HWCTRS = " << PAPI_MAX_HWCTRS << endl;
cout << "PAPI_HWINFO = " << PAPI_HWINFO << endl;
cout << "PAPI_EXEINFO = " << PAPI_EXEINFO << endl;
cout << "PAPI_MAX_CPUS = " << PAPI_MAX_CPUS << endl;
cout << "PAPI_ATTACH = " << PAPI_ATTACH << endl;
cout << "PAPI_SHLIBINFO = " << PAPI_SHLIBINFO << endl;
cout << "PAPI_LIB_VERSION = " << PAPI_LIB_VERSION << endl;
cout << "PAPI_COMPONENTINFO = " << PAPI_COMPONENTINFO << endl;
cout << "PAPI_DATA_ADDRESS = " << PAPI_DATA_ADDRESS << endl;
cout << "PAPI_INSTR_ADDRESS = " << PAPI_INSTR_ADDRESS << endl;
//cout << "PAPI_DEF_ITIMER		25		/**< = " << PAPI_DEF_ITIMER		25		/**< << endl;
//cout << "PAPI_DEF_ITIMER_NS	26		/**< = " << PAPI_DEF_ITIMER_NS	26		/**< << endl;
cout << "PAPI_CPU_ATTACH = " << PAPI_CPU_ATTACH  << endl;
cout << "PAPI_INHERIT = " << PAPI_INHERIT	 << endl;
cout << "PAPI_USER_EVENTS_FILE = " << PAPI_USER_EVENTS_FILE << endl;
cout << "PAPI_INIT_SLOTS = " << PAPI_INIT_SLOTS << endl;
cout << "PAPI_MIN_STR_LEN = " << PAPI_MIN_STR_LEN << endl;
cout << "PAPI_MAX_STR_LEN = " << PAPI_MAX_STR_LEN << endl;
cout << "PAPI_2MAX_STR_LEN = " << PAPI_2MAX_STR_LEN << endl;
cout << "PAPI_HUGE_STR_LEN = " << PAPI_HUGE_STR_LEN << endl;
cout << "PAPI_DERIVED = " << PAPI_DERIVED << endl;
/*cout << "PAPI_ENUM_ALL = " << PAPI_ENUM_ALL << endl;
cout << "PAPI_PRESET_BIT_MSC = " << PAPI_PRESET_BIT_MSC << endl;
cout << "PAPI_PRESET_BIT_INS = " << PAPI_PRESET_BIT_INS  << endl;
cout << "PAPI_PRESET_BIT_IDL = " << PAPI_PRESET_BIT_IDL << endl;
cout << "PAPI_PRESET_BIT_BR = " << PAPI_PRESET_BIT_BR		  << endl;
cout << "PAPI_PRESET_BIT_CND = " << PAPI_PRESET_BIT_CND		  << endl;
cout << "PAPI_PRESET_BIT_MEM = " << PAPI_PRESET_BIT_MEM		  << endl;
cout << "PAPI_PRESET_BIT_CACH = " << PAPI_PRESET_BIT_CACH	  << endl;
cout << "PAPI_PRESET_BIT_L1 = " << PAPI_PRESET_BIT_L1		  << endl;
cout << "PAPI_PRESET_BIT_L2 = " << PAPI_PRESET_BIT_L2		  << endl;
cout << "PAPI_PRESET_BIT_L3 = " << PAPI_PRESET_BIT_L3		  << endl;
cout << "PAPI_PRESET_BIT_TLB = " << PAPI_PRESET_BIT_TLB	  << endl;
cout << "PAPI_PRESET_BIT_FP = " << PAPI_PRESET_BIT_FP		  << endl;
//cout << "PAPI_NTV_GROUP_AND_MASK		0x00FF0000	/* = " << PAPI_NTV_GROUP_AND_MASK		0x00FF0000	/* << endl;
//cout << "PAPI_NTV_GROUP_SHIFT		16			/* = " << PAPI_NTV_GROUP_SHIFT		16			/* << endl;
cout << "long_long = " << long_long << endl;
*/
//cout << "u_long_long = " << u_long_long << endl;
cout << "PAPI_MH_TYPE_EMPTY = " << PAPI_MH_TYPE_EMPTY << endl;
cout << "PAPI_MH_TYPE_INST = " << PAPI_MH_TYPE_INST << endl;
cout << "PAPI_MH_TYPE_DATA = " << PAPI_MH_TYPE_DATA << endl;
cout << "PAPI_MH_TYPE_VECTOR = " << PAPI_MH_TYPE_VECTOR << endl;
cout << "PAPI_MH_TYPE_TRACE = " << PAPI_MH_TYPE_TRACE << endl;
cout << "PAPI_MH_TYPE_UNIFIED = " << PAPI_MH_TYPE_UNIFIED << endl;
//cout << "PAPI_MH_CACHE_TYPE(a) = " << PAPI_MH_CACHE_TYPE(a) << endl;
cout << "PAPI_MH_TYPE_WT = " << PAPI_MH_TYPE_WT << endl;
cout << "PAPI_MH_TYPE_WB = " << PAPI_MH_TYPE_WB << endl;
//cout << "PAPI_MH_CACHE_WRITE_POLICY(a) = " << PAPI_MH_CACHE_WRITE_POLICY(a) << endl;
cout << "PAPI_MH_TYPE_UNKNOWN = " << PAPI_MH_TYPE_UNKNOWN << endl;
cout << "PAPI_MH_TYPE_LRU = " << PAPI_MH_TYPE_LRU << endl;
cout << "PAPI_MH_TYPE_PSEUDO_LRU = " << PAPI_MH_TYPE_PSEUDO_LRU << endl;
//cout << "PAPI_MH_CACHE_REPLACEMENT_POLICY(a) = " << PAPI_MH_CACHE_REPLACEMENT_POLICY(a) << endl;
cout << "PAPI_MH_TYPE_TLB = " << PAPI_MH_TYPE_TLB << endl;
cout << "PAPI_MH_TYPE_PREF = " << PAPI_MH_TYPE_PREF << endl;
cout << "PAPI_MH_MAX_LEVELS = " << PAPI_MH_MAX_LEVELS << endl;
cout << "PAPI_MAX_MEM_HIERARCHY_LEVELS = " << PAPI_MAX_MEM_HIERARCHY_LEVELS << endl;
cout << "PAPIF_DMEM_VMPEAK = " << PAPIF_DMEM_VMPEAK << endl;
cout << "PAPIF_DMEM_VMSIZE = " << PAPIF_DMEM_VMSIZE << endl;
cout << "PAPIF_DMEM_RESIDENT = " << PAPIF_DMEM_RESIDENT << endl;
cout << "PAPIF_DMEM_HIGH_WATER = " << PAPIF_DMEM_HIGH_WATER << endl;
cout << "PAPIF_DMEM_SHARED = " << PAPIF_DMEM_SHARED << endl;
cout << "PAPIF_DMEM_TEXT = " << PAPIF_DMEM_TEXT << endl;
cout << "PAPIF_DMEM_LIBRARY = " << PAPIF_DMEM_LIBRARY << endl;
cout << "PAPIF_DMEM_HEAP = " << PAPIF_DMEM_HEAP << endl;
cout << "PAPIF_DMEM_LOCKED = " << PAPIF_DMEM_LOCKED << endl;
cout << "PAPIF_DMEM_STACK = " << PAPIF_DMEM_STACK << endl;
cout << "PAPIF_DMEM_PAGESIZE = " << PAPIF_DMEM_PAGESIZE << endl;
cout << "PAPIF_DMEM_PTE = " << PAPIF_DMEM_PTE << endl;
cout << "PAPIF_DMEM_MAXVAL = " << PAPIF_DMEM_MAXVAL << endl;
//cout << "PAPI_MAX_INFO_TERMS = " << PAPI_MAX_INFO_TERMS << endl;
//cout << "PAPI_MAX_INFO_TERMS = " << PAPI_MAX_INFO_TERMS << endl;
return 0;
}
