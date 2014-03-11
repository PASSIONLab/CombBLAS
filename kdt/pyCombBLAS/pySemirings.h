// This file is for built-in semirings used for whatever purposes.
// For general Python semirings see pyOperations.h/.cpp

enum OmniBusOptions {TEST_X, TEST_Y, SET_X, SET_Y};

template <class OUT, class XT, class YT>
class SRFilterHelper
{
	public:
	
	// Ideally the filter predicates would just be static members of this class,
	// but that would mean they would have to be defined separately for every
	// combination of template parameters. This hack avoids that requirement and
	// keeps this class self-contained.
	// This class gets re-used for all the semirings. The reason there's no problem
	// with using the same datastructure for multiple semirings is that only one semiring
	// operation is supported at a time. If that's no longer true then the exceptions
	// thrown here should bring that to our attention.
	
	static bool filterOmnibusFunc(op::UnaryPredicateObj* pred, const XT* Xval, const YT* Yval, OmniBusOptions whatAmIDoing)
	{
		static op::UnaryPredicateObj* Ypred = NULL;
		static op::UnaryPredicateObj* Xpred = NULL;
		
		switch (whatAmIDoing)
		{
			case SET_X:
				if (pred != NULL && Xpred != NULL)
					throw string("Trying to override a SRFilterHelper predicate X!");
				Xpred = pred;
				return true;
			case SET_Y:
				if (pred != NULL && Ypred != NULL)
					throw string("Trying to override a SRFilterHelper predicate Y!");
				Ypred = pred;
				return true;
			case TEST_X:
				if (Xpred == NULL)
					return true;
				return (*Xpred)(*Xval);
			case TEST_Y:
				if (Ypred == NULL)
					return true;
				return (*Ypred)(*Yval);
			default:
				throw string("Invalid omnibus option!");
		}
	}

	static void setFilterX(op::UnaryPredicateObj *pred) { filterOmnibusFunc(pred, NULL, NULL, SET_X); }
	static void setFilterY(op::UnaryPredicateObj *pred) { filterOmnibusFunc(pred, NULL, NULL, SET_Y); }
	static bool testFilterX(const XT& val) 		 { return filterOmnibusFunc(NULL, &val, NULL, TEST_X); }
	static bool testFilterY(const YT& val) 		 { return filterOmnibusFunc(NULL, NULL, &val, TEST_Y); }
};

// This semiring is used in indexing (SpParMat::operator())
template <class OUT>
struct PCBBoolCopy2ndSRing
{
	static OUT id() { return OUT(); }
	static bool returnedSAID(bool setFlagTo=false)
	{
		static bool flag = false;
		
		bool temp = flag; // save the current flag value to be returned later. Saves an if statement.
		flag = setFlagTo; // set/clear the flag.
		return temp;
	}

	static OUT add(const OUT & arg1, const OUT & arg2)
	{
		throw std::string("Add should not happen!");
		return arg2;
	}
	static const OUT& multiply(bool arg1, const OUT & arg2)
	{
		if (!SRFilterHelper<OUT, bool, OUT>::testFilterY(arg2))
			returnedSAID(true);
		return arg2;
	}
	static void axpy(bool a, const OUT & x, OUT & y)
	{
		y = multiply(a, x);
	}
// MPI boilerplate
	static MPI_Op mpi_op()
	{
		static MPI_Op mpiop;
		static bool exists = false;
		if (exists)
			return mpiop;
		else
		{
			MPI_Op_create(MPI_func, true, &mpiop);
			exists = true;
			return mpiop;
		}
	}

	static void MPI_func(void * invec, void * inoutvec, int * len, MPI_Datatype *datatype)
	{
		if (*len > 0)
		{
			throw string("MPI Add should not happen (BoolCopy2ndSRing)!");
		}
	}
};

// This semiring is used in indexing (SpParMat::operator())
template <class OUT>
struct PCBBoolCopy1stSRing
{
	static OUT id() { return OUT(); }
	static bool returnedSAID(bool setFlagTo=false)
	{
		static bool flag = false;
		
		bool temp = flag; // save the current flag value to be returned later. Saves an if statement.
		flag = setFlagTo; // set/clear the flag.
		return temp;
	}
	static OUT add(const OUT & arg1, const OUT & arg2)
	{
		throw std::string("Add should not happen!");
		return arg2;
	}
	static const OUT& multiply(const OUT & arg1, bool arg2)
	{
		if (!SRFilterHelper<OUT, OUT, bool>::testFilterX(arg1))
			returnedSAID(true);
		return arg1;
	}
	static void axpy(const OUT& a, bool x, OUT & y)
	{
		y = multiply(a, x);
	}

// MPI boilerplate
	static MPI_Op mpi_op()
	{
		static MPI_Op mpiop;
		static bool exists = false;
		if (exists)
			return mpiop;
		else
		{
			MPI_Op_create(MPI_func, true, &mpiop);
			exists = true;
			return mpiop;
		}
	}

	static void MPI_func(void * invec, void * inoutvec, int * len, MPI_Datatype *datatype)
	{
		if (*len > 0)
		{
			throw string("MPI Add should not happen (BoolCopy1stSRing)!");
		}
	}
};


template <class NT1, class NT2, class OUT>
struct PCBSelect2ndSRing
{
	static OUT id() { return OUT(); }
	static bool returnedSAID(bool setFlagTo=false)
	{
		static bool flag = false;
		
		bool temp = flag; // save the current flag value to be returned later. Saves an if statement.
		flag = setFlagTo; // set/clear the flag.
		return temp;
	}
	static OUT add(const OUT & arg1, const OUT & arg2)
	{
		return arg2;
	}
	static const OUT& multiply(const NT1 & arg1, const NT2& arg2)
	{
		if (!SRFilterHelper<OUT, NT1, NT2>::testFilterX(arg1) || !SRFilterHelper<OUT, NT1, NT2>::testFilterY(arg2))
			returnedSAID(true);
		return arg2;
	}
	static void axpy(const OUT& a, const NT2& x, OUT & y)
	{
		y = multiply(a, x);
	}

// MPI boilerplate
	static MPI_Op mpi_op()
	{
		static MPI_Op mpiop;
		static bool exists = false;
		if (exists)
			return mpiop;
		else
		{
			MPI_Op_create(MPI_func, true, &mpiop);
			exists = true;
			return mpiop;
		}
	}

	static void MPI_func(void * invec, void * inoutvec, int * len, MPI_Datatype *datatype)
	{
		// do nothing because the inoutvec already contains the 2nd element.
	}
};

template <class NT1, class NT2, class OUT>
struct PCBPlusTimesSRing
{
	static OUT id() { return OUT(); }
	static bool returnedSAID(bool setFlagTo=false)
	{
		static bool flag = false;
		
		bool temp = flag; // save the current flag value to be returned later. Saves an if statement.
		flag = setFlagTo; // set/clear the flag.
		return temp;
	}
	static OUT add(const OUT & arg1, const OUT & arg2)
	{
		return arg1 + arg2;
	}
	static const OUT multiply(const NT1 & arg1, const NT2& arg2)
	{
		if (!SRFilterHelper<OUT, NT1, NT2>::testFilterX(arg1) || !SRFilterHelper<OUT, NT1, NT2>::testFilterY(arg2))
			returnedSAID(true);

#ifndef _MSC_VER
		OUT ret = arg1 * arg2;
		if (ret == id())
			returnedSAID(true);
		return ret;
#else
		// maybe not the most efficient way, but this will only be used on scalars VC can't figure out anything else
		OUT ret = static_cast<OUT>(static_cast<NT1>(arg1) * static_cast<NT1>(arg2));
		if (ret == id())
			returnedSAID(true);
		return ret;
#endif
	}
	static void axpy(const OUT& a, const NT2& x, OUT & y)
	{
		y += a*x;
	}

// MPI boilerplate
	static MPI_Op mpi_op()
	{
		static MPI_Op mpiop;
		static bool exists = false;
		if (exists)
			return mpiop;
		else
		{
			MPI_Op_create(MPI_func, true, &mpiop);
			exists = true;
			return mpiop;
		}
	}

	static void MPI_func(void * invec, void * inoutvec, int * len, MPI_Datatype *datatype)
	{
		for (int i = 0; i < *len; i++)
		{
			static_cast<OUT*>(inoutvec)[i] = static_cast<OUT*>(invec)[i] + static_cast<OUT*>(inoutvec)[i];
		}
	}
};

