// This file is for built-in semirings used for whatever purposes.
// For general Python semirings see pyOperations.h/.cpp

enum OmniBusOptions {TEST_X, TEST_Y, SET_X, SET_Y};

// This semiring is used in indexing (SpParMat::operator())
template <class OUT>
struct PCBBoolCopy2ndSRing
{
// boilerplate:
	// Ideally the filter predicates would just be static members of this class,
	// but that would mean they would have to be defined separately for every
	// combination of template parameters. This hack avoids that requirement and
	// keeps this class self-contained.
	static bool filterOmnibusFunc(op::UnaryPredicateObj* pred, const OUT& val, OmniBusOptions whatAmIDoing)
	{
		static op::UnaryPredicateObj* Ypred = NULL;
		
		switch (whatAmIDoing)
		{
			case SET_Y:
				if (Ypred != NULL)
					throw string("Trying to override a predicate!");
				Ypred = pred;
				return true;
			case TEST_Y:
				if (Ypred == NULL)
					return true;
				return (*Ypred)(val);
			default:
				throw string("Invalid omnibus option!");
		}
	}

	//static bool setFilterX(op::UnaryPredicateObj *pred) {}
	static void setFilterY(op::UnaryPredicateObj *pred)
	{
		filterOmnibusFunc(pred, OUT(), SET_Y);
	}
	//static bool testFilterX()
	static bool testFilterY(const OUT& val) { return filterOmnibusFunc(NULL, val, TEST_Y); }

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

// actual logic
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
		if (!testFilterY(arg2))
			returnedSAID(true);
		return arg2;
	}
	static void axpy(bool a, const OUT & x, OUT & y)
	{
		y = multiply(a, x);
	}
};

// This semiring is used in indexing (SpParMat::operator())
template <class OUT>
struct PCBBoolCopy1stSRing
{
	static OUT id() { return OUT(); }
	static bool returnedSAID() { return false; }
	static OUT add(const OUT & arg1, const OUT & arg2)
	{
		throw std::string("Add should not happen!");
		return arg2;
	}
	static const OUT& multiply(const OUT & arg1, bool arg2)
	{
		return arg1;
	}
	static void axpy(const OUT& a, bool x, OUT & y)
	{
		y = multiply(a, x);
	}

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
