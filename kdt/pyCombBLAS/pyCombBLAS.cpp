#include "pyCombBLAS.h"

void testFunc(double (*f)(double))
{
	cout << "f(10) = " << f(10) << endl;
}


////////////////// OPERATORS

pySpParVec EWiseMult(const pySpParVec& a, const pySpParVec& b, bool exclude)
{
	pySpParVec ret;
	//ret->v = ::EWiseMult(a.v, b.v, exclude);
	cout << "EWiseMult(sparse, sparse) not implemented!" << endl;
	return ret;
}

pySpParVec EWiseMult(const pySpParVec& a, const pyDenseParVec& b, bool exclude, double zero)
{
	/*
	pySpParVec ret = new pySpParVec();
	FullyDistSpVec<pySpParVec::INDEXTYPE, doubleint> result = EWiseMult(a.v, b.v, exclude, doubleint(zero));
	ret->v.stealFrom(result);
	return ret;
	*/
	return pySpParVec(EWiseMult(a.v, b.v, exclude, doubleint(zero)));
}

void EWiseMult_inplacefirst(pySpParVec& a, const pyDenseParVec& b, bool exclude, double zero)
{
	a.v = EWiseMult(a.v, b.v, exclude, doubleint(zero));
}

EWiseArg EWise_Index()
{
	EWiseArg ret;
	ret.type = EWiseArg::GLOBAL_INDEX;
	return ret;
}

EWiseArg EWise_OnlyNZ(pySpParVec* v)
{
	EWiseArg ret;
	ret.type = EWiseArg::SPARSE_NZ;
	ret.sptr = v;
	return ret;
}

EWiseArg EWise_OnlyNZ(pyDenseParVec* v) // shouldn't be used, but here for completeness
{
	EWiseArg ret;
	ret.type = EWiseArg::DENSE;
	ret.dptr = v;
	return ret;
}

// This function performs elementwise operations on an arbitrary number of vectors. It handles vectors
// using iterators defined in CombBLAS/VecIterator.h. These handle both sparse and dense vectors, allowing
// EWise() to mix both. The vectors have to be the same length or else their elements might not be on
// the same physical node.
// The operation itself is defined by the Python function pyfunc. Element values are passed in to
// pyfunc, and any changes pyfunc does to them is reflected back to the vectors.
void EWise(PyObject *pyfunc, int argc, EWiseArgDescriptor* argv, PyObject *argList)
{
	if (argc == 0)
		return;
	
	// Find all the iterators
	int nnz_iters = 0;
	int total_iters = 0;
	for (int i = 0; i < argc; i++)
	{
		if (argv[i].type == EWiseArgDescriptor::ITERATOR)
		{
			total_iters++;
			if (argv[i].onlyNZ)
				nnz_iters++;
		}
	}
	
	if (total_iters == 0)
		return; // nothing to do
	
	VectorLocalIterator<int64_t, doubleint>** iters = new VectorLocalIterator<int64_t, doubleint>*[total_iters];
	
	// Fill in the iterator array
	int nz_ptr = 0;
	int other_ptr = nnz_iters;
	for (int i = 0; i < argc; i++)
	{
		if (argv[i].type == EWiseArgDescriptor::ITERATOR)
		{
			if (argv[i].onlyNZ)
			{
				iters[nz_ptr] = argv[i].iter;
				nz_ptr++;
			}
			else
			{
				iters[other_ptr] = argv[i].iter;
				other_ptr++;
			}
		}
	}
	if (nnz_iters == 0)
		nnz_iters = 1; // Just use the first iterator's nonzero values
		
	PyObject* arglistlist = Py_BuildValue("(O)", argList);  // Build argument list
	
	bool hasNext = true;
	bool continue_;
	int64_t index = iters[0]->GetLocIndex();
	int64_t GlobalIndexStart = iters[0]->LocalToGlobal(0);
	while (hasNext && index != -1)
	{
		continue_ = false;
		
		// parse and advance the individual iterators
		for (int vec_i = 0; vec_i < total_iters; vec_i++)
		{
			iters[vec_i]->NextTo(index);
			
			if (vec_i < nnz_iters)
			{
				int64_t thisIndex = iters[vec_i]->GetLocIndex();
				
				if (thisIndex == -1) // we're done
				{
					hasNext = false;
					continue_ = true;
					break;
				}
				
				if (index < thisIndex)
				{
					index = thisIndex;
					continue_ = true;
					break;
				}
			}
		}
		if (continue_)
			continue;
				
		// Now that we've found a position with non-nulls, assemble the arguments
		for (int i = 0; i < argc; i++)
		{
			switch (argv[i].type)
			{
				case EWiseArgDescriptor::ITERATOR:
					{
						int64_t idx = argv[i].iter->GetLocIndex();
						if (idx == -1 || idx > index)
						{
							// This vector doesn't have an element in for this index, so pass None instead
							PyList_SetItem(argList, i, Py_None);
						}
						else
						{
							doubleint& v = argv[i].iter->GetValue();
							PyObject* value = Py_BuildValue("d", v.d);
							PyList_SetItem(argList, i, value);
						}
					}
					break;
				case EWiseArgDescriptor::GLOBAL_INDEX:
					{
						PyObject* value = Py_BuildValue("i", (GlobalIndexStart + index));
						PyList_SetItem(argList, i, value);
					}
					break;
				case EWiseArgDescriptor::PYTHON_OBJ:
					// it's already there
					break;
			}
		}
		
		// call the visitor
		PyEval_CallObject(pyfunc, arglistlist);                 // Call Python
	
		// update the vectors to reflect changes by the visitor
		for (int i = 0; i < argc; i++)
		{
			switch (argv[i].type)
			{
				case EWiseArgDescriptor::ITERATOR:
					{
						PyObject* value = PyList_GetItem(argList, i);
						if (value == Py_None)
						{
							// see if there used to be a value
							if (argv[i].iter->GetLocIndex() == index)
							{
								//cout << "Deleting arg_" << i << "[" << index << "]" << endl;
								argv[i].iter->Del();
							}
							else
							{
								//cout << "Found None, but not deleting arg_" << i << "[" << index << "]" << endl;
							}
						}
						else if (PyFloat_Check(value) || PyInt_Check(value))
						{
							doubleint val;
							if (PyFloat_Check(value))
								val.d = PyFloat_AsDouble(value);
							else
								val.d = PyInt_AsLong(value);

							argv[i].iter->Set(index, val);
							//cout << "Setting arg_" << i << "[" << index << "] = " << val.d << endl;
						}
						else
						{
							cout << "Ignoring arg_" << i << "[" << index << "]. Got something unknown." << endl;
						}
					}
					break;
				case EWiseArgDescriptor::GLOBAL_INDEX:
				case EWiseArgDescriptor::PYTHON_OBJ:
					break;
			}
		}

		// advance
		if (iters[0]->GetLocIndex() == index) // we might have already advanced by deleting an element
			iters[0]->Next();
		index = iters[0]->GetLocIndex();
	}
	
	Py_DECREF(arglistlist);                                 // Trash arglist
	delete [] iters;
}

void Graph500VectorOps(pySpParVec& fringe_v, pyDenseParVec& parents_v)
{
	SparseVectorLocalIterator<int64_t, doubleint> fringe(fringe_v.v);
	DenseVectorLocalIterator<int64_t, doubleint> parents(parents_v.v);

	pySpParVec newfringe_v(fringe_v.__len__());	
	SparseVectorLocalIterator<int64_t, doubleint> newfringe(newfringe_v.v);

	int64_t index;
	int64_t globalIndexStart = fringe.LocalToGlobal(0);
	while ((index = fringe.GetLocIndex()) >= 0)
	{
		parents.NextTo(index);
		doubleint& f = fringe.GetValue();
		doubleint& p = parents.GetValue();
		
		if (p == -1)
		{
			// discovered new vertex, set its parent
			parents.Set(index, f);
			//newfringe.Set(index, doubleint(globalIndexStart+index));
			newfringe.Append(index, doubleint(globalIndexStart+index));
			//fringe.Set(index, doubleint(globalIndexStart+index));
			//fringe.Next();
		}
		else
		{
			// this has already been discovered
			//fringe.Del();
		}
		fringe.Next();
	}
	fringe_v.v.stealFrom(newfringe_v.v);
	//fringe_v.v = newfringe_v.v;
}

////////////////////////// INITALIZATION/FINALIZE

bool has_MPI_Init_been_called = false;

void init_pyCombBLAS_MPI()
{
	if (!has_MPI_Init_been_called)
	{
		//cout << "calling MPI::Init" << endl;
		MPI::Init();
		
		MPI::Datatype type[1] = {MPI::DOUBLE};
		int blocklen[1] = {1};
		MPI::Aint disp[1];
		
		doubleint data;
		disp[0] = (MPI::Get_address(&data.d) - MPI::Get_address(&data));
	
		doubleint_MPI_datatype = MPI::Datatype::Create_struct(1,blocklen,disp,type);
		doubleint_MPI_datatype.Commit();
		has_MPI_Init_been_called = true;
	}
}

void finalize()
{
	//cout << "calling MPI::Finalize" << endl;
	MPI::Finalize();
}

bool root()
{
	return MPI::COMM_WORLD.Get_rank() == 0;
}

int _nprocs()
{
	return MPI::COMM_WORLD.Get_size();
}

MPI::Datatype doubleint_MPI_datatype;

template<> MPI::Datatype MPIType< doubleint >( void )
{
	//cout << "returning doubleint MPIType" << endl;
	return doubleint_MPI_datatype;
}; 
