import pyCombBLAS as pcb

DenseCheck = False;
SparseCheck = False;

class PySpParVec:
	def __init__(self, sz):
		self.pySPV = pcb.pySpParVec(sz);

	def __abs__(self, other):	#FIX:  why other?
		ret = PySpParVec(self.getnnz());
		ret.pySPV = self.pySPV.copy();
		ret.pySPV.abs();
		return ret;

	def __add__(self, other):
		ret = PySpParVec(self.getnnz());
		ret.pySPV = self.pySPV.copy();
		ret.pySPV += other.pySPV;
		return ret;

	def __copy__(self):
		return self.pySPV.copy();

	def __delitem__(self, key):
		if isinstance(key,PyDenseParVec):
			if self.len() <> key.len():
				raise IndexError;
			pcb.EWiseMult_inplacefirst(self.pySPV,key.pyDPV,1,0)
		else:
			raise IndexError
		return;

	def __getitem__(self, key):
		if type(key) == int:	# scalar index
			ret = self.pySPV.GetElement(key);
		elif type(key) == type(self):	#HACK:  ideally compare for PySpParVec
			ret = PySpParVec(key.getnnz());
			ret.pySPV = self.pySPV.SubsRef(key.pySPV);
		else:
			raise IndexError
			return;
		return ret;

	def __iadd__(self, other):
		self.pySPV += other.pySPV;
		return ret;

	def __isub__(self, other):
		self.pySPV -= other.pySPV;
		return ret;

	def __len__(self):
		return self.pySPV.len();

	def __mul__(self, other):
		if not isinstance(other,PyDenseParVec):
			raise TypeError;
		ret = PySpParVec(self.len());
		ret.pySPV = pcb.EWiseMult(self.pySPV,other.pyDPV,0,0);
		return ret;

	def __repr__(self):
		self.pySPV.printall()
		return ' ';

	def __setitem__(self, key, value):
		#if type(value) != int:
		#	print "__setitem__ only supports an integer scalar right-hand side"
		#	return
		if type(key) == int:
			self.pySPV.SetElement(key,value);
			#self.pySPV[key] = value;
		elif isinstance(key, PyDenseParVec):
			if self.pySPV.len() != key.pyDPV.len():
				raise KeyError, 'Vector and Key different lengths';
			tmp = PyDenseParVec(self.len(),0);
			tmp = self.dense();
			pcb.EWiseMult_inplacefirst(self.pySPV, key.pyDPV, True, 0);
			tmp.pyDPV += value.pyDPV;
			self.pySPV = tmp.sparse().pySPV;
		elif type(key)==str and key=='existent':
			self.pySPV.Apply_SetTo(value);
		else:
			raise KeyError, "Invalid key in PySpParVec:__setitem__";
		return self;

	def __sub__(self, other):
		ret = PySpParVec(self.getnnz());
		ret.pySPV = self.pySPV.copy();
		ret.pySPV -= other.pySPV;
		return ret;

	def all(self):
		return  self.all()

	def any(self):
		return  self.any()

	def copy(self):
		ret = PySpParVec(self.len());
		ret.pyDPV = self.pyDPV.copy()
		return ret;

	def dense(self):
		ret = PyDenseParVec(self.len(),0)
		ret.pyDPV = self.pySPV.dense();
		return ret; 
	# "get # of existent elements" (name difference from getnnz())
	def getnee(self):
		return self.pySPV.getnnz();
	
	def getnnz(self):
		return self.pySPV.getnnz();
	
	def len(self):
		return self.pySPV.len();

	def printall(self):
		self.pySPV.printall()

	def range(value):
		if type(value) == int:
			self = pySpParVec(0);
			del self.pySPV;
			self.pySPV = pcb.PySpParVec.range(value,0);
		else:
			self = PySpParVec(0);
			del self.pySPV;
			self.pySPV = value.pySPV.copy();
			self.pySPV.setNumToInd();
		return self;

	def sum(self):
		return self.pySPV.Reduce_sum();
	

class PyDenseParVec:

	def __init__(self, sz, init):
		self.pyDPV = pcb.pyDenseParVec(sz,init);

	def __abs__(self, other):
		ret = PyDenseParVec(0, 0);
		ret.pyDPV = self.pyDPV.copy();
		ret.pyDPV.abs();
		return ret;

	def __add__(self, other):
		ret = PyDenseParVec(0, 0);
		ret.pyDPV = self.pyDPV.copy();
		if type(other) == int:
			otherscalar = other;
			other = PyDenseParVec(0,0);
			other.pyDPV = pcb.pyDenseParVec(len(self), otherscalar);
		ret.pyDPV += other.pyDPV;

		return ret;

	def __and__(self, other):
		if type(other) == int:
			otherscalar = other;
			other = PyDenseParVec(0,0);
			other.pyDPV = pcb.pyDenseParVec(len(self), otherscalar);
		#FIX:  check for self/other values other than 0/1
		tmp1 = PyDenseParVec(0,0);
		tmp1.pyDPV = self.pyDPV.copy();
		tmp1.pyDPV += other.pyDPV;
		tmp2 = PySpParVec(0);
		tmp2.pySPV = tmp1.pyDPV.Find_GreaterThan(1);
		tmp2.pySPV.Apply_SetTo(1);
		ret = PyDenseParVec(len(self),0);
		ret[tmp2] = 1;
		return ret;
			

	def __copy__(self):
		return self.pyDPV.copy();

	def __getitem__(self, key):
		if type(key) == int:	# scalar index
			ret = self.pyDPV.GetElement(key);
		elif type(key) == type(self):	#HACK:  ideally compare for PyDenseParVec
			ret = PyDenseParVec(0, 0);
			ret.pyDPV = self.pyDPV.SubsRef(key.pyDPV);
		else:
			print "__getitem__ only supports scalar or PyDenseParVec subscript"
			return;
		return ret;

	def __iadd__(self, other):
                if 'pyDPV' in other.__dict__:
		        self.pyDPV += other.pyDPV;
                else:
                        self.pyDPV += other.pySPV;
                        pass
                        #self.pyDPV.add(other);
		return self;

	def __isub__(self, other):
		self.pyDPV -= other.pyDPV;
		return self;

	def __len__(self):
		return self.pyDPV.len();

	def __mul__(self, other):
		if not isinstance(other,PySpParVec):
			raise TypeError;
		ret = PySpParVec(self.len());
		ret.pySPV = pcb.EWiseMult(other.pySPV,self.pyDPV,0,0);
		return ret;

	def __ne__(self, other):
		if type(other) == int:  # vector <> scalar; expand scalar
			other = PyDenseParVec(len(self),other)
	
		diff = self - other;
		trues = PySpParVec(0);
		trues.pySPV = diff.pyDPV.Find_NotEqual(0);
		ret = PyDenseParVec(len(self),0);
		ret[trues] = 1;
		return ret;
		
	def __repr__(self):
		self.pyDPV.printall()
		return ' ';

	def __setitem__(self, key, value):
		if type(key) == int:	# index is a scalar
			self.pyDPV.SetElement(key,value);
					# index is a sparse vector
		elif isinstance(key, PySpParVec):
			self.pyDPV.ApplyMasked_SetTo(key.pySPV, 0);
			if type(value) == int:	# value is a scalar
				tmp = key.pySPV.copy();
				tmp.Apply_SetTo(value)
				self.pyDPV += tmp;
			else:			# value is a sparse vector
				self.pyDPV += value.pySPV;
			pass
		elif type(key)==type(str) and key=='existent':
			self.pyDPV.ApplyMasked_SetTo(key.pySPV, value);
		else:
			raise KeyError, "Invalid key in PyDenseParVec:__setitem__";
		return self;

	def __sub__(self, other):
		ret = PyDenseParVec(0, 0);
		ret.pyDPV = self.pyDPV.copy();
		ret.pyDPV -= other.pyDPV;
		return ret;

	def any(self):
		tmp = self.pyDPV.Find_NotEqual(0).getnnz()
		return tmp;

	def copy(self):
		ret = PyDenseParVec(0,0);
		ret.pyDPV = self.pyDPV.copy()
		return ret;
		
	def find(self):
		ret = PyDenseParVec(0,0);
		ret.pyDPV = self.pyDPV.FindInds_NotEqual(0);
		return ret;

	def findGt(self, value):
		if type(value) != int:		#HACK, should check length
			raise TypeError, 'value must be scalar'
		ret = PyDenseParVec(0,0);
		ret.pyDPV = self.pyDPV.FindInds_GreaterThan(value);
		return ret;

	# "get # of existent elements" (name difference from getnnz())
	def getnee(self):
		return self.DPV.getnnz();
	
	def getnnz(self):
		return self.DPV.getnnz();
	
	def getnz(self):
		return self.DPV.getnz();
	
	def len(self):
		return self.pyDPV.len();

	def printall(self):
		self.pyDPV.printall()

	def randPerm(self):
		self.pyDPV.RandPerm();
		return self;
	
	def range(self, sz, start=0):
		self.pyDPV = pcb.pyDenseParVec.range(sz, start);
		return self;
	
	def sparse(self):
		ret = PySpParVec(self.len());
		ret.pySPV = self.pyDPV.sparse();
		return ret;

	def sum(self):
		return self.pyDPV.Reduce_sum();
	

if DenseCheck:
	a = PyDenseParVec(12,0);
	b = PyDenseParVec(12,4);
	c = a+b;
	d = b+b;
	#c.printall(), d.printall()
	e = range(12);
	e.printall()
	len(e)
	f = range(5,3);
	g = (d+e)[f];
	h = d+e;
	h[f].printall()
	h[8]

if SparseCheck:
	a = PySpParVec(12);
	a[2] = 2;
	a[4] = 4;
	a[5] = -5;
	a[6] = 6;
	b = PySpParVec(12);
	b[2] = 4;
	b[3] = -9;
	b[5] = -25;
	b[7] = -49;
	c = a+b;
	d = b+b;
	a.printall(), b.printall()
	c.printall(), d.printall()
	e = PySpParVec(3);
	e[0] = 3;
	e[1] = 7;
	e[2] = 10;
	e.printall()
	f = c[e];
	h = c;
	j = h[f]
	j[8]
	j[8] = 777;
	j[8];


#h[f] = 3;
