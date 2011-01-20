import pyCombBLAS as pcb

DenseCheck = False;
SparseCheck = False;

class PySpParMat:
	def __init__(self):
		self.pySPM = pcb.pySpParMat();

	def copy(self):
		ret = PySpParMat();
		ret.pySPM = self.pySPM.copy();
		return ret;

	def nedge(self):
		return self.pySPM.getnnz();

	def nvert(self):
		return self.pySPM.getnrow();

	@staticmethod
	def load(fname):
		ret = PySpParMat();
		ret.pySPM = pcb.pySpParMat();
		ret.pySPM.load(fname);
		return ret

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
		return '';

	def __setitem__(self, key, value):
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
			self.pySPV.Apply(pcb.set(value));
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
		ret = PySpParVec(len(self));
		ret.pySPV = copy(self.pySPV);
		return ret

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
		return '';

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
		tmp2.pySPV = tmp1.pyDPV.Find(pcb.bind2nd(pcb.greater(),1));
		tmp2.pySPV.Apply(pcb.set(1));
		ret = PyDenseParVec(len(self),0);
		ret[tmp2] = 1;
		return ret;
			
	def __ge__(self, other):
		if type(other) == int:  # vector <> scalar; expand scalar
			other = PyDenseParVec(len(self),other)
		diff = self - other;
		trues = PySpParVec(0);
		trues.pySPV = diff.pyDPV.Find(pcb.bind2nd(pcb.greater_equal(),0));
		ret = PyDenseParVec(len(self),0);
		ret[trues] = 1;
		return ret;
		
	def __getitem__(self, key):
		if type(key) == int:	# scalar index
			ret = self.pyDPV.GetElement(key);
		elif isinstance(key,PyDenseParVec):
			ret = PyDenseParVec(0, 0);
			tmp1 = len((key<0).findInds())==0;
			tmp2 = len((key>1).findInds())==0;
			keybool = tmp1 & tmp2 & (len(self)==len(key));
			if not keybool:
				ret.pyDPV = self.pyDPV.SubsRef(key.pyDPV);
			else:
				raise NotImplementedError, "PyDenseParVe logical indexing on right-hand side not implemented yet"
		else:
			raise KeyError, "PyDenseParVec:__getitem__ only supports scalar or PyDenseParVec subscript"
			return;
		return ret;

	def __gt__(self, other):
		if type(other) == int:  # vector <> scalar; expand scalar
			other = PyDenseParVec(len(self),other)
		diff = self - other;
		trues = PySpParVec(0);
		trues.pySPV = diff.pyDPV.Find(pcb.bind2nd(pcb.greater(),0));
		ret = PyDenseParVec(len(self),0);
		ret[trues] = 1;
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

	def __le__(self, other):
		if type(other) == int:  # vector <> scalar; expand scalar
			other = PyDenseParVec(len(self),other)
		diff = self - other;
		trues = PySpParVec(0);
		trues.pySPV = diff.pyDPV.Find(pcb.bind2nd(pcb.less_equal(),0));
		ret = PyDenseParVec(len(self),0);
		ret[trues] = 1;
		return ret;
		
	def __len__(self):
		return self.pyDPV.len();

	def __lt__(self, other):
		if type(other) == int:  # vector <> scalar; expand scalar
			other = PyDenseParVec(len(self),other)
		diff = self - other;
		trues = PySpParVec(0);
		trues.pySPV = diff.pyDPV.Find(pcb.bind2nd(pcb.less(),0));
		ret = PyDenseParVec(len(self),0);
		ret[trues] = 1;
		return ret;
		
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
		trues.pySPV = diff.pyDPV.Find(pcb.bind2nd(pcb.not_equal_to(),0));
		ret = PyDenseParVec(len(self),0);
		ret[trues] = 1;
		return ret;

	def __repr__(self):
		self.pyDPV.printall()
		return '';

	def __setitem__(self, key, value):
		if type(key) == int:	# index is a scalar
			self.pyDPV.SetElement(key,value);
					# index is a sparse vector
		elif isinstance(key, PySpParVec):
			self.pyDPV.ApplyMasked(pcb.set(0), key.pySPV);
			if type(value) == int:	# value is a scalar
				tmp = key.pySPV.copy();
				tmp.Apply(pcb.set(value));
				self.pyDPV += tmp;
			else:			# value is a sparse vector
				self.pyDPV += value.pySPV;
		elif isinstance(key, PyDenseParVec):
			tmp1 = len((key<0).findInds())==0;
			tmp2 = len((key>1).findInds())==0;
			keybool = tmp1 & tmp2 & (len(self)==len(key));
			if type(value) == int:
				self.pyDPV.ApplyMasked(pcb.set(0), key.pyDPV.sparse());
				tmp = key.pyDPV.sparse();
				tmp.Apply(pcb.set(value));
				self.pyDPV += tmp;
			else:			# value is a sparse vector
				if not keybool:
					self.pyDPV += value.pyDPV.sparse();
				else:		#key is boolean; 
					#restrict updates from values to those indicated by key
					value[key.logical_not()] = 0;
					self[key] = 0;
					self += value;
		else:
			raise KeyError, "PyDenseParVec indexing on the left-hand side only accepts integer, PyDenseParVec, or PySpParVec keys";
		return self;

	def __sub__(self, other):
		ret = PyDenseParVec(0, 0);
		ret.pyDPV = self.pyDPV.copy();
		if type(other) == int:
			otherscalar = other;
			other = PyDenseParVec(0,0);
			other.pyDPV = pcb.pyDenseParVec(len(self), otherscalar);
		ret.pyDPV -= other.pyDPV;
		return ret;

	def any(self):
		tmp = self.pyDPV.Find(pcb.bind2nd(pcb.not_equal_to(),0))
		if tmp.getnnz() > 0:
			return True;
		else:
			return False;

	def copy(self):
		ret = PyDenseParVec(len(self),0);
		ret.pyDPV = copy(self.pyDPV);
		return ret;
		
	def find(self):
		ret = PySpParVec(0);
		ret.pySPV = self.pyDPV.Find(pcb.bind2nd(pcb.not_equal_to(),0));
		return ret;

	def findInds(self):
		ret = PyDenseParVec(0,0);
		ret.pyDPV = self.pyDPV.FindInds(pcb.bind2nd(pcb.not_equal_to(),0));
		return ret;

	def findGt(self, value):
		if type(value) != int:		#HACK, should check length
			raise TypeError, 'value must be scalar'
		ret = PyDenseParVec(0,0);
		ret.pyDPV = self.pyDPV.FindInds(pcb.bind2nd(pcb.greater(),value));
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

	def logical_not(self):
		ret = self.copy();
		ret.pyDPV.Apply(pcb.logical_not());
		return ret;
		
	def printall(self):
		self.pyDPV.printall()
		return '';

	def randPerm(self):
		self.pyDPV.RandPerm();
		return self;
	
	@staticmethod
	def range(arg1, *args):
                if len(args) == 0:
                        start = 0;
                        stop = arg1;
                elif len(args) == 1:
                        start = arg1;
                        stop = args[0];
                else:
                        raise NotImplementedError, "No 3-argument range()"
		ret = PyDenseParVec(0,0);
		ret.pyDPV = pcb.pyDenseParVec.range(stop-start, start);
		return ret;
	
	def sparse(self):
		ret = PySpParVec(self.len());
		ret.pySPV = self.pyDPV.sparse();
		return ret;

	def sum(self):
		return self.pyDPV.Reduce_sum();
	
def toPyDenseParVec(pyDPV):
	ret = PyDenseParVec(0,0);
	ret.pyDPV = pyDPV;
	return ret;

def ones(sz):
	ret = PyDenseParVec(sz,1);
	return ret;

def zeros(sz):
	ret = PyDenseParVec(sz,0);
	return ret;

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
