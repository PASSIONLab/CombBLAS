#ifndef PCB_OBJ_H
#define PCB_OBJ_H

#include <iostream>
#include <ctime>

/* Note to users:
 It's ok to change the members of these two structures.
*/
//INTERFACE_INCLUDE_BEGIN
class Obj1 {
public:
////////////////////////////////////////////////////
////////////////////////////////////////////////////
///// USER CHANGEABLE CODE BEGIN

	double weight;
	int category;

	// Note: It's important that this default constructor creates a "zero" element. Some operations
	// (eg. Reduce) need a starting element, and this constructor is used to create one. If the
	// "zero" rule is not followed then you may get different results on different numbers of
	// processors.
	Obj1(): weight(0), category(0) {}
	Obj1(const Obj1& other): weight(other.weight), category(other.category) {}

	char *__repr__() const {
		static char temp[256];
		sprintf(temp,"[ %lf, %d ]", weight,category);
		return &temp[0];
	}
	
	bool __eq__(const Obj1& other) const {
		return weight == other.weight && category == other.category;
	}

	bool __ne__(const Obj1& other) const {
		return !(__eq__(other));
	}

	// For sorting
	bool __lt__(const Obj1& other) const {
		return weight < other.weight;
	}

//INTERFACE_INCLUDE_END
	// The functions below are NOT accessible from Python.

	// This function is used to load this object from a file.
	// 'is' is a regular C++ input stream (like cin), and row and column specify
	// the 0-based row and column of this object in the input file.
	template <typename c, typename t>
	void loadCpp(std::basic_istream<c,t>& is, int64_t row, int64_t column)
	{
		is >> weight;
		is >> category;
		//category = 0;
	}

	// This function is used to save this object to a file.
	template <typename c, typename t>
	void saveCpp(std::basic_ostream<c,t>& os) const
	{
		os << weight << "\t" << category;
	}

	template <typename INDEXTYPE>
	static void binaryfill(FILE * rFile, INDEXTYPE & row, INDEXTYPE & col, Obj1 & val)
	{
		size_t read = 0;
		read += fread(&row, sizeof(INDEXTYPE), 1,rFile);
		read += fread(&col, sizeof(INDEXTYPE), 1,rFile);
		read += fread(&val, sizeof(Obj1), 1,rFile);
		if (read != 3)
			throw string("Not enough bytes read in binaryfill");
	}
	
	template <typename INDEXTYPE>
	static size_t entrylength()
	{
		return 2*sizeof(INDEXTYPE)+sizeof(Obj1);
	}

///// USER CHANGEABLE CODE END
////////////////////////////////////////////////////
////////////////////////////////////////////////////

	bool operator==(const Obj1& other) const {
		return __eq__(other);
	}

	bool operator!=(const Obj1& other) const {
		return __ne__(other);
	}

	// For sorting
	bool operator<(const Obj1& other) const {
		return __lt__(other);
	}
	
	static swig_type_info*& SwigTypeInfo;
	
	/*
	Obj1& operator=(const Obj1& other) {
		if (this != &other)
			memcpy(this, &other, sizeof(Obj1));
	}*/

	// for copying just a nonzero structure
	operator bool() const { return true; }

	// to keep an unused by KDT CombBLAS optimization compiling
	void operator=(int64_t b)
	{
		cout << "obj operator=() has been called! should not happen!" << endl;
	}

//INTERFACE_INCLUDE_BEGIN
	// for filtering matrices.
	//bool hasPassedFilter;
};


class Obj2 {
public:
////////////////////////////////////////////////////
////////////////////////////////////////////////////
///// USER CHANGEABLE CODE BEGIN

	bool follower;		// default constructor sets all to zero
	long latest;		// not assigned if no retweets happened
	short count;		

	// Note: It's important that this default constructor creates a "zero" element. Some operations
	// (eg. Reduce) need a starting element, and this constructor is used to create one. If the
	// "zero" rule is not followed then you may get different results on different numbers of
	// processors.
	Obj2(): follower(0), latest(0), count(0) {}
	Obj2(const Obj2& other): follower(other.follower), latest(other.latest), count(other.count) {}

	char *__repr__() const {
		static char temp[256];
		if (count == 0)
		{
			sprintf(temp,"[ %d, %d ]", follower, count);
		}
		else
		{
#ifndef _MSC_VER
			struct tm timeinfo;
			gmtime_r((time_t*)&latest, &timeinfo);
			
			char s[256];
			sprintf(s, "%d-%d-%d %d:%d:%d", timeinfo.tm_year+1900,
											timeinfo.tm_mon+1,
											timeinfo.tm_mday,
											timeinfo.tm_hour,
											timeinfo.tm_min,
											timeinfo.tm_sec);
			sprintf(temp,"[ %d, %d, %s ]", follower, count, s);
#else
			sprintf(temp,"[Unsup. on Win.]");
#endif
			}
		return &temp[0];
	}
	
	bool __eq__(const Obj2& other) const {
		return follower == other.follower && count == other.count && latest == other.latest;
	}

	bool __ne__(const Obj2& other) const {
		return !(operator==(other));
	}

	// For sorting
	bool __lt__(const Obj2& other) const {
		return count < other.count;
	}

//INTERFACE_INCLUDE_END
	// The functions below are NOT accessible from Python.

	// This function is used to load this object from a file.
	// 'is' is a regular C++ input stream (like cin), and row and column specify
	// the 0-based row and column of this object in the input file.
	template <typename c, typename t>
	void loadCpp(std::basic_istream<c,t>& is, int64_t row, int64_t column)
	{
		is >> follower;
		is >> count;
		if(count > 0)
		{
#ifndef _MSC_VER
			string date;
			string time;
			is >> date;
			is >> time;

			struct tm timeinfo;
						int year, month, day, hour, min, sec;
			sscanf (date.c_str(),"%d-%d-%d",&year, &month, &day);
			sscanf (time.c_str(),"%d:%d:%d",&hour, &min, &sec);

			memset(&timeinfo, 0, sizeof(struct tm));
			timeinfo.tm_year = year - 1900; // year is "years since 1900"
			timeinfo.tm_mon = month - 1 ;   // month is in range 0...11
			timeinfo.tm_mday = day;         // range 1...31
			timeinfo.tm_hour = hour;        // range 0...23
			timeinfo.tm_min = min;          // range 0...59
			timeinfo.tm_sec = sec;          // range 0.
			latest = timegm(&timeinfo);
			if(latest == -1) { cout << "Can not parse time date" << endl; exit(-1);}
#else
			throw string("Loading Obj2 is unsupported on Windows due to lack of timegm().");
#endif
		}
		else
		{
			latest = 0;
		}
	}

	// This function is used to save this object to a file.
	template <typename c, typename t>
	void saveCpp(std::basic_ostream<c,t>& os) const
	{
		os << follower << "\t" << count;
		if (count > 0)
		{
#ifndef _MSC_VER
			struct tm timeinfo;
			gmtime_r((time_t*)&latest, &timeinfo);
			
			char s[256];
			sprintf(s, "%d-%d-%d %d:%d:%d", timeinfo.tm_year+1900,
											timeinfo.tm_mon+1,
											timeinfo.tm_mday,
											timeinfo.tm_hour,
											timeinfo.tm_min,
											timeinfo.tm_sec);
			os << s;
#else
			throw string("Saving Obj2 is unsupported on Windows due to lack of gmtime_r().");
#endif
		}
	}

	struct TwitterInteraction
	{
		int32_t from;
		int32_t to;
		bool follow;
		int16_t retweets;
		time_t twtime;
	};

	// for binary loader
	template <typename INDEXTYPE>
	static void binaryfill(FILE * rFile, INDEXTYPE & row, INDEXTYPE & col, Obj2 & val)
	{
		TwitterInteraction twi;
		size_t entryLength = fread (&twi,sizeof(TwitterInteraction),1,rFile);
		row = twi.from - 1 ;
		col = twi.to - 1;
		//val = TwitterEdge(twi.retweets, twi.follow, twi.twtime); 
		val.follower = twi.follow;
		val.count = twi.retweets;
		val.latest = twi.twtime;
		if(entryLength != 1)
			throw string("Not enough bytes read in binaryfill");
	}
	
	template <typename INDEXTYPE>
	static size_t entrylength()
	{
		return sizeof(TwitterInteraction);
	}

///// USER CHANGEABLE CODE END
////////////////////////////////////////////////////
////////////////////////////////////////////////////

	bool operator==(const Obj2& other) const {
		return __eq__(other);
	}

	bool operator!=(const Obj2& other) const {
		return __ne__(other);
	}

	// For sorting
	bool operator<(const Obj2& other) const {
		return __lt__(other);
	}

	static swig_type_info*& SwigTypeInfo;
	
	/*
	Obj2& operator=(const Obj2& other) {
		if (this != &other)
			memcpy(this, &other, sizeof(Obj2));
	}*/

	// for copying just a nonzero structure
	operator bool() const { return true; }
	
	// to keep an unused by KDT CombBLAS optimization compiling
	void operator=(int64_t b)
	{
		cout << "obj operator=() has been called! should not happen!" << endl;
	}

//INTERFACE_INCLUDE_BEGIN
	// for filtering matrices.
	//bool hasPassedFilter;	
};
//INTERFACE_INCLUDE_END

template <typename c, typename t>
inline std::basic_ostream<c,t>& operator<<(std::basic_ostream<c,t>& lhs, const Obj1& rhs) { return lhs << rhs.__repr__(); }

template <typename c, typename t>
inline std::basic_ostream<c,t>& operator<<(std::basic_ostream<c,t>& lhs, const Obj2& rhs) { return lhs << rhs.__repr__(); }

//template <typename c, typename t>
//inline std::basic_istream<c,t>& operator>>(std::basic_istream<c,t>& lhs, const doubleint& rhs) { return lhs >> rhs.d; }


#ifndef NO_SWIGPYRUN
#define SWIGTYPE_p_Obj1 SWIG_Obj1Info
#define SWIGTYPE_p_Obj2 SWIG_Obj2Info
#endif


// From CombBLAS/promote.h:
DECLARE_PROMOTE(Obj1, Obj1, Obj1)
DECLARE_PROMOTE(Obj2, Obj2, Obj2)

DECLARE_PROMOTE(Obj1, Obj2, Obj2) // for semirings
DECLARE_PROMOTE(Obj2, Obj1, Obj1) // for semirings

DECLARE_PROMOTE(bool, Obj1, Obj1)
DECLARE_PROMOTE(Obj1, bool, Obj1)

// From CombBLAS/MPIType.h
#ifndef PYCOMBBLAS_CPP
/////////////////////////////////////////////////////////////
// Forward declarations
/////////////////////////////////////////////////////////////

// SWIG datatypes, needed so the swig wrapper routines can be used outside of pyCombBLAS_wrap.cpp
extern "C" {
extern swig_type_info *SWIG_Obj1Info;
extern swig_type_info *SWIG_Obj2Info;
}

extern MPI::Datatype Obj1_MPI_datatype;
extern MPI::Datatype Obj2_MPI_datatype;

//extern "C" {
void create_EDGE_and_VERTEX_MPI_Datatypes();
//}

template<> MPI::Datatype MPIType< Obj1 >( void );
template<> MPI::Datatype MPIType< Obj2 >( void );

#else 
/////////////////////////////////////////////////////////////
// Definitions for the above forward declarations
/////////////////////////////////////////////////////////////

// The code that assigns these is at top of the SWIG interface, i.e. pyCombBLAS.i.templ, in the init section.
extern "C" {
swig_type_info *SWIG_Obj1Info = NULL;
swig_type_info *SWIG_Obj2Info = NULL;
}
swig_type_info*& Obj1::SwigTypeInfo = SWIG_Obj1Info;
swig_type_info*& Obj2::SwigTypeInfo = SWIG_Obj2Info;


// definitions
MPI::Datatype Obj1_MPI_datatype;
MPI::Datatype Obj2_MPI_datatype;

// called from init_pyCombBLAS_MPI() in pyCombBLAS.cpp
//extern "C" {
void create_EDGE_and_VERTEX_MPI_Datatypes()
{
	Obj1_MPI_datatype = MPI::CHAR.Create_contiguous(sizeof(Obj1));
	Obj1_MPI_datatype.Commit();

	Obj2_MPI_datatype = MPI::CHAR.Create_contiguous(sizeof(Obj2));
	Obj2_MPI_datatype.Commit();
}
//}

template<> MPI::Datatype MPIType< Obj1 >( void )
{
	return Obj1_MPI_datatype;
}

template<> MPI::Datatype MPIType< Obj2 >( void )
{
	return Obj2_MPI_datatype;
}

#endif




#endif

