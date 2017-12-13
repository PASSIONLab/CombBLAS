#include <mpi.h>
#include <sys/time.h> 
#include <iostream>
#include <functional>
#include <algorithm>
#include <vector>
#include <sstream>
#include "CombBLAS/CombBLAS.h"

using namespace std;
using namespace combblas;

#define myidlen 35
#define myseqlen 150

template <unsigned int IDLEN, unsigned int SEQLEN>
class ShortRead
{
public:
	ShortRead(){ id[0] = '\0'; seq[0] = '\0'; qual[0] = '\0'; };
    ShortRead(const string & s_id, const string & s_seq, const string & s_qual)
    {
        s_id.copy(id, IDLEN);
        s_seq.copy(seq, SEQLEN);
        s_qual.copy(qual, SEQLEN);
        id[IDLEN] = '\0';
        seq[SEQLEN] = '\0';
        qual[SEQLEN] = '\0';

    }
	template <unsigned int NS_IDLEN, unsigned int NS_SEQLEN>
	friend ostream& operator<<( ostream& os, const ShortRead<NS_IDLEN,NS_SEQLEN> & sread);
    
private:
	char id[IDLEN+1];   // (+1)s are for null termination
    char seq[SEQLEN+1];
    char qual[SEQLEN+1];
	
	template <typename IT,unsigned int NS_IDLEN, unsigned int NS_SEQLEN>    // NS: no-shadow
	friend class ShortReadSaveHandler;
};

template <unsigned int IDLEN, unsigned int SEQLEN>
ostream& operator<<(ostream& os, const ShortRead<IDLEN, SEQLEN> & sread )
{
	os << sread.id << " " << sread.seq << " " << sread.qual << endl;
	return os;
};


template <class IT, unsigned int IDLEN, unsigned int SEQLEN>
class ShortReadSaveHandler
{
public:
    ShortReadSaveHandler() {};
    ShortRead<IDLEN,SEQLEN> getNoNum(IT ind) { return ShortRead<IDLEN,SEQLEN>(); }
    
    MPI_Datatype getMPIType()
    {
        return MPIType< ShortRead<IDLEN,SEQLEN> >(); // utilize the MPI type cache
    }
    
    void binaryfill(FILE * rFile, IT & ind, ShortRead<IDLEN,SEQLEN> & val)
    {
        size_t entryLength = fread (&ind,sizeof(ind),1,rFile);      // read the index first
        entryLength += fread (&val,sizeof(ShortRead<IDLEN,SEQLEN>),1,rFile);
        if(entryLength != 2)
            cout << "Not enough bytes read in binaryfill " << endl;
    }
    size_t entrylength() { return sizeof(ShortRead<IDLEN,SEQLEN>); }
    
    template <typename c, typename t>
    ShortRead<IDLEN,SEQLEN> read(std::basic_istream<c,t>& is, IT ind)
    {
        string s_id, s_seq, s_qual, s_null;
        getline (is,s_id);
        getline (is,s_seq);
        getline (is,s_null); // basically the '+' sign
        getline (is,s_qual);
        
        return ShortRead<IDLEN,SEQLEN>(s_id, s_seq, s_qual);
    }
	
    template <typename c, typename t>
    void save(std::basic_ostream<c,t>& os, const ShortRead<IDLEN,SEQLEN> & tw, IT ind)
    {
       // make it compatible with read...
    }
};



int main(int argc, char* argv[])
{
	int nprocs, myrank;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD,&myrank);

	if(argc < 4)
	{
		if(myrank == 0)
		{
			cout << "Usage: ./VectorIOPermute <VectorFile>" << endl;
			cout << "<VectorFile> is a binary file" << endl;
		}
		MPI_Finalize(); 
		return -1;
	}				
	{

        FullyDistSpVec<int64_t, ShortRead<myidlen, myseqlen> > ShortReads;
        ShortReads.ReadDistribute(string(argv[1]), 0, ShortReadSaveHandler<int64_t, myidlen, myseqlen>(), true);	// read it from binary file in parallel
			
		
	}
	MPI_Finalize();
	return 0;
}
