extern double comm_bcast;
extern double comm_reduce;
extern double comp_summa;
extern double comp_reduce;


typedef struct
{
	void * cp;	// bytes: (nzc+1) * sizeof(int32_t)
	void * jc;	// bytes: nzc * sizeof(int32_t)
	void * ir;	// bytes: nnz * sizeof(int32_t)
	void * num;	// bytes: nnz * sizeof(double)
} SpDCCol_Arrays;


typedef struct 
{
	size_t nnz;
	size_t m;
	size_t n;
	size_t nzc;
} SpDCCol_Essentials;	

typedef struct
{
	int GRROWS;
	int GRCOLS;
	int RANKINROW;
	int RANKINCOL;
	int rankinlayer;
	int layer_grid;
} CCGrid;


