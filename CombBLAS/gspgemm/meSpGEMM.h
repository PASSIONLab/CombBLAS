/**
 * @file
 *	meSpGEMM.h
 *
 * @author
 *	Oguz Selvitopi
 *
 * @date
 *	2018.10.XX
 *
 * @brief
 *	Memory efficient spgemm of CombBLAS with GPUs
 *
 * @todo
 *
 * @note
 *
 */

#ifndef __MESPGEMM_H__
#define __MESPGEMM_H__

#include <pthread.h>

#include <cmath>
#include <stack>

#include "gSpGEMM.h"

// #define LOG_GNRL_ME_SPGEMM


typedef enum {LSPG_CPU, LSPG_RMERGE2,
			  LSPG_BHSPARSE, LSPG_NSPARSE, LSPG_HYBRID} lspg_t;


template <typename UDERA, typename UDERB, typename LIC,
		  typename NUO, typename NU1>
struct thd_data_spgemm
{
	lspg_t			  local_spgemm;
	UDERA			**ARecv;
	UDERB			**BRecv;
	int				  phase, iter;	// gpu output
	int				  nstages;
	int				  Aself;
	int				  Bself;
	GSpGEMM<NU1>	 *gsp;
	std::ofstream	 *lfile;
	double			 *t_lspgemm;
	double			 *t_wait;
	// std::vector< combblas::SpTuples<LIC,NUO>  *> *tomerge;
	combblas::SpTuples<LIC,NUO>	**tomerge; // produces for merger
	
	// thread coordination
	pthread_cond_t	*input_ready;
	pthread_cond_t	*input_freed;
	pthread_mutex_t *mutex;
	int				*signal_sent;

	// thread coordination - merger
	pthread_cond_t	*output_ready;
	pthread_mutex_t *mutex_merger;
	int				*stage_idx;
};



template <typename LIC, typename NUO>
struct thd_data_merger
{
	int					  nstages;
	LIC					  m;
	LIC					  n;	
	combblas::SpTuples<LIC,NUO>	**merged;	// output
	std::ofstream		 *lfile;
	double				 *t_multiwaymerge;
	// std::vector< combblas::SpTuples<LIC,NUO>  *> *tomerge;
	combblas::SpTuples<LIC,NUO>  **tomerge;	// consumes from spgemm thread
	
	// thread coordination - spgemm
	pthread_cond_t	*output_ready;
	pthread_mutex_t *mutex_merger;
	int				*stage_idx;
};



template <typename UDERA, typename UDERB,
		  typename LIC, typename NUO, typename SR, typename NU1>
void
run_spgemm (
    thd_data_spgemm<UDERA, UDERB, LIC, NUO, NU1> *tds
			)
{
	#if defined(LOG_GNRL_ME_SPGEMM) || defined(TIMING)
	double t_tmp;
	#endif

	typedef typename UDERA::LocalIT LIA;
	typedef typename UDERB::LocalIT LIB;
	
	GSpGEMM<NU1> *gsp = tds->gsp;
	for (int i = 0; i < tds->nstages; ++i)
	{
		pthread_mutex_lock(tds->mutex);
		#ifdef LOG_GNRL_ME_SPGEMM
		t_tmp = MPI_Wtime();
		#endif
		while (*(tds->signal_sent) == 0)
			pthread_cond_wait(tds->input_ready, tds->mutex);
		*(tds->signal_sent) = 0;
		#ifdef LOG_GNRL_ME_SPGEMM
		*(tds->t_wait) += MPI_Wtime() - t_tmp;
		#endif
		pthread_mutex_unlock(tds->mutex);

		#if defined(LOG_GNRL_ME_SPGEMM) || defined(TIMING)
		t_tmp = MPI_Wtime();
		#endif

		combblas::SpTuples<LIC, NUO> *C_cont = NULL;
		if (tds->local_spgemm == LSPG_CPU)
		{
			C_cont = combblas::LocalHybridSpGEMM<SR, NUO>
				(**(tds->ARecv), **(tds->BRecv),
				 i != tds->Aself, i != tds->Bself);
			// cpu cannot overlap, main thread is assured to be waiting on this
			// condition
			pthread_mutex_lock(tds->mutex);
			pthread_cond_signal(tds->input_freed);
			pthread_mutex_unlock(tds->mutex);
		}
		// else if (tds->local_spgemm == LSPG_RMERGE2)
		// 	C_cont = gsp->template mult<LIC, RMerge>
		// 		(**(tds->ARecv), **(tds->BRecv),
		// 		 i != tds->Aself, i != tds->Bself,
		// 		 tds->iter, tds->phase, i, tds->input_freed,
		// 		 tds->mutex);
		// else if (tds->local_spgemm == LSPG_BHSPARSE)
		// 	C_cont = gsp->template mult<LIC, Bhsparse>
		// 		(**(tds->ARecv), **(tds->BRecv),
		// 		 i != tds->Aself, i != tds->Bself,
		// 		 tds->iter, tds->phase, i, tds->input_freed,
		// 		 tds->mutex);		
		else if (tds->local_spgemm == LSPG_NSPARSE)
			C_cont = gsp->template mult<LIC, NSparse>
				(**(tds->ARecv), **(tds->BRecv),
				 i != tds->Aself, i != tds->Bself,
				 tds->iter, tds->phase, i, tds->input_freed,
				 tds->mutex);		
		
		#ifdef TIMING
		mcl_localspgemmtime += MPI_Wtime() - t_tmp;
		#endif

		#ifdef LOG_GNRL_ME_SPGEMM
		*(tds->t_lspgemm) += MPI_Wtime() - t_tmp;
		#endif

		tds->tomerge[i] = C_cont;
		pthread_mutex_lock(tds->mutex_merger);
		*(tds->stage_idx) += 1;
		pthread_cond_signal(tds->output_ready);
		pthread_mutex_unlock(tds->mutex_merger);
	}
}



template <typename LIC, typename NUO, typename SR>
void
run_merger (thd_data_merger<LIC, NUO> *tdm)
{
	#ifdef TIMING
	double t6=MPI_Wtime();
	#endif
	
	std::stack<combblas::SpTuples<LIC, NUO> *> merge_stack;
	
	for (int i = 0; i < tdm->nstages; ++i)
	{	
		pthread_mutex_lock(tdm->mutex_merger);
		while (i >= *(tdm->stage_idx))
			pthread_cond_wait(tdm->output_ready, tdm->mutex_merger);
		pthread_mutex_unlock(tdm->mutex_merger);
		
		int nmerges = log2((i+1) & -(i+1)) + 1;
		merge_stack.push(tdm->tomerge[i]);
		if (nmerges == 1)
			continue;
		
		std::vector< combblas::SpTuples<LIC, NUO>  *> tmp;
		while (nmerges-- > 0)
		{
			if (!(merge_stack.top()->isZero()))
				tmp.push_back(merge_stack.top());
			merge_stack.pop();
		}

		#ifdef LOG_GNRL_ME_SPGEMM
		double t_tmp = MPI_Wtime();
		#endif
		
		combblas::SpTuples<LIC,NUO> *merged = combblas::MultiwayMerge<SR>(tmp, tdm->m, tdm->n, true);
		
		#ifdef LOG_GNRL_ME_SPGEMM
		*(tdm->t_multiwaymerge) += MPI_Wtime() - t_tmp;
		#endif
		
		merge_stack.push(merged);
	}


	combblas::SpTuples<LIC,NUO> *result = NULL;
	if (merge_stack.size() > 1)
	{
		std::vector< combblas::SpTuples<LIC, NUO>  *> tmp;
		while (!merge_stack.empty())
		{
			if (!(merge_stack.top()->isZero()))
				tmp.push_back(merge_stack.top());
			merge_stack.pop();
		}

		#ifdef LOG_GNRL_ME_SPGEMM
		double t_tmp = MPI_Wtime();
		#endif
		
		result = combblas::MultiwayMerge<SR>(tmp, tdm->m, tdm->n, true);

		#ifdef LOG_GNRL_ME_SPGEMM
		*(tdm->t_multiwaymerge) += MPI_Wtime() - t_tmp;
		#endif
	}
	else
	{
		result = merge_stack.top();
		merge_stack.pop();
	}

	*(tdm->merged) = result;

	#ifdef TIMING
	double t7=MPI_Wtime();
	mcl_multiwaymergetime += (t7-t6);
	#endif
}


/**
 * Broadcasts A multiple times (#phases) in order to save storage in the output
 * Only uses 1/phases of C memory if the threshold/max limits are proper
 */
// Edits for accessing as if unfriended
template <typename SR, typename NUO, typename UDERO,
		  typename IU, typename NU1, typename NU2,
		  typename UDERA, typename UDERB>
combblas::SpParMat<IU,NUO,UDERO>
MemEfficientSpGEMMg (
	combblas::SpParMat<IU,NU1,UDERA> &	A,
	combblas::SpParMat<IU,NU2,UDERB> &	B,
	int									phases,
	NUO									hardThreshold,
	IU									selectNum,
	IU									recoverNum,
	NUO									recoverPct,
	int									kselectVersion,
	int64_t								perProcessMemory,
	lspg_t								local_spgemm,
	int 								nrounds
					 )
{
    typedef typename UDERA::LocalIT LIA;
    typedef typename UDERB::LocalIT LIB;
    typedef typename UDERO::LocalIT LIC;

	int np;
	MPI_Comm_size(MPI_COMM_WORLD, &np);
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);

	GSpGEMM<NU1> gsp(myrank);

	static int iter = -1;
	++iter;
	std::ofstream lfile;
	
	#ifdef LOG_GNRL_ME_SPGEMM
	double t_tmp;
	char logstr[64];
	memset(logstr, '\0', 64);
	sprintf(logstr, "p%04d_%s_log.%04d", np,
			(local_spgemm == LSPG_CPU ? "cpu" :
			 (local_spgemm == LSPG_RMERGE2 ? "gpu_rmerge2" :
			  (local_spgemm == LSPG_BHSPARSE ? "gpu_bhsparse" :
			   (local_spgemm == LSPG_NSPARSE ? "gpu_nsparse" :
		"hybrid")))), myrank);	
	lfile.open(std::string(logstr), std::ofstream::out | std::ofstream::app);
	gsp.lfile_ = &lfile;
	
	lfile << ">iter " << iter << std::endl;
	static double	t_Abcast		= 0.0;
	static double	t_Bbcast		= 0.0;
	static double	t_lspgemm		= 0.0;
	static double	t_multiwaymerge = 0.0;
	static double	t_prunerecsel   = 0.0;	// this might differ from timing of hipmcl
	static double	t_estimate_nnz	= 0.0;
	static double   t_phase         = 0.0;

	static double	t_thd_spgemm_wait = 0.0;
	static double	t_thd_main_wait	  = 0.0;

	static double t_est_comm = 0.0;
	static double t_est_comp = 0.0;
	static double t_est_comp_sampling = 0.0;
	#endif

    if(A.getncol() != B.getnrow())
    {
        std::ostringstream outs;
        outs << "Can not multiply, dimensions does not match"<< std::endl;
        outs << A.getncol() << " != " << B.getnrow() << std::endl;
        combblas::SpParHelper::Print(outs.str());
        MPI_Abort(MPI_COMM_WORLD, DIMMISMATCH);
        return combblas::SpParMat< IU,NUO,UDERO >();
    }
	
    if(phases <1 || phases >= A.getncol())
    {
        combblas::SpParHelper::Print("MemEfficientSpGEMM: The value of phases is "
									 "too small or large. Resetting to 1.\n");
        phases = 1;
    }

	int stages, dummy; 	// last two parameters of ProductGrid are ignored for
						// Synch multiplication
	
    std::shared_ptr<combblas::CommGrid> GridC =
		ProductGrid(A.getcommgrid().get(), B.getcommgrid().get(),
					stages, dummy, dummy);

	std::vector<std::pair<int64_t, double> > stage_stats(stages);

	// estimate the number of phases permitted by memory
	if(perProcessMemory>0) 
    {
        int p;
        MPI_Comm World = GridC->GetWorld();
        MPI_Comm_size(World,&p);
        
        int64_t perNNZMem_in = sizeof(IU)*2 + sizeof(NU1);
        int64_t perNNZMem_out = sizeof(IU)*2 + sizeof(NUO);
        
        // max nnz(A) in a porcess
        int64_t lannz = A.getlocalnnz();
        int64_t gannz;
        MPI_Allreduce(&lannz, &gannz, 1,
					  combblas::MPIType<int64_t>(), MPI_MAX, World);
        int64_t inputMem = gannz * perNNZMem_in * 4; // for four copies (two for
													 // SUMMA)

		#ifdef LOG_GNRL_ME_SPGEMM
		t_tmp = MPI_Wtime();
		#endif
		
        // max nnz(A^2) stored by summa in a porcess
		int64_t asquareNNZ = 0;
		// if (local_spgemm == LSPG_CPU)
		// EstPerProcessNnzSpMV(A, B);
		asquareNNZ = EstPerProcessNnzSUMMA(A, B, nrounds, stage_stats, iter);
		// else
		//	asquareNNZ = EstPerProcessNnzSUMMAg(A, B, gsp, lfile);

		#ifdef LOG_GNRL_ME_SPGEMM
		lfile << "estimated number of nnzs for iter " << iter << ": "
			  << asquareNNZ << std::endl << flush;
		t_estimate_nnz += MPI_Wtime()-t_tmp;
		#endif
		
		// an extra copy in multiway merge and in selection/recovery step
        int64_t asquareMem = asquareNNZ * perNNZMem_out * 2; 
        
        
        // estimate kselect memory
		// average nnz per column in A^2 (it is an overestimate because
		// asquareNNZ is estimated based on unmerged matrices)
        int64_t d = ceil( (asquareNNZ * sqrt(p))/ B.getlocalcols() ); 
        // this is equivalent to (asquareNNZ * p) / B.getcol()
        int64_t k = std::min(int64_t(std::max(selectNum, recoverNum)), d );
        int64_t kselectmem = B.getlocalcols() * k * 8 * 3;
        
        // estimate output memory
        int64_t outputNNZ = (B.getlocalcols() * k)/sqrt(p);
        int64_t outputMem = outputNNZ * perNNZMem_in * 2;
        
        //inputMem + outputMem + asquareMem/phases + kselectmem/phases < memory
        int64_t remainingMem = perProcessMemory*1000000000 -
			inputMem - outputMem;
        if(remainingMem > 0)
        {
            phases = 1 + (asquareMem+kselectmem) / remainingMem;
        }
        
        
        if(myrank==0)
        {
            if(remainingMem < 0)
            {
                std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
					"!!!!!\n Warning: input and output memory requirement is "
					"greater than per-process avaiable memory. Keeping phase "
					"to the value supplied at the command line. The program "
					"may go out of memory and crash! \n !!!!!!!!!!!!!!!!!!!!!"
					"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
            }
#ifdef SHOW_MEMORY_USAGE
            int64_t maxMemory = kselectmem/phases + inputMem + outputMem +
				asquareMem / phases;
            if(maxMemory>1000000000)
				std::cout << "phases: " << phases << ": per process memory: "
						  << perProcessMemory << " GB asquareMem: "
						  << asquareMem/1000000000.00 << " GB" << " inputMem: "
						  << inputMem/1000000000.00 << " GB" << " outputMem: "
						  << outputMem/1000000000.00 << " GB" << " kselectmem: "
						  << kselectmem/1000000000.00 << " GB" << std::endl;
            else
				std::cout << "phases: " << phases << ": per process memory: "
						  << perProcessMemory << " GB asquareMem: "
						  << asquareMem/1000000.00 << " MB" << " inputMem: "
						  << inputMem/1000000.00 << " MB" << " outputMem: "
						  << outputMem/1000000.00 << " MB" << " kselectmem: "
						  << kselectmem/1000000.00 << " MB" << std::endl;
#endif
            
        }
    }


	// hybrid spgemm selection
	int64_t flops = 0;
	double	cf	  = 0.0;
	for (int i = 0; i < stages; ++i)
	{
		flops += stage_stats[i].first;
		cf	  += stage_stats[i].second;
	}
	flops /= stages;
	cf	  /= stages;
	if (local_spgemm == LSPG_HYBRID)
	{
		local_spgemm = LSPG_NSPARSE;
		if (flops / phases < (2 * 1e6))
			local_spgemm = LSPG_CPU;
		else if (cf <= 3)
		{
			// local_spgemm = LSPG_RMERGE2;
			local_spgemm = LSPG_CPU;
		}
		
		#ifdef LOG_GNRL_ME_SPGEMM
		lfile << "local hybrid spgemm " <<
			(local_spgemm == LSPG_CPU ? "cpu" :
			 (local_spgemm == LSPG_RMERGE2 ? "gpu_rmerge2" :
			  (local_spgemm == LSPG_NSPARSE ? "gpu_nsparse" : "N/A")))
			  << std::endl;
		#endif
	}

	LIA C_m = A.seqptr()->getnrow();
    LIB C_n = B.seqptr()->getncol();

	// Each proc splits its own portion into #phases
    std::vector< UDERB > PiecesOfB;
    UDERB CopyB = *(B.seqptr()); // we allow alias matrices as input because of
								 // this local copy

	CopyB.ColSplit(phases, PiecesOfB); // CopyB's memory is destroyed at this
									   // point
    MPI_Barrier(GridC->GetWorld());

    
    LIA ** ARecvSizes = combblas::SpHelper::allocate2D<LIA>
		(UDERA::esscount, stages);
    LIB ** BRecvSizes = combblas::SpHelper::allocate2D<LIB>
		(UDERB::esscount, stages);
    
    static_assert(std::is_same<LIA, LIB>::value, "local index types for both "
				  "input matrices should be the same");
    static_assert(std::is_same<LIA, LIC>::value, "local index types for input "
				  "and output matrices should be the same");
    
    
	combblas::SpParHelper::GetSetSizes( *(A.seqptr()), ARecvSizes,
										(A.getcommgrid())->GetRowWorld());
    
    // Remotely fetched matrices are stored as pointers
    UDERA * ARecv;
    UDERB * BRecv;

	// partial output SpDCSC to concatenate
    std::vector< UDERO > toconcatenate;
    
    int Aself = (A.getcommgrid())->GetRankInProcRow();
    int Bself = (B.getcommgrid())->GetRankInProcCol();

    for(int p = 0; p< phases; ++p)
    {
		#ifdef LOG_GNRL_ME_SPGEMM
		double t_phase_begin = MPI_Wtime();
		#endif
		
        combblas::SpParHelper::GetSetSizes( PiecesOfB[p], BRecvSizes,
											(B.getcommgrid())->GetColWorld());
        // std::vector< combblas::SpTuples<LIC,NUO>  *> tomerge;
		combblas::SpTuples<LIC,NUO> **tomerge = new combblas::SpTuples<LIC,NUO> *[stages];
		combblas::SpTuples<LIC,NUO> *merged_tuples;

		// spgemm thread
		pthread_cond_t	input_ready, input_freed, output_ready;
		pthread_mutex_t mutex, mutex_merger;
		int				signal_sent, stage_idx;
		pthread_cond_init(&input_ready, NULL);
		pthread_cond_init(&input_freed, NULL);
		pthread_mutex_init(&mutex, NULL);
		signal_sent = 0;
		pthread_cond_init(&output_ready, NULL);
		pthread_mutex_init(&mutex_merger, NULL);
		stage_idx	  = 0;

		// spgemm thread data
		thd_data_spgemm<UDERA, UDERB, LIC, NUO, NU1> tds;
		tds.local_spgemm = local_spgemm;
		tds.ARecv		 = &ARecv;
		tds.BRecv		 = &BRecv;
		tds.phase		 = p;
		tds.iter		 = iter;
		tds.nstages		 = stages;
		tds.tomerge		 = tomerge;
		tds.Aself		 = Aself;
		tds.Bself		 = Bself;
		tds.gsp			 = &gsp;
		#ifdef LOG_GNRL_ME_SPGEMM
		tds.lfile		 = &lfile;
		tds.t_lspgemm    = &t_lspgemm;
		tds.t_wait       = &t_thd_spgemm_wait;
		#endif
		tds.input_ready	 = &input_ready;
		tds.input_freed	 = &input_freed;
		tds.mutex		 = &mutex;
		tds.signal_sent	 = &signal_sent;
		tds.mutex_merger = &mutex_merger;
		tds.output_ready = &output_ready;
		tds.stage_idx	 = &stage_idx;

		pthread_t thd_spgemm;
		pthread_create(&thd_spgemm, (pthread_attr_t *) NULL,
					   (void * (*) (void *))
					   run_spgemm<UDERA, UDERB, LIC, NUO, SR, NU1>,
					   &tds);

		// merger thread
		thd_data_merger<LIC, NUO>  tdm;
		tdm.nstages			= stages;
		tdm.m				= C_m;
		tdm.n				= PiecesOfB[p].getncol();
		tdm.tomerge			= tomerge;
		tdm.merged			= &merged_tuples;
		#ifdef LOG_GNRL_ME_SPGEMM
		tdm.lfile			= &lfile;
		tdm.t_multiwaymerge = &t_multiwaymerge;
		#endif
		tdm.output_ready	= &output_ready;
		tdm.mutex_merger	= &mutex_merger;
		tdm.stage_idx		= &stage_idx;
		pthread_t thd_merger;
		pthread_create(&thd_merger, (pthread_attr_t *) NULL,
					   (void * (*) (void *)) run_merger<LIC, NUO, SR>,
					   &tdm);
		
        for(int i = 0; i < stages; ++i)
        {
			// Put essential elem counts of this stage into ess
            std::vector<LIA> ess;
            if(i == Aself)  ARecv = A.seqptr();	// shallow-copy
            else
            {
                ess.resize(UDERA::esscount);
                for(int j=0; j< UDERA::esscount; ++j)
                    ess[j] = ARecvSizes[j][i];		// essentials of the ith
													// matrix in this row
                ARecv = new UDERA();				// first, create the object
            }
            
			#if defined(TIMING) || defined(LOG_GNRL_ME_SPGEMM)
            double t0=MPI_Wtime();
			#endif

			// Broadcasts this stage's A submatrix. Three index
			// arrays and the numerical values
			// then, receive its elements
            combblas::SpParHelper::BCastMatrix(GridC->GetRowWorld(),
											   *ARecv, ess, i);

			#ifdef TIMING
            double t1=MPI_Wtime();
            mcl_Abcasttime += (t1-t0);
			#endif
			
			#ifdef LOG_GNRL_ME_SPGEMM
			t_Abcast += t1 - t0;
			#endif
			
            ess.clear();

			// Repeat for pieces of B submatrix
            if(i == Bself)  BRecv = &(PiecesOfB[p]);
            else
            {
                ess.resize(UDERB::esscount);
                for(int j=0; j< UDERB::esscount; ++j)
                    ess[j] = BRecvSizes[j][i];
                BRecv = new UDERB();
            }
			
			#if defined(TIMING) || defined(LOG_GNRL_ME_SPGEMM)
            double t2=MPI_Wtime();
			#endif

			// then, receive its elements
            combblas::SpParHelper::BCastMatrix(GridC->GetColWorld(),
											   *BRecv, ess, i);

			#if defined(TIMING) || defined(LOG_GNRL_ME_SPGEMM)
            double t3=MPI_Wtime();
            mcl_Bbcasttime += (t3-t2);
			#endif
			
			#ifdef LOG_GNRL_ME_SPGEMM
			t_Bbcast += t3 - t2;
			#endif

			// inputs ready
			pthread_mutex_lock(&mutex);
			signal_sent = 1;
			pthread_cond_signal(&input_ready);
			#ifdef LOG_GNRL_ME_SPGEMM
			t_tmp = MPI_Wtime();
			#endif
			if (i != stages-1)
				pthread_cond_wait(&input_freed, &mutex);
			#ifdef LOG_GNRL_ME_SPGEMM
			t_thd_main_wait += MPI_Wtime() - t_tmp;
			#endif
			pthread_mutex_unlock(&mutex);
			
        }   // all stages executed

		pthread_join(thd_spgemm, NULL);
		pthread_join(thd_merger, NULL);

		#ifdef LOG_GNRL_ME_SPGEMM
		lfile << "iter " << iter << " phase " << p << " merged number of elements "
			  << merged_tuples->getnnz() << "\n";
		t_phase += MPI_Wtime() - t_phase_begin;		
		#endif

		UDERO * OnePieceOfC = new UDERO(*merged_tuples, false);
        delete merged_tuples;
		        
		#ifdef LOG_GNRL_ME_SPGEMM
		double t8 = MPI_Wtime();
		#endif
        
        combblas::SpParMat<IU,NUO,UDERO> OnePieceOfC_mat(OnePieceOfC, GridC);
        MCLPruneRecoverySelect(OnePieceOfC_mat, hardThreshold, selectNum,
							   recoverNum, recoverPct, kselectVersion);

		#ifdef LOG_GNRL_ME_SPGEMM
		t_prunerecsel += MPI_Wtime() - t8;
		#endif

		#ifdef SHOW_MEMORY_USAGE
        int64_t gcnnz_pruned, lcnnz_pruned ;
        lcnnz_pruned = OnePieceOfC_mat.getlocalnnz();
        MPI_Allreduce(&lcnnz_pruned, &gcnnz_pruned, 1,
					  combblas::MPIType<int64_t>(), MPI_MAX, MPI_COMM_WORLD);
        
        
        // TODO: we can remove gcnnz_merged memory here because we don't need to
        // concatenate anymore
        int64_t prune_memory = gcnnz_pruned*2*20;
		//(gannz*2 + phase_nnz + gcnnz_pruned*2) * 20 + kselectmem; // 3 extra
		//copies of OnePieceOfC_mat, we can make it one extra copy!
        //phase_nnz += gcnnz_pruned;
        
        if(myrank==0)
        {
            if(prune_memory>1000000000)
                std::cout << "Prune: " << prune_memory/1000000000.00
						  << "GB " << std::endl ;
            else
                std::cout << "Prune: " << prune_memory/1000000.00
						  << " MB " << std::endl ;
            
        }
		#endif
		
        // Each phase contributes to a column stripe of C, hence we only need to
        // concatenate
        // ABAB: Change this to accept pointers to objects
        toconcatenate.push_back(OnePieceOfC_mat.seq());
    }
    
	// ABAB: Change this to accept a vector of pointers to pointers to DER
	// objects
    UDERO * C = new UDERO(0,C_m, C_n,0);
    C->ColConcatenate(toconcatenate);

    
    combblas::SpHelper::deallocate2D(ARecvSizes, UDERA::esscount);
    combblas::SpHelper::deallocate2D(BRecvSizes, UDERA::esscount);

	#ifdef LOG_GNRL_ME_SPGEMM
	lfile << std::fixed << std::setprecision(4);
	lfile << "time Abcast        " << t_Abcast << std::endl;
	lfile << "time Bbcast        " << t_Bbcast << std::endl;
	lfile << "time spgemm        " << t_lspgemm << std::endl;
	lfile << "time merge         " << t_multiwaymerge << std::endl;
	lfile << "time prune/rec/sel " << t_prunerecsel << std::endl;
	lfile << "time estimate nnz  " << t_estimate_nnz << std::endl;
	lfile << "time phase         " << t_phase << std::endl;
	lfile << "time wait" << std::endl;
	lfile << "  spgemm thread    " << t_thd_spgemm_wait << std::endl;
	lfile << "  main thread      " << t_thd_main_wait << std::endl;
	lfile << "time est comm            " << t_est_comm << std::endl;
	lfile << "time est comp (hash)     " << t_est_comp << std::endl;
	lfile << "time est comp (sampling) " << t_est_comp_sampling << std::endl;
	if (local_spgemm != LSPG_CPU)
	{
		lfile << "gpu time details" << std::endl;
		gsp.report_time(lfile, phases*stages);
	}
	lfile.close();
	#endif

	return combblas::SpParMat<IU,NUO,UDERO> (C, GridC);
}


#endif
