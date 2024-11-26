#include <sys/time.h>
#include <iostream>
#include <functional>
#include <algorithm>
#include <vector>
#include <sstream>
#include <cstdlib>
#include <cmath>
#include "CombBLAS/CombBLAS.h"
#include "CombBLAS/CommGrid3D.h"
#include "CombBLAS/ParFriends.h"
#include "../CC.h"
#include "../WriteMCLClusters.h"
#include "IncClust.h"

using namespace std;
using namespace combblas;


#ifdef _OPENMP
int cblas_splits = omp_get_max_threads();
#else
int cblas_splits = 1;
#endif

typedef int64_t IT;
typedef double NT;
typedef int T;
typedef SpDCCols < int64_t, double > DER;
typedef PlusTimesSRing<double, double> PTFF;
typedef PlusTimesSRing<int64_t, int64_t> PTII;
typedef PlusTimesSRing<bool, double> PTBOOLNT;
typedef PlusTimesSRing<double, bool> PTNTBOOL;
typedef std::array<char, MAXVERTNAME> LBL;

//template<typename T>
static std::tuple<int, int, double, double> getStats(std::vector<int> &arr, bool print = false){
    double mu = 0;
    double sig2 = 0;
    int max_val = arr[0];
    int min_val = arr[0];
#pragma omp parallel for default(shared) reduction(+: mu) reduction(max:max_val) reduction(min:min_val)
    for(int i = 0; i < arr.size(); i++){
        mu += (double)arr[i];
        if(arr[i] > max_val) max_val = arr[i];
        if(arr[i] < min_val) min_val = arr[i];
    }
    mu = mu / arr.size();
#pragma omp parallel for default(shared) reduction(+: sig2)
    for(int i = 0; i < arr.size(); i++) sig2 += (mu-(double)arr[i])*(mu-(double)arr[i]);
    sig2 = sig2 / arr.size();
    if(print){
        std::cout << "\t[getStats] Max:" << max_val << std::endl;
        std::cout << "\t[getStats] Min:" << min_val << std::endl;
        std::cout << "\t[getStats] Mean:" << mu << std::endl;
        std::cout << "\t[getStats] Std-Dev:" << sqrt(sig2) << std::endl;
    }
    return std::make_tuple(max_val, min_val, mu, sqrt(sig2));
}

void convNumToLbl(FullyDistVec<IT, IT>& distNum, FullyDistVec<IT, LBL>& distLbl){
    const std::vector<IT> locNum = distNum.GetLocVec();

    for(int i = 0; i < locNum.size(); i++){
        std::string labelStr = std::to_string(locNum[i]); 
        LBL labelArr{};
        for ( IT j = 0; (j < labelStr.length()) && (j < MAXVERTNAME); j++){
            labelArr[j] = labelStr[j]; 
        }
        distLbl.SetLocalElement(i, labelArr);
    }
}

void convLblToNum(FullyDistVec<IT, LBL>& distLbl, FullyDistVec<IT, IT>& distNum){
    const std::vector<LBL> locLbl = distLbl.GetLocVec();

    for(int i = 0; i < locLbl.size(); i++){
        std::string labelStr(locLbl[i].data());
        IT labelNum = atoi(labelStr.c_str());
        distNum.SetLocalElement(i, labelNum);
    }
}

int calcExtraMemoryRequirement(SpParMat<IT, NT, DER>&A){
    int nprocs, myrank, nthreads = 1;
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    IT nnz = A.getnnz();
    IT n = A.getnrow();
    IT matrixExtraMemory = nnz * (sizeof(IT)*2+sizeof(NT)) * 6; // 6 equivalent copies at various places
    IT vectorExtraMemory = n * sizeof(IT) * 13; // 13 equivalent copies at various places
    IT extraMemory = matrixExtraMemory + vectorExtraMemory;
    double perProcExtraMem = double(matrixExtraMemory + vectorExtraMemory) / (double)nprocs;
    double perProcExtraMemGB = double(perProcExtraMem) / (1024 * 1024 * 1024);
    
    return ceil( perProcExtraMemGB ); // Per process extra memory in GB
}

void reversePermutation(SpParMat<IT, NT, DER>&M, FullyDistVec<IT, LBL>& lbl, FullyDistVec<IT, IT>& permMap){
    shared_ptr<CommGrid> fullWorld;
    fullWorld.reset( new CommGrid(MPI_COMM_WORLD, 0, 0) );
    IT totalLength = lbl.TotalLength();

    FullyDistVec<IT, IT> num(fullWorld, totalLength, 0);
    convLblToNum(lbl, num);

    // Reorder the vertex order of whatever will be used as M11 in next iteration
    FullyDistVec<IT,IT> iota(fullWorld);
    iota.iota(totalLength, 0);
    SpParMat<IT, NT, DER> P = SpParMat<IT,NT,DER>(totalLength, totalLength, iota, iota, 1.0, false); // Identity
    //SpParMat<IT, IT, DER> P = SpParMat<IT,IT,DER>(totalLength, totalLength, iota, iota, 1, false); // Identity
    (P)(permMap,iota,true); // Row permute matrix that caused vertex order to be shuffled
    SpParMat<IT, NT, DER> Q(P);
    //SpParMat<IT, IT, DER> Q(P);
    Q.Transpose(); // Column permute matrix because the permutation was symmetric
    
    FullyDistSpVec<IT, IT> iotaSp(iota); 
    FullyDistSpVec<IT, IT> revPermMapSp = SpMV<PTFF>(Q, iotaSp, false);
    //FullyDistSpVec<IT, IT> revPermMapSp = SpMV<PTII>(Q, iotaSp, false);
    FullyDistVec<IT, IT> revPermMap(revPermMapSp);

    M(revPermMap, revPermMap, true);

    FullyDistSpVec<IT, IT> numSp(num); 
    FullyDistSpVec<IT, IT> origNumSp = SpMV<PTFF>(Q, numSp, false);
    //FullyDistSpVec<IT, IT> origNumSp = SpMV<PTII>(Q, numSp, false);
    num = FullyDistVec<IT, IT>(origNumSp);
    convNumToLbl(num, lbl);
}

int main(int argc, char* argv[])
{
    int nprocs, myrank, nthreads = 1;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
#ifdef THREADED
#pragma omp parallel
    {
        nthreads = omp_get_num_threads();
    }
#endif
    if(myrank == 0)
    {
        cout << "Process Grid (p x p x t): " << sqrt(nprocs) << " x " << sqrt(nprocs) << " x " << nthreads << endl;
        cout << "argc " << argc << endl;
    }
    if(argc < 3)
    {
        if(myrank == 0)
        {
            cout << "Usage: ./testideas -M <MATRIX_FILENAME>\n";
        }
    }
    else{
        string Mname = "";
        int base = 1;
        bool isMatrixMarket = true;

        for (int i = 1; i < argc; i++)
        {
            if (strcmp(argv[i],"-M")==0)
            {
                Mname = string(argv[i+1]);
                if(myrank == 0) printf("Matrix filename: %s\n", Mname.c_str());
            }
        }

        shared_ptr<CommGrid> fullWorld;
        fullWorld.reset( new CommGrid(MPI_COMM_WORLD, 0, 0) );
        
        double t0, t1, t2, t3, t4, t5;

        SpParMat<IT, NT, DER> M(fullWorld);

        M.ParallelReadMM(Mname, base, maximum<NT>());
        M.PrintInfo();

        float loadImbalance;
        loadImbalance = M.LoadImbalance();
        if(myrank == 0) fprintf(stderr, "Load Imbalance before random permutation: %f\n", loadImbalance);

        std::vector<int> recvData(nprocs);
        int nnz = M.seqptr()->getnnz();
        MPI_Allgather(&nnz, 1, MPI_INT, recvData.data(), 1, MPI_INT, MPI_COMM_WORLD);

		std::tuple<int, int, double, double>stats = getStats(recvData);
	    if (myrank == 0){
            fprintf(stderr, "max %d\n", std::get<0>(stats));
            fprintf(stderr, "min %d\n", std::get<1>(stats));
            fprintf(stderr, "mean %lf\n", std::get<2>(stats));
            fprintf(stderr, "std-dev %lf\n", std::get<3>(stats));
        }
        
        //MPI_Barrier(MPI_COMM_WORLD);
        //for(int i = 0; i < 32; i++){
            //for (int j = 0; j < 32; j++){
                //int idx = i * 32 + j;
                //if (myrank == 0) printf("| %0.1lf ", double(recvData[idx])/std::get<1>(stats));
            //}
            //if (myrank == 0) printf("|\n");
        //}
        //if (myrank == 0) printf("---\n");
        //MPI_Barrier(MPI_COMM_WORLD);
        if (myrank == 0) fprintf(stderr, "---\n");

        /*
         * SUMMA on the original input matrix
         * */
        SpParMat<IT, NT, DER> X(fullWorld);
        IT XNNZ;
        {
            SpParMat<IT, NT, DER> A(M);
            SpParMat<IT, NT, DER> B(M);

            t0 = MPI_Wtime();
            X = Mult_AnXBn_Synch<PTFF, NT, DER>(A, B);
            t1 = MPI_Wtime();
            if (myrank == 0) fprintf(stderr, "Multiplication time for un-altered Minc: %lf\n",t1-t0 );
            XNNZ = X.getnnz();
            if (myrank == 0) fprintf(stderr, "nnz(X): %lld\n", XNNZ );
            X.ParallelWriteMM("randmat-X.mtx", 1);
        }
        if (myrank == 0) fprintf(stderr, "---\n");
        
        FullyDistVec<IT, IT> p( M.getcommgrid());
        /*
         * Random permutation
         * */
        //p.iota(M.getnrow(), 0);
        //p.RandPerm();
        ////(M)(p,p,true);// in-place permute to save memory
        ////loadImbalance = M.LoadImbalance();
        ////if(myrank == 0) fprintf(stderr, "Load Imbalance after random permutation: %f\n", loadImbalance);
        ////nnz = M.seqptr()->getnnz();
        ////MPI_Allgather(&nnz, 1, MPI_INT, recvData.data(), 1, MPI_INT, MPI_COMM_WORLD);
        
		////stats = getStats(recvData);
		////if (myrank == 0){
            ////fprintf(stderr, "max %d\n", std::get<0>(stats));
            ////fprintf(stderr, "min %d\n", std::get<1>(stats));
            ////fprintf(stderr, "mean %lf\n", std::get<2>(stats));
            ////fprintf(stderr, "std-dev %lf\n", std::get<3>(stats));
        ////}

        ////if (myrank == 0) printf("---\n");
        //////MPI_Barrier(MPI_COMM_WORLD);
        //////for(int i = 0; i < 32; i++){
            //////for (int j = 0; j < 32; j++){
                //////int idx = i * 32 + j;
                //////if (myrank == 0) printf("| %0.1lf ", double(recvData[idx])/std::get<1>(stats));
            //////}
            //////if (myrank == 0) printf("|\n");
        //////}
        //////if (myrank == 0) printf("---\n");
        //////MPI_Barrier(MPI_COMM_WORLD);
        
        /*
         * Diagonal removal
         * */
        //p.iota(M.getnrow(), 0);
        //SpParMat<IT, NT, DER> P = SpParMat<IT,NT,DER>(M.getnrow(), M.getnrow(), p, p, 1.0, false); // Identity
        //M.SetDifference(P);
        //loadImbalance = M.LoadImbalance();
        //if(myrank == 0) fprintf(stderr, "Load Imbalance after diagonal removal: %f\n", loadImbalance);
        //nnz = M.seqptr()->getnnz();
        //MPI_Allgather(&nnz, 1, MPI_INT, recvData.data(), 1, MPI_INT, MPI_COMM_WORLD);
        
        //stats = getStats(recvData);
        //if (myrank == 0){
            //fprintf(stderr, "max %d\n", std::get<0>(stats));
            //fprintf(stderr, "min %d\n", std::get<1>(stats));
            //fprintf(stderr, "mean %lf\n", std::get<2>(stats));
            //fprintf(stderr, "std-dev %lf\n", std::get<3>(stats));
        //}
        //{
            //SpParMat<IT, NT, DER> A(M);
            //SpParMat<IT, NT, DER> B(M);

            //t0 = MPI_Wtime();
            //SpParMat<IT, NT, DER> C = Mult_AnXBn_Synch<PTFF, NT, DER>(A, B);
            //t1 = MPI_Wtime();
            //if (myrank == 0) fprintf(stderr, "Multiplication time for diagonal removed Minc: %lf\n",t1-t0 );
            //IT Cnnz = C.getnnz();
            //if (myrank == 0) fprintf(stderr, "nnz after multiplication: %lld\n",Cnnz );
        //}
        ////MPI_Barrier(MPI_COMM_WORLD);
        ////for(int i = 0; i < 32; i++){
            ////for (int j = 0; j < 32; j++){
                ////int idx = i * 32 + j;
                ////if (myrank == 0) printf("| %0.1lf ", double(recvData[idx])/std::get<1>(stats));
            ////}
            ////if (myrank == 0) printf("|\n");
        ////}
        ////if (myrank == 0) printf("---\n");
        ////MPI_Barrier(MPI_COMM_WORLD);
        //if (myrank == 0) printf("---\n");
        
        /*
         * Test custom SpGEMM
         * */
        //{
            //SpParMat<IT, NT, DER> A(M);

            //t0 = MPI_Wtime();
            //IncrementalMCLSquare<PTFF, IT, NT, DER>(
                 //A,
                 //1, 
                 //1.0/10000.0, 
                 //1100, 
                 //1400, 
                 //0.9, 
                 //1, 
                 //1, 
                 //30
            //);
            //t1 = MPI_Wtime();
            //if (myrank == 0) fprintf(stderr, "IncrementalMCLSquare time for un-altered Minc: %lf\n",t1-t0 );
        //}
        //{
            //SpParMat<IT, NT, DER> A(M);

            //t0 = MPI_Wtime();
            //MemEfficientSpGEMM<PTFF, NT, DER, IT, NT, NT, DER, DER >(
                 //A, A,
                 //1, 
                 //1.0/10000.0, 
                 //1100, 
                 //1400, 
                 //0.9, 
                 //1, 
                 //1, 
                 //30
            //);
            //t1 = MPI_Wtime();
            //if (myrank == 0) fprintf(stderr, "MemEfficientSpGEMM time for un-altered Minc: %lf\n",t1-t0 );
        //}

        /*
         * Manually test custom SpGEMM: A^2 + AD + DA + D^2 
         * */
        {
            IT MaskNNZ, DNNZ, ANNZ, ADNNZ, DANNZ, CNNZ;
            SpParMat<IT, NT, DER> MaskOffDiag(M);

            MaskNNZ = MaskOffDiag.getnnz();
            if (myrank == 0) fprintf(stderr, "nnz(MaskOffDiag) before diagonal removal: %lld\n", MaskNNZ );

            t0 = MPI_Wtime();
            FullyDistVec<IT, IT> iota( fullWorld );
            iota.iota(MaskOffDiag.getnrow(), 0); // Intialize with consecutive numbers
            FullyDistVec<IT, NT> iotaNT(iota); // To convert value types from IT to NT. Because DimApply and PruneColumn requires that.

            //Mask.DimApply(Row, iotaNT, [](NT mv, NT vv){return vv;});
            //Mask.PruneColumn(iotaNT, [](NT mv, NT vv){return static_cast<NT>(vv != mv);}, true);

            t0 = MPI_Wtime();
            IT RemovedNNZ = MaskOffDiag.RemoveLoops();
            t1 = MPI_Wtime();
            if (myrank == 0) fprintf(stderr, "Time to remove diagonal from Mask using RemoveLoops: %lf\n",t1-t0 );
            if (myrank == 0) fprintf(stderr, "Number of loops removed: %lld\n", RemovedNNZ );

            SpParMat<IT, NT, DER> D(M);
            t0 = MPI_Wtime();
            D.SetDifference(MaskOffDiag); // D: Diagonal matrix
            t1 = MPI_Wtime();
            if (myrank == 0) fprintf(stderr, "Time to get diagonal matrix: %lf\n",t1-t0 );
            DNNZ = D.getnnz();
            if (myrank == 0) fprintf(stderr, "DNNZ: %lld\n", DNNZ );
            D.ParallelWriteMM("randmat-D.mtx", 1);

            SpParMat<IT, NT, DER> A(M);
            t0 = MPI_Wtime();
            A.SetDifference(D); // A: Matrix without diagonal elements
            t1 = MPI_Wtime();
            if (myrank == 0) fprintf(stderr, "Time to get off-diagonal matrix: %lf\n",t1-t0 );
            ANNZ = A.getnnz();
            if (myrank == 0) fprintf(stderr, "ANNZ: %lld\n", ANNZ );
            A.ParallelWriteMM("randmat-A.mtx", 1);

            //A += D;
            //ANNZ = A.getnnz();
            //if (myrank == 0) fprintf(stderr, "ANNZ after adding back diagonals: %lld\n", ANNZ );

            bool eq = (A == M);
            //if (myrank == 0) fprintf(stderr, "Did we get back original matrix?: %d\n", eq );

            //A.SetDifference(D); // A: Matrix without diagonal elements

            t0 = MPI_Wtime();
            D.Apply(bind(exponentiate(), std::placeholders::_1,  2));
            t1 = MPI_Wtime();
            if (myrank == 0) fprintf(stderr, "Time to exponentiate diagonal matrix: %lf\n",t1-t0 );
            D.ParallelWriteMM("randmat-D2.mtx", 1);

            t0 = MPI_Wtime();
            FullyDistVec<IT, NT> diag = D.Reduce(Column, plus<NT>(), 0.0); // diag: Vector with diagonal entries of D
            t1 = MPI_Wtime();
            if (myrank == 0) fprintf(stderr, "Time to get diagonal vector: %lf\n",t1-t0 );

            SpParMat<IT, NT, DER> AD(A);
            t0 = MPI_Wtime();
            AD.DimApply(Column, diag, [](NT mv, NT vv){return mv * vv;});
            AD.Prune(std::bind(std::less_equal<NT>(), std::placeholders::_1,  1e-8), true);
            t1 = MPI_Wtime();
            if (myrank == 0) fprintf(stderr, "Time to perform dimApply column-wise: %lf\n",t1-t0 );
            ADNNZ = AD.getnnz();
            if (myrank == 0) fprintf(stderr, "ADNNZ: %lld\n", ADNNZ );
            AD.ParallelWriteMM("randmat-AD.mtx", 1);

            SpParMat<IT, NT, DER> DA(A);
            t0 = MPI_Wtime();
            DA.DimApply(Row, diag, [](NT mv, NT vv){return mv * vv;});
            DA.Prune(std::bind(std::less_equal<NT>(), std::placeholders::_1,  1e-8), true);
            t1 = MPI_Wtime();
            if (myrank == 0) fprintf(stderr, "Time to perform dimApply row-Wise: %lf\n",t1-t0 );
            DANNZ = DA.getnnz();
            if (myrank == 0) fprintf(stderr, "DANNZ: %lld\n", DANNZ );
            DA.ParallelWriteMM("randmat-DA.mtx", 1);

            SpParMat<IT, NT, DER> B(A);
            t0 = MPI_Wtime();
            SpParMat<IT, NT, DER> AB = Mult_AnXBn_Synch<PTFF, NT, DER>(A, B);
            t1 = MPI_Wtime();
            if (myrank == 0) fprintf(stderr, "Multiplication time for non-diagonal matrix: %lf\n",t1-t0 );
            IT ABNNZ = AB.getnnz();
            if (myrank == 0) fprintf(stderr, "ABNNZ: %lld\n", ABNNZ );
            AB.ParallelWriteMM("randmat-AB.mtx", 1);

            t0 = MPI_Wtime();
            std::vector< SpTuples<IT,NT>*> tomerge;
            if(!AB.seqptr()->isZero()) tomerge.push_back(new SpTuples<IT, NT>( *AB.seqptr() ) );
            if(!D.seqptr()->isZero()) tomerge.push_back(new SpTuples<IT, NT>( *D.seqptr() ) );
            if(!AD.seqptr()->isZero()) tomerge.push_back(new SpTuples<IT, NT>( *AD.seqptr() ) );
            if(!DA.seqptr()->isZero()) tomerge.push_back(new SpTuples<IT, NT>( *DA.seqptr() ) );

            SpTuples<IT,NT> * OnePieceOfC_tuples;
            OnePieceOfC_tuples = MultiwayMergeHash<PTFF>(tomerge, A.seqptr()->getnrow(), A.seqptr()->getncol(), true, false);
            DER * OnePieceOfC = new DER(* OnePieceOfC_tuples, false);
            delete OnePieceOfC_tuples;
        
            SpParMat<IT, NT, DER> C = SpParMat<IT, NT, DER>(OnePieceOfC, fullWorld);
            t1 = MPI_Wtime();

            //t0 = MPI_Wtime();
            //C += D;
            //C += AD;
            //C += DA;
            //t1 = MPI_Wtime();
            if (myrank == 0) fprintf(stderr, "Time for adding matrices in custom multiplication: %lf\n",t1-t0 );
            CNNZ = C.getnnz();
            C.ParallelWriteMM("randmat-C.mtx", 1);

            eq = (C == X);
            if (myrank == 0) fprintf(stderr, "Is the custom multiplication correct?: %d\n", eq );
            if (myrank == 0) fprintf(stderr, "%lld == %lld\n", CNNZ, XNNZ );

            eq = (C == A);
            if (myrank == 0) fprintf(stderr, "Is C == A?: %d\n", eq );

        }
    }
    MPI_Finalize();
    return 0;
}

