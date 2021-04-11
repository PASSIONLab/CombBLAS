#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <string.h>
#include <assert.h>
#include <omp.h>
using namespace std;

template <typename T>
vector<size_t> sortIndices(const vector<T> &v) {
    
    // initialize original index locations
    vector<size_t> idx(v.size());
    iota(idx.begin(), idx.end(), 0);
    
    // sort indexes based on comparing values in v
    sort(idx.begin(), idx.end(),
         [&v](size_t i1, size_t i2) {return v[i1] > v[i2];});
    
    return idx;
}



double Fscore(string mclFile, string hipmclFile, int base)
{
    ifstream mclStream (mclFile);
    ifstream hipmclStream (hipmclFile);
    string line;
    int64_t item, clustID = 0;
    int64_t numMCLClusters, numHipMCLClusters;
    int64_t nproteins = 0;
    // item 0-based
    
    vector<vector<int64_t>> mclClusters;
    
    if (mclStream.is_open())
    {
        while ( getline (mclStream,line) )
        {
            istringstream iss(line);
            if(clustID >= mclClusters.size())
            {
                mclClusters.resize(clustID * 2 + 1);
            }
            while ( iss >> item)
            {
                if(base==1) item --;
                mclClusters[clustID].push_back(item);
            }
            
            nproteins += mclClusters[clustID].size();
            clustID++;
        }
        mclStream.close();
        numMCLClusters = clustID;
    }
    else
    {
        cout << "Unable to open " << mclFile << endl;
        return -1;
    }
    
    cout << "Number of clusters from MCL: " << numMCLClusters << endl;
    vector<int64_t> clust2(nproteins, -1);
    clustID = 0;
    if (hipmclStream.is_open())
    {
        while ( getline (hipmclStream,line) )
        {
            istringstream iss(line);
            while ( iss >> item)
            {
                if(base==1) item --;
                clust2[item] = clustID;
                if(item >= nproteins)
                {
                    cout << "The number of vertices in MCL and HipMCL outputs does not match. \n Please check if there are isolated vertices in HipMCL.\n Exiting.............." << endl;
                    exit(1);
                }
            }
            clustID++;
        }
        hipmclStream.close();
        numHipMCLClusters = clustID;
    }
    else
    {
        cout << "Unable to open " << hipmclFile << endl;
        return -1;
    }
    
    // this will account for isolated vertices. These vertices are assigned to their own clusters 
    for(int i=0; i<nproteins; i++)
    {
        if(clust2[i]==-1)
		clust2[i]=numHipMCLClusters++;
    }
    cout << "Number of clusters from HipMCL: " << numHipMCLClusters << endl;
    
    vector<int64_t> clusterSizes2(numHipMCLClusters,0);
    for(int i=0; i<nproteins; i++)
    {
        clusterSizes2[clust2[i]]++;
    }
    
    
    vector<double> F(numMCLClusters);
    vector<int64_t> nisect(numHipMCLClusters); // number of items in the intersection
    vector<int64_t> isect(numHipMCLClusters); // clusters with nonzero intersection with the current cluster
    int mismatch = 0;

    for(int i=0; i<numMCLClusters; i++)
    {
        int64_t isectCount = 0;
        fill(nisect.begin(), nisect.end(), 0);
        for(int j=0; j<mclClusters[i].size(); j++)
        {
            int64_t item1 = mclClusters[i][j];
            int64_t c2 = clust2[item1];
            if(nisect[c2]==0) isect[isectCount++] = c2;
            nisect[c2] ++;
        }
        auto maxoverlap = max_element(nisect.begin(), nisect.end());
        int64_t c2_max = distance(nisect.begin(), maxoverlap);
        if(*maxoverlap!=mclClusters[i].size() || *maxoverlap!=clusterSizes2[c2_max])
        {
            cout << "Mismatch# " << mismatch++ << ":: MCL Cluster: "<< i << " Size: " << mclClusters[i].size() << " HipMCL Cluster: "<< c2_max <<" Size: " << clusterSizes2[c2_max] << " last vertex in MCL output: " << mclClusters[i].back()<< endl;
            
        }
        double Fi = 0;
        for(int j=0; j<isectCount; j++)
        {
            int64_t c2 = isect[j];
            double precision = (double) nisect[c2] / clusterSizes2[c2];
            double recall = (double) nisect[c2] / mclClusters[i].size();
            double Fij = 2 * precision * recall / (precision + recall);
            Fi = max(Fi, Fij);
        }
        
        Fi = Fi * mclClusters[i].size()/ nproteins;
        F[i] = Fi;
    }
    return accumulate(F.begin(), F.end(), 0.0);
}

int main(int argc, char* argv[])
{
    
    string ifilename1 = "";
    string ifilename2 = "";
    int base = 0;
  
    
    if(argc != 7)
    {
        cout << "Usage: ./fscore -M1 <MCLOut> -M2 <HipMCLOut> -base <Base of vertices (same as HipMCL) 1 or 0>\n";
        cout << "Example: ./fscore -M1 input1.txt -M2 input2.txt -base 0 " << endl;
        return -1;
    }
    
    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i],"-M1")==0)
        {
            ifilename1 = string(argv[i+1]);
            printf("\nMCL output file: %s",ifilename1.c_str());
        }
        else if (strcmp(argv[i],"-M2")==0)
        {
            ifilename2 = string(argv[i+1]);
            printf("\nHipMCL output file: %s",ifilename2.c_str());
        }
        else if (strcmp(argv[i],"-base")==0)
        {
            base = atoi(argv[i+1]);
            printf("\nbase: %d",base);
        }
    }
    printf("\n");
    double F = Fscore(ifilename1, ifilename2, base);
    cout << "F score: " << F << endl;
    return 0;
}
