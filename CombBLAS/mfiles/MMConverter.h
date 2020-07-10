#ifndef _MM_CONVERTER_
#define _MM_CONVERTER_

#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <iterator>
#include <omp.h>
#include <sys/stat.h>
#include <string.h>
#include <omp.h>
#include "mmio.h"
#include <zlib.h>
#include "Tommy/tommyhashdyn.h"
#include "compress_string.h"
#include "TommyObj.h"
using namespace std;

#define BATCH 100000000  // 100MB
#define MAXLINELENGTH 512
#define VERTEX_HEAD ""
#define MAXVERTNAME 128

string chop_head(const string & full_str, const string & head_str)
{
	if (full_str.compare(0, head_str.length(), head_str) == 0)	// begins_with
	{
		return full_str.substr(head_str.length(), string::npos);
	}
	else
	{
		cout << "String doesn't start with " << head_str << endl;
		return full_str;
	}
} 


// typedef void tommy_foreach_arg_func(void* arg, void* obj);

void* shuffledprintfunc(void* arg, void* obj)
{
    pair<vector<uint32_t>*, ofstream*> * mypair = (pair<vector<uint32_t>*, ofstream*> *) arg;    // cast argument
    vector<uint32_t> * shuffler = mypair->first;
    ofstream * out = mypair->second;
#ifdef COMPRESS_STRING
        (*out) << (*shuffler)[((tommy_object *) obj)->vid] << "\t" << decompress_string(((tommy_object *) obj)->vname) << "\n";
#else
        (*out) << (*shuffler)[((tommy_object *) obj)->vid] << "\t" << ((tommy_object *) obj)->vname << "\n";
#endif
}


template <typename IT1, typename NT1, typename IT2, typename NT2>
void push_to_vectors(vector<IT1> & rows, vector<IT1> & cols, vector<NT1> & vals, IT2 ii, IT2 jj, NT2 vv)
{
    rows.push_back(ii);
    cols.push_back(jj);
    vals.push_back(vv);
}

template <typename IT1, typename NT1>
void ProcessLines(vector<IT1> & rows, vector<IT1> & cols, vector<NT1> & vals, vector<string> & lines, tommy_hashdyn & hashdyn, const vector<uint32_t> & shuffler)
{
    char from[MAXVERTNAME];
    char to[MAXVERTNAME];
    double vv;
    for (vector<string>::iterator itr=lines.begin(); itr != lines.end(); ++itr)
    {
        // string::c_str() -> Returns a pointer to an array that contains a null-terminated sequence of characters (i.e., a C-string)
        sscanf(itr->c_str(), "%s %s %lg", from, to, &vv);
        string s_from = chop_head(string(from), VERTEX_HEAD);
        string s_to = chop_head(string(to), VERTEX_HEAD);
        
        tommy_object * obj1 = (tommy_object *) tommy_hashdyn_search(&hashdyn, compare, &s_from, tommy_hash_u32(0, from, strlen(from)));
        tommy_object * obj2 = (tommy_object *) tommy_hashdyn_search(&hashdyn, compare, &s_to, tommy_hash_u32(0, to, strlen(to)));


        uint32_t fr_id = shuffler[obj1->vid];
        uint32_t to_id = shuffler[obj2->vid];
        push_to_vectors(rows, cols, vals, fr_id, to_id, vv);
    }
    vector<string>().swap(lines);
}


void check_newline(int *bytes_read, int bytes_requested, char *buf)
{
    if ((*bytes_read) < bytes_requested) {
        // fewer bytes than expected, this means EOF
        if (buf[(*bytes_read) - 1] != '\n') {
            // doesn't terminate with a newline, add one to prevent infinite loop later
            buf[(*bytes_read) - 1] = '\n';
            cout << "Error in input format, appending missing newline at end of file" << endl;
            (*bytes_read)++;
        }
    }
}

// updates to curpos are reflected in the caller function
bool FetchBatch(FILE * f_local, long int & curpos, long int end_fpos, bool firstcall, vector<string> & lines)
{
    size_t bytes2fetch = BATCH;    // we might read more than needed but no problem as we won't process them
    bool begfile = (curpos == 0);
    if(firstcall && (!begfile))
    {
        curpos -= 1;    // first byte is to check whether we started at the beginning of a line
        bytes2fetch += 1;
    }
    char * buf = new char[bytes2fetch]; // needs to happen **after** bytes2fetch is updated
    char * originalbuf = buf;   // so that we can delete it later because "buf" will move
    
    int seekfail = fseek(f_local, curpos, SEEK_SET); // move the file pointer to the beginning of thread data
    if(seekfail != 0)
        cout << "fseek failed to move to " << curpos << endl;
    
    int bytes_read = fread(buf, sizeof(char), bytes2fetch, f_local);  // read byte by byte
    if(!bytes_read)
    {
        delete [] originalbuf;
        return true;    // done
    }
    check_newline(&bytes_read, bytes2fetch, buf);
    if(firstcall && (!begfile))
    {
        if(buf[0] == '\n')  // we got super lucky and hit the line break
        {
            buf += 1;
            bytes_read -= 1;
            curpos += 1;
        }
        else    // skip to the next line and let the preceeding thread take care of this partial line
        {
            char *c = (char*)memchr(buf, '\n', MAXLINELENGTH); //  return a pointer to the matching byte or NULL if the character does not occur
            if (c == NULL) {
                cout << "Unexpected line without a break" << endl;
            }
            int n = c - buf + 1;
            bytes_read -= n;
            buf += n;
            curpos += n;
        }
    }
    while(bytes_read > 0 && curpos < end_fpos)  // this will also finish the last line
    {
        char *c = (char*)memchr(buf, '\n', bytes_read); //  return a pointer to the matching byte or NULL if the character does not occur
        if (c == NULL) {
            delete [] originalbuf;
            return false;  // if bytes_read stops in the middle of a line, that line will be re-read next time since curpos has not been moved forward yet
        }
        int n = c - buf + 1;
        
        // string constructor from char * buffer: copies the first n characters from the array of characters pointed by s
        lines.push_back(string(buf, n-1));  // no need to copy the newline character
        bytes_read -= n;   // reduce remaining bytes
        buf += n;   // move forward the buffer
        curpos += n;
    }
    delete [] originalbuf;
    if (curpos >= end_fpos) return true;  // don't call it again, nothing left to read
    else    return false;
}

void MMConverter(const string & filename, ofstream & dictout, const string & outprefix)
{
    FILE *f;
    if ((f = fopen(filename.c_str(), "r")) == NULL)
    {
        printf("file can not be found\n");
        exit(1);
    }
        
    // Use fseek again to go backwards two bytes and check that byte with fgetc
    struct stat st;     // get file size
    if (stat(filename.c_str(), &st) == -1)
    {
        exit(1);
    }
    int64_t file_size = st.st_size;
    cout << "File is " << file_size << " bytes" << endl;
    long int ffirst = ftell(f); // doesn't change
    long int fpos = ffirst;
    long int end_fpos = file_size;
    
    vector<string> lines;
    bool finished = FetchBatch(f, fpos, end_fpos, true, lines); // fpos will move
    int64_t entriesread = lines.size();
   
    tommy_hashdyn hashdyn;
    tommy_hashdyn_init(&hashdyn);
    
    char from[MAXVERTNAME];
    char to[MAXVERTNAME];
    double vv;
    uint32_t vertexid = 0;
    for (vector<string>::iterator itr=lines.begin(); itr != lines.end(); ++itr)
    {
        // string::c_str() -> Returns a pointer to an array that contains a null-terminated sequence of characters (i.e., a C-string)
        sscanf(itr->c_str(), "%s %s %lg", from, to, &vv);
        string s_from = chop_head(string(from), VERTEX_HEAD);
        string s_to = chop_head(string(to), VERTEX_HEAD);

        tommy_object* obj1 = (tommy_object*) tommy_hashdyn_search(&hashdyn, compare, &s_from, tommy_hash_u32(0, from, strlen(from)));
        if(!obj1)
        {
            tommy_object * obj1 = new tommy_object(vertexid, s_from);   // vertexid is the vid
            tommy_hashdyn_insert(&hashdyn, &(obj1->node), obj1, tommy_hash_u32(0, from, strlen(from)));
            ++vertexid; // new entry
        }
        
        tommy_object* obj2 = (tommy_object*) tommy_hashdyn_search(&hashdyn, compare, &s_to, tommy_hash_u32(0, to, strlen(to)));
        if(!obj2)
        {
            tommy_object* obj2 = new tommy_object(vertexid, s_to);   // vertexid is the vid
            tommy_hashdyn_insert(&hashdyn, &(obj2->node), obj2, tommy_hash_u32(0, to, strlen(to)));
            ++vertexid; // new entry
        }
    }
    vector<string>().swap(lines);
    
    while(!finished)
    {
        finished = FetchBatch(f, fpos, end_fpos, false, lines);
        entriesread += lines.size();
        cout << "entriesread: " << entriesread << ", current vertex id: " << vertexid << endl;
        
        // Process files
        char from[MAXVERTNAME];
        char to[MAXVERTNAME];
        double vv;
        for (vector<string>::iterator itr=lines.begin(); itr != lines.end(); ++itr)
        {
            // string::c_str() -> Returns a pointer to an array that contains a null-terminated sequence of characters (i.e., a C-string)
            sscanf(itr->c_str(), "%s %s %lg", from, to, &vv);
            
            string s_from = chop_head(string(from), VERTEX_HEAD);
            string s_to = chop_head(string(to), VERTEX_HEAD);
            
            tommy_object* obj1 = (tommy_object*) tommy_hashdyn_search(&hashdyn, compare, &s_from, tommy_hash_u32(0, from, strlen(from)));
            if(!obj1)
            {
                tommy_object* obj1 = new tommy_object(vertexid, s_from);   // vertexid is the vid
                tommy_hashdyn_insert(&hashdyn, &(obj1->node), obj1, tommy_hash_u32(0, from, strlen(from)));
                ++vertexid; // new entry
            }
            
            tommy_object* obj2 = (tommy_object*) tommy_hashdyn_search(&hashdyn, compare, &s_to, tommy_hash_u32(0, to, strlen(to)));
            if(!obj2)
            {
                tommy_object* obj2 = new tommy_object(vertexid, s_to);   // vertexid is the vid
                tommy_hashdyn_insert(&hashdyn, &(obj2->node), obj2, tommy_hash_u32(0, to, strlen(to)));
                ++vertexid; // new entry
            }
        }
        vector<string>().swap(lines);
    }
    cout << "There are " << vertexid << " vertices and " << entriesread << " edges" << endl;

#ifdef SUBGRAPHS
#define NSUBGRAPHS 6
    uint32_t ranges[NSUBGRAPHS] = {vertexid, vertexid/2, vertexid/4, vertexid/8, vertexid/16, vertexid/32};
    cout << "Printing submatrices with the following numbers of vertices: ";
    copy(ranges, ranges+NSUBGRAPHS, ostream_iterator<uint32_t>(cout," ")); cout << endl;
#endif

    uint32_t nvertices = vertexid;
    vector< uint32_t > shuffler(nvertices);
    iota(shuffler.begin(), shuffler.end(), static_cast<uint32_t>(0));
    random_shuffle ( shuffler.begin(), shuffler.end() );
   
    pair< vector<uint32_t>*, ofstream*> mypair(&shuffler, &dictout);
    tommy_hashdyn_foreach_arg(&hashdyn, (tommy_foreach_arg_func *) shuffledprintfunc, &mypair);
 
    cout << "Shuffled and wrote dictionary " << endl;
    fclose(f);

    
#pragma omp parallel
    {
        long int fpos, end_fpos; // override
        int this_thread = omp_get_thread_num();
        int num_threads = omp_get_num_threads();

        if(this_thread == 0) fpos = ffirst;
        else fpos = this_thread * file_size / num_threads;
        
	
#ifdef SUBGRAPHS
        string names[NSUBGRAPHS];
        ofstream outfiles[NSUBGRAPHS];
        for(int i= 0; i<NSUBGRAPHS; i++)
        {
            names[i] = "Renamed_subgraph";
            names[i] += std::to_string(i);
            names[i] += "_";
            names[i] += outprefix;
            names[i] += std::to_string(this_thread);
            cout << names[i] << endl;
            outfiles[i].open(names[i]);
        }
#else
	string name;
        ofstream outfile;
 	name = "Renamed_graph_";
	name += outprefix;
	name += std::to_string(this_thread);
	cout << name<< endl;
	outfile.open(name);
#endif
       
        if(this_thread != (num_threads-1)) end_fpos = (this_thread + 1) * file_size / num_threads;
        else end_fpos = file_size;
    
        FILE * f_perthread = fopen(filename.c_str(), "rb");   // reopen
        vector<string> lines;
        bool finished = FetchBatch(f_perthread, fpos, end_fpos, true, lines);
        size_t nnz = lines.size();
        vector<uint32_t> rows;
        vector<uint32_t> cols;
        vector<float> vals;
        ProcessLines(rows, cols, vals, lines, hashdyn, shuffler);
        
        if(this_thread == 0)
        {
            cout << "there are " << num_threads << " threads" << endl;
#ifdef SUBGRAPHS
            for(int i= 0; i<NSUBGRAPHS; i++)
            {
                outfiles[i] << "%%MatrixMarket matrix coordinate real symmetric\n";
                outfiles[i] << ranges[i] << "\t" << ranges[i] << "\t" << entriesread << "\n";
            }
#else
	     outfile << "%%MatrixMarket matrix coordinate real symmetric\n";
	     outfile << nvertices << "\t" << nvertices << "\t" << entriesread << "\n";
#endif
        }
        for(size_t k=0; k< nnz; ++k)
        {
#ifdef SUBGRAPHS
            for(int i= 0; i<NSUBGRAPHS; i++)
            {
                if(rows[k] < ranges[i] && cols[k] < ranges[i])
                    outfiles[i] << rows[k] << "\t" << cols[k] << "\t" << vals[k] << "\n";
            }
#else
	    outfile << rows[k] << "\t" << cols[k] << "\t" << vals[k] << "\n";
#endif

        }
        rows.clear();
        cols.clear();
        vals.clear();
        
        while(!finished)
        {
            finished = FetchBatch(f_perthread, fpos, end_fpos, false, lines);
            nnz = lines.size(); // without this in this exact place, it is buggy
            ProcessLines(rows, cols, vals, lines, hashdyn, shuffler);
            
            for(size_t k=0; k< nnz; ++k)
            {
#ifdef SUBGRAPHS
                for(int i= 0; i<NSUBGRAPHS; i++)
                {
                    if(rows[k] < ranges[i] && cols[k] < ranges[i])
                        outfiles[i] << rows[k] << "\t" << cols[k] << "\t" << vals[k] << "\n";
                }
#else
		outfile << rows[k] << "\t" << cols[k] << "\t" << vals[k] << "\n";
#endif
            }
            rows.clear();
            cols.clear();
            vals.clear();
        }
#ifdef SUBGRAPHS
        for(int i= 0; i<NSUBGRAPHS; i++)
        {
            outfiles[i].close();
        }
#else
	outfile.close();
#endif
		
    }
    

    tommy_hashdyn_foreach(&hashdyn, operator delete);
    tommy_hashdyn_done(&hashdyn);
}

#endif

