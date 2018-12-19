#pragma once
#include <string>
#include <stdio.h>
#include "General/fopen_s.h"

//This class is a convenience wrapper around the C struct FILE
//This is useful to use FILE* and still be exception safe.
class ClosingFile{
	FILE* file;	//The actual file
	
	//private copy constructor to disable usage
	ClosingFile(const ClosingFile& rhs){} 

public:

	//Default constructor
	ClosingFile():file(0){}
	
	//Constructor that takes a file name and mode as input
	ClosingFile(std::string fileName, std::string mode){
		int e=fopen_s(&file,fileName.c_str(),mode.c_str());
		if(e==ENOENT)
			throw std::runtime_error("File does not exist. "+fileName);
		else if(e==EACCES)
			throw std::runtime_error("Cannot access the file. Is it opened by another program?");
		else if(e!=0)
			throw std::runtime_error("Error opening the file "+fileName);

	}

	//Destructor
	~ClosingFile(){
		Close();
	}

	//Closes the internal file
	void Close(){
		if(file!=0){
			fclose(file);
			file=0;
		}
	}

	//Returns the file pointer
	FILE* GetFile(){return file;}
};