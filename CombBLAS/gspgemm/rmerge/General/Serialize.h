#pragma once

#include "General/ClosingFile.h"
#include "General/WinOnlyStatic.h"
#include <string>
#include "HostMatrix/int64.h"
namespace ExMI{

	template<typename T> 
	static void Serialize(ClosingFile& file, const T& t){
		fwrite(&t,sizeof(t),1,file.GetFile());
	}

	template<>
	WinOnlyStatic inline void Serialize<std::string>(ClosingFile& file, const std::string& s){
		Serialize(file,uint64(s.size()));
		for(uint64 i=0;i<s.size();i++)
			Serialize(file,s[i]);
	}

	template<typename T> 
	static void Serialize(ClosingFile& file, const T* p, int64 n){
		fwrite(p,sizeof(T),n,file.GetFile());
	}

	template<typename T> 
	static T Deserialize(ClosingFile& file){
		T t;
		fread(&t,sizeof(t),1,file.GetFile());
		return t;
	}

	template<>
	WinOnlyStatic inline std::string Deserialize<std::string>(ClosingFile& file){
		uint64 length=Deserialize<uint64>(file);
		std::string s;s.resize(length);
		for(uint64 i=0;i<s.size();i++)
			s[i]=Deserialize<char>(file);
		return s;
	}



	template<typename T> 
	static void Deserialize(ClosingFile& file, T* p, int64 n){
		fread(p,sizeof(T),n,file.GetFile());
	}

}