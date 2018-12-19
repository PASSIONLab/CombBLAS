#pragma once
#include "General/EndsWith.h"

#include <vector>
#include <string>

static std::vector<std::string> Split(const std::string& str, const std::string& delimiters){
	std::vector<std::string> tokens;
	// Skip delimiters at beginning.
	std::string::size_type lastPos = str.find_first_not_of(delimiters, 0);
	// Find first "non-delimiter".
	std::string::size_type pos     = str.find_first_of(delimiters, lastPos);

	while (std::string::npos != pos || std::string::npos != lastPos)
	{
		// Found a token, add it to the vector.
		tokens.push_back(str.substr(lastPos, pos - lastPos));
		// Skip delimiters.  Note the "not_of"
		lastPos = str.find_first_not_of(delimiters, pos);
		// Find next "non-delimiter"
		pos = str.find_first_of(delimiters, lastPos);
	}
	return tokens;
}
static std::string Remove(const std::string& s, char c){
	std::string a;
	for(int i=0;i<(int)s.size();i++){
		if(s[i]==c)
			continue;
		else 
			a.push_back(s[i]);
	}
	return a;
}

static void Replace(std::string& s, char c, char replacement){	
	for(int i=0;i<s.length();i++)
	{
		if(s[i]==c)
			s[i]=replacement;
	}
}

static void Replace(std::string& s, std::string a, std::string replacement){	
	std::string tmp=s;
	std::string out="";
	while(tmp.length()>0){
		if(StartsWith(tmp,a)){
			out=out+replacement;
			tmp=tmp.substr(a.size(),tmp.length()-a.size());
		}
		else
		{
			out.push_back(tmp[0]);
			tmp=tmp.substr(1,tmp.size()-1);
		}
	}
	s=out;
}

//Remove blanks, e.g. \r 
static std::string Trim(const std::string& s){
	int start=0;
	int end=(int)s.size();
	for(int i=0;i<(int)s.size();i++){
		char c=s[i];
		if(c==' ' || c=='\r' || c=='\n' || c=='\t')
			start=i+1;
		else
			break;
	}

	for(int i=(int)s.size()-1;i>=0;i--){
		char c=s[i];
		if(c==' ' || c=='\r' || c=='\n' || c=='\t')
			end=i;
		else
			break;
	}
	if(end<start)
		return "";
	return s.substr(start,end-start);
}