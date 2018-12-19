#pragma once

#include <algorithm>
#include <string>

static bool EndsWith(const std::string& str, const std::string& suffix){
	if(str.size()<suffix.size())return false;
	return std::equal(suffix.rbegin(), suffix.rend(), str.rbegin());
}

static bool StartsWith(const std::string& str, const std::string& prefix){
	if(str.size()<prefix.size())
		return false;
	return std::equal(prefix.begin(), prefix.end(), str.begin());
}

static std::string ToLower(std::string a){
	std::string b=a;
	std::transform(b.begin(), b.end(), b.begin(), ::tolower);
	return b;
}