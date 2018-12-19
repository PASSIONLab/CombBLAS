#pragma once
#include <string>
#include <sstream>
#include <iostream> 
#include <iomanip>
#include "General/WinOnlyStatic.h"

template<typename T>
static std::string ToString(T t){
	std::stringstream sstr;
	sstr<<std::setprecision(19);
	sstr<<t;
	return sstr.str();
}

template<typename T>
static std::string ToString(T t, int precision){
	std::stringstream sstr;
	sstr<<std::fixed;
	sstr<<std::setprecision(precision);
	sstr<<t;
	return sstr.str();
}

template<>
WinOnlyStatic inline std::string ToString<>(double d){
    std::stringstream sstr;
    sstr<<d;
    return sstr.str();
}

template<>
WinOnlyStatic inline std::string ToString<>(int i){
    std::stringstream sstr;
    sstr<<i;
    return sstr.str();
}

template<>
WinOnlyStatic inline std::string ToString<>(const std::string& s){
    return s;
}

static std::string PadLeft(std::string s, std::string padding, int maxSize){
    while((int)s.size()<maxSize)
        s=padding+s;
    return s;
}

//1 becomes 01
static std::string ToStringPad2(int i){
    std::stringstream sstr;
    if(i>=0&&i<=9)
        sstr<<"0";
    sstr<<i;
    return sstr.str();
}

static std::string ToStringPad3(int i){
    std::stringstream sstr;
    if(i>=0&&i<=9)
        sstr<<"00";
	else if(i>=10 && i<=99)
		sstr<<"0";
    sstr<<i;
    return sstr.str();
}

template<typename T>
static T Parse(const std::string& s){
    std::stringstream sstr(s);
    T tmp;
    sstr >> tmp;
    return tmp;
}

template<>
WinOnlyStatic inline bool Parse<bool>(const std::string& s){
    return s=="true" || s=="1" || s=="TRUE";
}

template<>
WinOnlyStatic inline std::string Parse<std::string>(const std::string& s){
    return s;
}

static int ToInt(const std::string& s){
	return Parse<int>(s);
}

static double ToDouble(const std::string& s){
	return Parse<double>(s);
}

static float ToFloat(std::string s){
	return Parse<float>(s);
}


//value in [0,15] is converted to ['0'..'9''A'...'F']
static char HexDigit(unsigned char c)
{
   if(c<10)
     return char(c+'0');
   else
     return char(c-10+'A');
}

//byte to hex string
static std::string ToHexString(unsigned char c)
{
    char bla[3];
    bla[0]=HexDigit(c>>4);
    bla[1]=HexDigit(c&0xF);
    bla[2]='\0';
    return std::string(bla);
}
