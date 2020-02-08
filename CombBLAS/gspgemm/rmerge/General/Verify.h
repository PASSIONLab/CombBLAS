#pragma once

//This file defines the Verify function.

#include <exception>
#include <stdexcept>
#include <string>


#define _STRING_LINE_(s) #s
#define _STRING_LINE2_(s) _STRING_LINE_(s)
#define __LINESTR__ _STRING_LINE2_(__LINE__)
#define FileAndLine __FILE__ ":" __LINESTR__


//Checks an expression at runtime and throws an exception iff false.
//The message is used for the exception.
static void Verify(bool expression,const char* message){
	if(!expression)
	{
		if(message==0)
			throw std::runtime_error("Verify failed.");
		else{
			//std::cout<<"Error "<<msg<<std::endl;
			throw std::runtime_error(message);
		}
	}	
}

//Checks an expression at runtime and throws an exception iff false.
//The message is used for the exception.
//This is slower...
static void Verify(bool expression,const std::string& message){
	Verify(expression,message.c_str());
}

//Checks an expression in debug mode
static void Assert_rmerge(bool expression,const char* message){
	#ifdef DEBUG
	Verify(expression,message);
	#endif
}
