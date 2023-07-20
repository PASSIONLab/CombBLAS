//  Project AC-SpGEMM
//  https://www.tugraz.at/institute/icg/research/team-steinberger/
//
//  Copyright (C) 2018 Institute for Computer Graphics and Vision,
//                     Graz University of Technology
//
//  Author(s):  Martin Winter - martin.winter (at) icg.tugraz.at
//              Daniel Mlakar - daniel.mlakar (at) icg.tugraz.at
//              Rhaleb Zayer - rzayer (at) mpi-inf.mpg.de
//              Hans-Peter Seidel - hpseidel (at) mpi-inf.mpg.de
//              Markus Steinberger - steinberger ( at ) icg.tugraz.at
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to deal
//  in the Software without restriction, including without limitation the rights
//  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in
//  all copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
//  THE SOFTWARE.
//

/*!/------------------------------------------------------------------------------
* CustomExceptions.h
*
* ac-SpGEMM
*
* Authors: Daniel Mlakar, Markus Steinberger, Martin Winter
*------------------------------------------------------------------------------
*/
#pragma once

#include <exception>

class SpGEMMException : public std::exception
{
public:
	virtual char const * what()  const noexcept{ return "SpGEMM: Stage failed"; }
};

class MergeSimpleCaseException : public std::exception
{
public:
virtual char const * what() const noexcept { return "MERGE: Simple Case failed"; }
};

class MergeMaxChunksCaseException : public std::exception
{
public:
	virtual char const * what() const noexcept { return "MERGE: Max Chunks Case failed"; }
};

class MergeGeneralizedCaseException : public std::exception
{
public:
	virtual char const * what() const noexcept { return "MERGE: Generalized Case failed"; }
};

class MergeLoopingException : public std::exception
{
public:
	virtual char const * what() const noexcept { return "MERGE: Merge Stage took longer than 10 seconds"; }
};

class RestartOutOfMemoryException : public std::exception
{
public:
	virtual char const * what() const noexcept { return "RESTART: SpGEMM out of memory"; }
};

class RestartOutOfChunkPointerException : public std::exception
{
public:
	virtual char const * what() const noexcept { return "RESTART: SpGEMM out of chunk pointers"; }
};

