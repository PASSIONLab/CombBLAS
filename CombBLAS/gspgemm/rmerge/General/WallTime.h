#pragma once

//This file defines a timer for performance measurements.
//Different implementations are provided using boost and C++11

//Either use boost or C++ 11 (chrono)
#define USE_BOOST_TIMER

#ifdef USE_BOOST_TIMER
#include <boost/timer/timer.hpp>
//This is required to avoid a linker error (this is a  problem in boost since version 1.60.0)
#include <boost/chrono/time_point.hpp>

namespace ExMI{

	//Implementation using boost
	class WallTime{
		boost::timer::cpu_timer timer;
	public:
		//Default constructor, starts the timer
		WallTime(){
			timer=boost::timer::cpu_timer();
		}

		//Resets the timer
		void Reset(){
			timer=boost::timer::cpu_timer();
		}

		//Returns the number of miliseconds passed
		double Milliseconds(){
			return Seconds()*1000.0;
		}

		//Returns the number of seconds passed. 
		double Seconds(){
			boost::timer::cpu_times times=timer.elapsed();	
			boost::timer::nanosecond_type wall=times.wall;	
			double secondsElapsed=double(wall)/1000000000.0;
			return secondsElapsed;
		}
	};
}
#else

#include <chrono>
namespace ExMI{

	//Implementation using 
	class WallTime{
		std::chrono::high_resolution_clock::time_point start; 
	public:

		//Default constructor, starts the timer
		WallTime(){
			start = std::chrono::high_resolution_clock::now();
		}

		//Resets the timer
		void Reset(){
			start = std::chrono::high_resolution_clock::now();
		}

		//Returns the number of miliseconds passed
		//Unfortunately this has low resolution with VC 2012
		double Milliseconds(){
			using namespace std;
			using namespace chrono;
			high_resolution_clock::time_point end = high_resolution_clock::now();			
			milliseconds time_span = duration_cast<std::chrono::milliseconds>(end-start);
			return (double)time_span.count();
		}

		//Returns the number of seconds passed. 
		double Seconds(){return Milliseconds()*0.001;}
	};
}

#endif
