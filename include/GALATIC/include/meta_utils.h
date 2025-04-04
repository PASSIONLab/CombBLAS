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


#ifndef INCLUDED_HIS_META_UTILS
#define INCLUDED_HIS_META_UTILS

#pragma once

#include <utility>
#include <type_traits>
#include "multi_arch_build.h"


	using std::enable_if;
	using std::declval;
	using std::is_empty;
	using std::conditional;

	template <class A, class B>
	struct type_match
	{
		static const bool value = false;
	};

	template <class A>
	struct type_match<A, A>
	{
		static const bool value = true;
	};

	template<int X, int Y>
	struct static_divup
	{
		static const int value = (X + Y - 1) / Y;
	};

	template<int X>
	struct static_popcnt
	{
		static const int value = ((X & 0x1) + static_popcnt< (X >> 1) >::value);
	};
	template<>
	struct static_popcnt<0>
	{
		static const int value = 0;
	};

	template<unsigned int X, int Completed = 0>
	struct static_clz
	{
		static const int value = (X & 0x80000000) ? Completed : static_clz< (X << 1), Completed + 1 >::value;
	};
	template<unsigned int X>
	struct static_clz<X, 32>
	{
		static const int value = 32;
	};

	template<int... VALUES>
	struct static_max;

	template<int VALUE>
	struct static_max<VALUE>
	{
		static const int value = VALUE;
	};

	template<int VALUE, int... VALUES>
	struct static_max<VALUE, VALUES...>
	{
		static const int next_value = static_max<VALUES...>::value;
		static const int value = VALUE > next_value ? VALUE : next_value;
	};

	template<int... VALUES>
	struct static_min;

	template<int VALUE>
	struct static_min<VALUE>
	{
		static const int value = VALUE;
	};

	template<int VALUE, int... VALUES>
	struct static_min<VALUE, VALUES...>
	{
		static const int next_value = static_min<VALUES...>::value;
		static const int value = VALUE < next_value ? VALUE : next_value;
	};

	template<int I, class... NCS>
	struct choose;

	template<int I, class NC, class... NCS>
	struct choose<I, NC, NCS...>
	{
		typedef typename choose<I - 1, NCS...>::type type;
	};
	template<class NC, class... NCS>
	struct choose<0, NC, NCS...>
	{
		typedef NC type;
	};


	template<bool COND>
	struct conditional_eval;

	template<>
	struct conditional_eval<true>
	{
		template<class F>
		DUAL_BUILD_FUNCTION static void eval(F f)
		{
			f();
		}
	};
	template<>
	struct conditional_eval<false>
	{
		template<class F>
		DUAL_BUILD_FUNCTION static void eval(F f)
		{
		}
	};

	template<template<int...> class CONSUMER, int V, int END, int STEP, bool DONE, int... VALUES>
	struct static_for_impl
	{
		using type = typename static_for_impl < CONSUMER, V+STEP, END, STEP, (V + STEP < END), VALUES..., V>::type;
	};
	template<template<int...> class CONSUMER, int V, int END, int STEP, int... VALUES>
	struct static_for_impl<CONSUMER, V, END, STEP, false, VALUES...>
	{
		using type = CONSUMER <VALUES...>;
	};

	template<template<int...> class CONSUMER, int END, int BEGIN = 0, int STEP = 1>
	struct static_for
	{
		using type = typename static_for_impl < CONSUMER, BEGIN, END, STEP, (BEGIN < END)>::type;
	};


	template<class...> 
	struct type_list { };

	template<template<class...> class APPLIER, class COMBLIST, class... TYPELISTS>
	struct apply_list_impl;
	template<template<class...> class APPLIER, class... DONETYPES, class... NEWTYPES, class... REMTYPELISTS>
	struct apply_list_impl<APPLIER, type_list<DONETYPES...>, type_list<NEWTYPES...>, REMTYPELISTS...>
	{
		using type = typename apply_list_impl<APPLIER, type_list<DONETYPES..., NEWTYPES...>, REMTYPELISTS...>::type;
	};
	template<template<class...> class APPLIER, class... DONETYPES>
	struct apply_list_impl<APPLIER, type_list<DONETYPES...>>
	{
		using type = APPLIER<DONETYPES...>;
	};
	template<template<class...> class APPLIER, class... TYPELISTS>
	struct apply_list
	{
		using type = typename apply_list_impl<APPLIER, type_list<>, TYPELISTS... >::type;
	};

	template<class INVERSE_LIST, class FORWARD_LIST>
	struct inverse_list_impl;
	template<class... INVERSE_TYPES, class FIRST, class... REMAINING>
	struct inverse_list_impl<type_list<INVERSE_TYPES...>, type_list<FIRST, REMAINING...>>
	{
		using type = typename inverse_list_impl<type_list<FIRST, INVERSE_TYPES...>, type_list<REMAINING...>>::type;
	};
	template<class INVERSE_LIST>
	struct inverse_list_impl<INVERSE_LIST, type_list<>>
	{
		using type = INVERSE_LIST;
	};
	template<class TYPELIST>
	struct inverse_list
	{
		using type = typename inverse_list_impl<type_list<>, TYPELIST>::type;
	};


	template<int... >
	struct sequence { };

	template<template<int...> class APPLIER, class SEQUENCE>
	struct apply_sequence;
	template<template<int...> class APPLIER, int... NUMS>
	struct apply_sequence<APPLIER, sequence<NUMS...>>
	{
		using type = APPLIER<NUMS...>;
	};

	template<unsigned MASK, bool TAKE, class TAKEN_SEQUENCE, class REM_SEQUENCE>
	struct select_from_impl;
	template<unsigned MASK, int... TAKEN, int NUM, int... NUMS>
	struct select_from_impl<MASK, true, sequence<TAKEN...>, sequence<NUM, NUMS...>>
	{
		using type = typename select_from_impl <(MASK >> 1U), MASK & 0x1, sequence<TAKEN..., NUM>, sequence<NUMS...> > ::type;
	};
	template<unsigned MASK, int... TAKEN, int NUM, int... NUMS>
	struct select_from_impl<MASK, false, sequence<TAKEN...>, sequence<NUM, NUMS...>>
	{
		using type = typename select_from_impl <(MASK >> 1U), MASK & 0x1, sequence<TAKEN...>, sequence<NUMS...> > ::type;
	};
	template<unsigned MASK, bool TAKE, int... TAKEN>
	struct select_from_impl<MASK, TAKE, sequence<TAKEN...>, sequence<>>
	{
		using type = sequence<TAKEN...>;
	};
	template<unsigned MASK, class SEQUENCE>
	struct select_from
	{
		using type = typename select_from_impl <(MASK >> 1U), MASK & 0x1, sequence<>, SEQUENCE > ::type;
	};
	

	template<template<int> class LOGICAL, class SEQUENCE>
	struct sequence_any;
	template<template<int> class LOGICAL, int NUM, int...NUMS>
	struct sequence_any<LOGICAL, sequence<NUM, NUMS...> >
	{
		static const bool value = LOGICAL<NUM>::value || sequence_any<LOGICAL, sequence<NUMS...>>::value;
	};
	template<template<int> class LOGICAL>
	struct sequence_any<LOGICAL, sequence<> >
	{
		static const bool value = false;
	};

	template<int A>
	struct static_is_zero
	{
		static const bool value = false;
	};
	template<>
	struct static_is_zero<0>
	{
		static const bool value = true;
	};


#endif //INCLUDED_HIS_META_UTILS
