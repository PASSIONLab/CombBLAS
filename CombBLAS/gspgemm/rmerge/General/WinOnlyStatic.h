#pragma once


//This is needed because of a MSVC compiler problem. 
//In MSVC, we must explicitly tell that a template specialization is static (internal linkage). 
//In gcc (and the C+++ standard) template specializations must not have a storage modifier (e.g. static) and instead get their storage class from the main template.
//Therefore you do it wrong for either gcc or MSVC...    ...or use WinOnlyStatic!

//This may result in another linker problem with explicit template specializatins with Cuda: LNK2005, multiply defined symbols. Solution is to declare them as inline which avoids symbol export.

#if defined(_MSC_VER)
#define WinOnlyStatic static
#elif defined(__GNUC__)
#define WinOnlyStatic
#else
#define WinOnlyStatic static
#endif

