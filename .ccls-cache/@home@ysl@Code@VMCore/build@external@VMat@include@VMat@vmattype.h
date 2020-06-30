#ifndef _VMAT_H_
#define _VMAT_H_

#include <limits>

namespace vm
{
using Float = float;
constexpr Float Pi = 3.14159265358979323846;
constexpr Float LOWEST_FLOAT = ( std::numeric_limits<Float>::lowest )();
constexpr Float MAX_VALUE = ( std::numeric_limits<Float>::max )();  // For fucking min/max macro defined in windows.h

template <typename T1, typename T2, bool>
struct WiderTypeImpl;

template <typename T1, typename T2>
struct WiderTypeImpl<T1, T2, true>
{
	using Type = T1;
};

template <typename T1, typename T2>
struct WiderTypeImpl<T1, T2, false>
{
	using Type = T2;
};

template <typename T1, typename T2>
using WiderType = typename WiderTypeImpl<T1, T2, ( sizeof( T1 ) > sizeof( T2 ) )>::Type;

template <typename T1, typename T2, bool>
struct PreferTypeImpl;

template <typename T1, typename T2>
struct PreferTypeImpl<T1, T2, false>
{
	using Type = WiderType<T1, T2>;
};

template <typename T1, typename T2>
struct PreferTypeImpl<T1, T2, true>
{
	using Type = Float;
};

template <typename T1, typename T2>
using PreferType = typename PreferTypeImpl<T1, T2, std::is_floating_point<T1>::value || std::is_floating_point<T2>::value>::Type;


}  // namespace vm

#endif /*_VMAT_H_*/
