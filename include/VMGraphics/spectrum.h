#ifndef SPECTRUM_H_
#define SPECTRUM_H_
#include <cassert>
#include <fstream>
#include <cmath>
#include <VMat/vmattype.h>

namespace vm
{
template <int nSamples>
class CoefficientSpectrum
{
public:
	Float c[ nSamples ];

	explicit CoefficientSpectrum( Float v = 0.0 )
	{
		for ( auto i = 0; i < nSamples; i++ )
			c[ i ] = v;
	}

	explicit CoefficientSpectrum( Float *v )
	{
		for ( auto i = 0; i < nSamples; i++ )
			c[ i ] = v[ i ];
	}

	bool IsNull() const
	{
		for ( auto i = 0; i < nSamples; i++ )
			if ( c[ i ] != 0.0f )
				return false;
		return true;
	}

	bool HasNaNs() const
	{
		for ( auto i = 0; i < nSamples; i++ )
			if ( std::isnan( c[ i ] ) )
				return true;
		return false;
	}

	CoefficientSpectrum &operator+=( const CoefficientSpectrum &spectrum )
	{
		for ( auto i = 0; i < nSamples; i++ )
			c[ i ] += spectrum.c[ i ];
		return *this;
	}

	CoefficientSpectrum &operator-=( const CoefficientSpectrum &spectrum )
	{
		for ( auto i = 0; i < nSamples; i++ )
			c[ i ] -= spectrum.c[ i ];
		return *this;
	}

	CoefficientSpectrum &operator*=( const CoefficientSpectrum &spectrum )
	{
		for ( auto i = 0; i < nSamples; i++ )
			c[ i ] *= spectrum.c[ i ];
		return *this;
	}

	CoefficientSpectrum &operator*=( Float s )
	{
		for ( auto i = 0; i < nSamples; i++ )
			c[ i ] *= s;
		return *this;
	}

	CoefficientSpectrum &operator/=( const CoefficientSpectrum &spectrum )
	{
		assert( !spectrum.HasNaNs() );
		for ( auto i = 0; i < nSamples; i++ )
			c[ i ] /= spectrum.c[ i ];
		return *this;
	}

	CoefficientSpectrum &operator/=( Float s )
	{
		Float inv = 1 / s;
		for ( auto i = 0; i < nSamples; i++ )
			c[ i ] *= inv;
		return *this;
	}

	Float operator[]( int i ) const
	{
		assert( i >= 0 && i < nSamples );
		return c[ i ];
	}

	Float &operator[]( int i )
	{
		assert( i >= 0 && i < nSamples );
		return c[ i ];
	}

	bool IsBlack() const
	{
		for ( auto i = 0; i < nSamples; i++ )
			if ( c[ i ] != 0.f )
				return false;
		return true;
	}

	void ToArray( Float *arr ) const
	{
		for ( auto i = 0; i < nSamples; i++ )
			*( arr + i ) = c[ i ];
	}
};

template <int nSamples>
CoefficientSpectrum<nSamples>
  operator+( const CoefficientSpectrum<nSamples> &s1, const CoefficientSpectrum<nSamples> &s2 )
{
	return CoefficientSpectrum<nSamples>( s1 ) += s2;  // More Effective C++: Item 22
}

template <int nSamples>
CoefficientSpectrum<nSamples>
  operator-( const CoefficientSpectrum<nSamples> &s1, const CoefficientSpectrum<nSamples> &s2 )
{
	return CoefficientSpectrum<nSamples>( s1 ) -= s2;  // More Effective C++: Item 22
}
template <int nSamples>
CoefficientSpectrum<nSamples>
  operator*( const CoefficientSpectrum<nSamples> &s1, const CoefficientSpectrum<nSamples> &s2 )
{
	return CoefficientSpectrum<nSamples>( s1 ) *= s2;  // More Effective C++: Item 22
}

template <int nSamples>
CoefficientSpectrum<nSamples>
  operator/( const CoefficientSpectrum<nSamples> &s1, const CoefficientSpectrum<nSamples> &s2 )
{
	return CoefficientSpectrum<nSamples>( s1 ) /= s2;  // More Effective C++: Item 22
}

template <int nSamples>
CoefficientSpectrum<nSamples>
  operator/( const CoefficientSpectrum<nSamples> &s1, Float s )
{
	return CoefficientSpectrum<nSamples>( s1 ) /= s;  // More Effective C++: Item 22
}

template <int nSamples>
CoefficientSpectrum<nSamples>
  operator*( const CoefficientSpectrum<nSamples> &s, Float v )
{
	auto r = s;
	for ( auto i = 0; i < nSamples; i++ ) r.c[ i ] *= v;
	return r;
}

template <int nSamples>
CoefficientSpectrum<nSamples>
  operator*( Float v, const CoefficientSpectrum<nSamples> &s )
{
	return s * v;
}

template <int nSamples>
std::ofstream &
  operator<<( std::ofstream &fs, const CoefficientSpectrum<nSamples> &coe )
{
	fs << "[";
	for ( auto i = 0; i < nSamples; i++ )
		fs << coe.c[ i ] << ", ";
	fs << "]";
	return fs;
}

template <int nSamples>
std::ifstream &
  operator>>( std::ifstream &fs, const CoefficientSpectrum<nSamples> &coe )
{
	for ( auto i = 0; i < nSamples; i++ )
		fs >> coe.c[ i ];
	return fs;
}

template <int nSamples>
std::ostream &
  operator<<( std::ostream &os, const CoefficientSpectrum<nSamples> &coe )
{
	os << "[";
	for ( auto i = 0; i < nSamples; i++ )
		os << coe.c[ i ] << ", ";
	os << "]";

	return os;
}

template <int nSamples>
std::istream &
  operator>>( std::istream &is, const CoefficientSpectrum<nSamples> &coe )
{
	for ( auto i = 0; i < nSamples; i++ )
		is >> coe.c[ i ];
	return is;
}

class SampledSpectrum : public CoefficientSpectrum<60>
{
public:
private:
};

//class RGBSpectrum: public CoefficientSpectrum<3>
//{
//public:
//	explicit RGBSpectrum(Float v = 0.f):CoefficientSpectrum<3>(v){}
//	explicit RGBSpectrum(const Float * rgb)
//	{
//		c[0] = rgb[0];
//		c[1] = rgb[1];
//		c[2] = rgb[2];
//	}
//	void ToRGB(Float * rgb)
//	{
//		rgb[0] = c[0];
//		rgb[1] = c[1];
//		rgb[2] = c[2];
//	}
//};

//class RGBASpectrum:public CoefficientSpectrum<4>
//{
//public:
//	explicit RGBASpectrum(Float v = 0.f):CoefficientSpectrum<4>(v){}
//	explicit RGBASpectrum(const Float * rgba)
//	{
//		c[0] = rgba[0];
//		c[1] = rgba[1];
//		c[2] = rgba[2];
//		c[3] = rgba[3];
//	}
//	void ToRGBA(Float * rgba)
//	{
//		rgba[0] = c[0];
//		rgba[1] = c[1];
//		rgba[2] = c[2];
//		rgba[3] = c[3];
//	}
//};

using RGBASpectrum = CoefficientSpectrum<4>;
using RGBSpectrum = CoefficientSpectrum<3>;

inline RGBSpectrum
  Lerp( Float t, const RGBSpectrum &s1, const RGBSpectrum &s2 )
{
	return s1 * ( 1 - t ) + s2 * t;
}

inline RGBASpectrum
  Lerp( Float t, const RGBASpectrum &s1, const RGBASpectrum &s2 )
{
	return ( 1 - t ) * s1 + t * s2;
}

enum class Color
{
	green,
	red,
	blue,
	white,
	black,
	yellow,
	transparent,
	gray
};

inline RGBASpectrum TranslateColor( vm::Color color )
{
	float c[ 4 ];
	switch ( color ) {
	case Color::green: ( c[ 0 ] = 0, c[ 1 ] = 1, c[ 2 ] = 0, c[ 3 ] = 1 ); break;
	case Color::red: ( c[ 0 ] = 1, c[ 1 ] = 0, c[ 2 ] = 0, c[ 3 ] = 1 ); break;
	case Color::blue: ( c[ 0 ] = 0, c[ 1 ] = 0, c[ 2 ] = 1, c[ 3 ] = 1 ); break;
	case Color::white: ( c[ 0 ] = 1, c[ 1 ] = 1, c[ 2 ] = 1, c[ 3 ] = 1 ); break;
	case Color::yellow: ( c[ 0 ] = 1, c[ 1 ] = 1, c[ 2 ] = 0, c[ 3 ] = 1 ); break;
	case Color::black: ( c[ 0 ] = 0, c[ 1 ] = 0, c[ 2 ] = 0, c[ 3 ] = 1 ); break;
	case Color::transparent: ( c[ 0 ] = 1, c[ 1 ] = 1, c[ 2 ] = 1, c[ 3 ] = 0 ); break;
	case Color::gray: ( c[ 0 ] = 0.5, c[ 1 ] = 0.5, c[ 2 ] = 0.5, c[ 3 ] = 0.5 ); break;
	}
	return RGBASpectrum( c );
}

}  // namespace ysl

#endif
