#include "common.h"

void GenerateABCFlow( int width, int height, int depth, const vm::Bound3i & sub, char *blockBuf )
{
	const size_t sideZ = width;
	const size_t sideY = height;
	const size_t sideX = depth;

	constexpr double Pi = 3.1415926535;
	constexpr auto minValue = 0.0031238;
	constexpr auto maxValue = 3.4641;
	const auto A = std::sqrt( 3 );
	const auto B = std::sqrt( 2 );
	const auto C = 1;
	for ( int z = sub.min.z; z < sub.max.z; z++ ) {
		for ( int y = sub.min.y; y < sub.max.y; y++ ) {
			for ( int x = sub.min.x; x < sub.max.x; x++ ) {
				const auto globalx = x;
				const auto globaly = y;
				const auto globalz = z;
				const auto index = globalx + globaly * sideX + globalz * sideX * sideY;
				const double X = globalx * 2 * Pi / sideX, Y = globaly * 2 * Pi / sideY, Z = globalz * 2 * Pi / sideZ;
				const auto value = std::sqrt( 6 + 2 * A * std::sin( Z ) * std::cos( Y ) + 2 * B * std::sin( Y ) * std::cos( X ) + 2 * std::sqrt( 6 ) * sin( X ) * std::cos( Z ) );
				blockBuf[ index ] = ( ( value - minValue ) / ( maxValue - minValue ) * 255 + 0.5 );
			}
		}
	}
}
