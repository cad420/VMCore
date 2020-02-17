
#pragma once
#include <memory>


#include <VMat/geometry.h>

namespace vm
{
	
class Film
{
public:
	const Vector2i Resolution;
	struct Pixel
	{
		unsigned char v[ 4 ];
	};
	std::unique_ptr<Pixel[]> pixel;
	Film( const Vector2i &resolution );
};

}  // namespace vm
