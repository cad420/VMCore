
#include <VMGraphics/film.h>

namespace vm
{
Film::Film( const Vector2i &resolution ) :
  resolution( resolution )
{
	std::size_t c = std::size_t( resolution.x ) * resolution.y;
	pixel.reset( new Pixel[ c ] );
}
}  // namespace ysl
