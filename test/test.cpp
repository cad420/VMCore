
#include <iostream>

#include <VMFoundation/rawreader.h>
#include <VMUtils/log.hpp>

int main()
{
	using namespace vm;

	RawReader reader( R"(E:\Desktop\mixfrac.raw)", { 480, 720, 120 }, 1 );
	println( "Dimension: {}, VoxelSize: {}.", reader.GetDimension(), reader.GetElementSize() );

	Vec3i start{ -50, -50, -50 };

	Size3 size{720, 720, 720};

	std::unique_ptr<unsigned char[]> buf( new unsigned char[ size.Prod() * reader.GetElementSize() ] );

	reader.readRegionNoBoundary( start, size, buf.get() );

	std::ofstream out( R"(E:\Desktop\output.raw)", std::ios::binary );

	out.write( (const char *)buf.get(), size.Prod() * reader.GetElementSize() );

	out.close();

	return 0;
}
