
#include <iostream>

#include <VMFoundation/rawreader.h>
#include <VMUtils/log.hpp>

int main()
{
	using namespace vm;

	RawReaderIO reader( R"(E:\Desktop\mixfrac.raw)", { 480, 720, 120 }, 1 );
	println( "Dimension: {}, VoxelSize: {}.", reader.GetDimension(), reader.GetElementSize() );

	Vec3i start{ -100, -100, -100 };

	Size3 size{ 580, 820, 220 };

	std::unique_ptr<unsigned char[]> buf( new unsigned char[ size.Prod() * reader.GetElementSize() ] );
	reader.readRegionNoBoundary( start, size, buf.get() );

	std::ofstream out( R"(E:\Desktop\output.raw)", std::ios::binary );

	out.write( (const char *)buf.get(), size.Prod() * reader.GetElementSize() );

	out.close();

	return 0;
}
