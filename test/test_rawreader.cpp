

#include <gtest/gtest.h>
#include <VMFoundation/rawreader.h>
#include <VMUtils/log.hpp>
#include <VMFoundation/dataarena.h>
#include <fstream>


TEST(test_rawreader,basic)
{
	using namespace vm;

	RawReader reader( R"(E:\Desktop\mixfrac.raw)", { 480, 720, 120 }, 1 );
	println( "Dimension: {}, VoxelSize: {}.", reader.GetDimension(), reader.GetElementSize() );
	//std::unique_ptr<unsigned char[]> buf( new unsigned char[ Size3( 480, 720, 120 ).Prod() * reader.GetElementSize() ] );

	std::vector<std::pair<Vec3i, Size3>> cases = {
		{ { -100, -100, -100 }, { 720, 720, 720 } },
		{ { 0, 0, 0 }, { 480, 720, 120 } },
		{ { 240, 360, 60 }, { 50, 50, 50 } },
		{ { 500, 750, 140 }, { 128, 128, 128 } },
		{ { -40, -40, -40 }, { 480, 720, 120 } }
	};

	int id = 0;

	DataArena<64> arena;

	for ( const auto &c : cases ) {
		auto buf = arena.Alloc<unsigned char>( c.second.Prod(), true );

		auto  f = reader.asyncReadRegionNoBoundary( c.first, Vec3i( c.second ), buf, [buf,id,c]() {
			std::cout << "read finished\n";
			std::string sid = std::to_string( id );
			std::ofstream out( R"(E:\Desktop\output)" + sid + ".raw", std::ios::binary );
			out.write( (const char *)buf, c.second.Prod() );
			out.close();
		} );
		while ( f.wait_for( std::chrono::seconds(1) ) != std::future_status::ready ) {
			std::cout << "not ready\n";
		}
		id++;
	}
}
