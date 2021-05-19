#include <gtest/gtest.h>
#include <VMFoundation/rawstream.h>
#include <VMUtils/log.hpp>
#include <VMFoundation/dataarena.h>
#include <VMFoundation/lvdreader.h>
#include <VMFoundation/blockarray.h>
#include <random>
#include <fstream>

TEST(test_rawreader,basic)
{
	using namespace vm;

	RawStream reader( R"(mixfrac.raw)", { 480, 720, 120 }, 1 );
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
		auto  f = reader.ReadRegionNoBoundary( c.first,  c.second , buf);
		id++;
	}
}

TEST( test_lvdreader, write_back )
{
	using namespace vm;
	using namespace std;
	string fileName="sb__128_128_128.lvd";
	LVDReader reader( fileName );
	auto bsize = reader.SizeByBlock();
	auto size = reader.OriginalDataSize();
	auto blockSize = reader.BlockDataCount();
	Block3DArray<char, 6> data( size.x, size.y, size.z, nullptr );

	std::unique_ptr<char[]> bbuf( new char[ blockSize ] );

	// just test 64 block, read all from file
	ASSERT_EQ( 6, reader.BlockSizeInLog() );
	for (auto i = 0; i < bsize.Prod(); i++) {
		memcpy(bbuf.get(),reader.ReadBlock( i, 0 ), blockSize);
		data.SetBlockData( i, bbuf.get() );
	}

	std::default_random_engine e;
	std::uniform_int_distribution<int> u1( 0, bsize.Prod() - 1 );
	std::uniform_int_distribution<int> u2;

	// randomly pick a block to modify
	const auto n = 10;
	for (int i = 0; i < n; i++) {
		const auto ind = u1( e ); // block
		const auto val = u2( e ); // value

		memset( bbuf.get(), char(val), blockSize );

		data.SetBlockData( ind, bbuf.get() );

		reader.WriteBlock( bbuf.get(), ind, 0 );
		reader.Flush( ind ,0);
	}

	reader.Close();


	LVDReader reader2( fileName );

	// just test 64 block, read all from file
	for (auto i = 0; i < bsize.Prod(); i++) {
		auto d1 = data.BlockData( i );
		memcpy( bbuf.get(), reader2.ReadBlock( i, 0 ), blockSize );;
		for (int j = 0; j < blockSize; j++) {
			ASSERT_EQ( d1[ j ], bbuf[ j ] );
		}
	}
	
}
