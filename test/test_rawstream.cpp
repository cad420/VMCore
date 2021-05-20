#include <gtest/gtest.h>
#include <VMFoundation/rawstream.h>
#include <VMUtils/log.hpp>
#include <VMat/geometry.h>
#include <VMFoundation/dataarena.h>
#include <VMFoundation/lvdreader.h>
#include <VMFoundation/blockarray.h>
#include <random>
#include <fstream>

#include "common.h"

TEST( test_rawreader, basic )
{
	using namespace vm;
	using namespace std;

	const size_t elemSize = 1;
	const Size3 dataSize{ 480, 720, 120 };

	RawStream rawStream( R"(rawstream_testdata)", dataSize, elemSize );

	std::vector<std::tuple<Vec3i, Size3, char>> cases = {
		{ { -100, -100, -100 }, { 720, 720, 720 }, 0 },
		{ { 0, 0, 0 }, { 480, 720, 120 }, 1 },
		{ { 240, 360, 60 }, { 50, 50, 50 }, 2 },
		{ { 500, 750, 140 }, { 128, 128, 128 }, 3 },
		{ { -40, -40, -40 }, { 480, 720, 120 }, 4 }
	};

	std::vector<Bound3i> isectRect;
	const Bound3i dataBound{ Point3i{ 0, 0, 0 }, Vec3i{ dataSize }.ToPoint3() };

	std::for_each( cases.begin(), cases.end(), [ &dataBound, &isectRect ]( const auto &c ) {
		const auto &start = get<0>( c );
		const auto &size = get<1>( c );
		isectRect.push_back( dataBound.IntersectWidth( Bound3i{ start.ToPoint3(), start.ToPoint3() + Vec3i{ size }.ToPoint3() } ) );
	} );

	std::for_each( isectRect.begin(), isectRect.end(), []( const auto &i ) {
		std::cout << i.min << " " << i.max << std::endl;
	} );

	int id = 0;

	DataArena<64> arena( 1024 * 1024 * 50 );

	for ( int i = 0; i < cases.size(); i++ ) {
		const auto &c = cases[ i ];
		const auto &isect = isectRect[ i ];
		const auto &start = get<0>( c );
		const auto &size = get<1>( c );
		const auto &val = get<2>( c );
		auto buf = arena.Alloc<unsigned char>( size.Prod() * elemSize, true );
		memset( buf, val, size.Prod() );
		auto writeCount = rawStream.WriteRegionNoBoundary( start, size, buf );

		const auto readSize = isect.IsNull() ? Size3{0,0,0} : Size3{ isect.Diagonal() };
		auto buf2 = arena.Alloc<unsigned char>( readSize.Prod() * elemSize, true );
		auto readCount = rawStream.ReadRegion( isect.min.ToVector3(), readSize, buf2 );
		ASSERT_EQ( writeCount, readSize.Prod() );
		ASSERT_EQ( readCount, readSize.Prod() );
		for ( int i = 0; i < readSize.Prod(); i++ ) {
			ASSERT_EQ( buf2[ i ], val );
		}
	}
}

TEST( test_rawreader, write_back )
{
	using namespace vm;
	using namespace std;
	string fileName = "sb__128_128_128.lvd";
	LVDReader reader( fileName );
	auto bsize = reader.SizeByBlock();
	auto size = reader.OriginalDataSize();
	auto blockSize = reader.BlockDataCount();
	Block3DArray<char, 6> data( size.x, size.y, size.z, nullptr );

	std::unique_ptr<char[]> bbuf( new char[ blockSize ] );

	// just test 64 block, read all from file
	ASSERT_EQ( 6, reader.BlockSizeInLog() );
	for ( auto i = 0; i < bsize.Prod(); i++ ) {
		memcpy( bbuf.get(), reader.ReadBlock( i, 0 ), blockSize );
		data.SetBlockData( i, bbuf.get() );
	}

	std::default_random_engine e;
	std::uniform_int_distribution<int> u1( 0, bsize.Prod() - 1 );
	std::uniform_int_distribution<int> u2;

	// randomly pick a block to modify
	const auto n = 10;
	for ( int i = 0; i < n; i++ ) {
		const auto ind = u1( e );  // block
		const auto val = u2( e );  // value

		memset( bbuf.get(), char( val ), blockSize );

		data.SetBlockData( ind, bbuf.get() );

		reader.WriteBlock( bbuf.get(), ind, 0 );
		reader.Flush( ind, 0 );
	}

	reader.Close();

	LVDReader reader2( fileName );

	// just test 64 block, read all from file
	for ( auto i = 0; i < bsize.Prod(); i++ ) {
		auto d1 = data.BlockData( i );
		memcpy( bbuf.get(), reader2.ReadBlock( i, 0 ), blockSize );
		;
		for ( int j = 0; j < blockSize; j++ ) {
			ASSERT_EQ( d1[ j ], bbuf[ j ] );
		}
	}
}


TEST( test_rawreader, generate_abcflow )
{
	using namespace vm;
	using namespace std;
	const Size3 dataSize{ 256, 256, 256 };
	const size_t elemSize = 1;
	const std::string fileName = "abcflow)"+fmt("{}_{}_{}",dataSize.x,dataSize.y,dataSize.z) + ".raw";

	RawStream rs( fileName, dataSize, elemSize );


	std::vector<std::tuple<Vec3i, Size3, char>> cases = {
		{ { -100, -100, -100 }, { 720, 720, 720 }, 0 },
		{ { 0, 0, 0 }, { 480, 720, 120 }, 1 },
		{ { 240, 360, 60 }, { 50, 50, 50 }, 2 },
		{ { 500, 750, 140 }, { 128, 128, 128 }, 3 },
		{ { -40, -40, -40 }, { 480, 720, 120 }, 4 }
	};

	std::vector<Bound3i> isectRect;
	const Bound3i dataBound{ Point3i{ 0, 0, 0 }, Vec3i{ dataSize }.ToPoint3() };

	std::for_each( cases.begin(), cases.end(), [ &dataBound, &isectRect ]( const auto &c ) {
		const auto &start = get<0>( c );
		const auto &size = get<1>( c );
		isectRect.push_back( dataBound.IntersectWidth( Bound3i{ start.ToPoint3(), start.ToPoint3() + Vec3i{ size }.ToPoint3() } ) );
	} );

	DataArena<64> arena( 1024 * 1024 * 50 );
	for ( int i = 0; i < cases.size(); i++ ) {
		const auto &c = cases[ i ];
		const auto &isect = isectRect[ i ];
		const auto &start = get<0>( c );
		const auto &size = get<1>( c );
		const auto &val = get<2>( c );

		const auto readSize = isect.IsNull() ? Size3{0,0,0} : Size3{ isect.Diagonal() };
		auto buf = arena.Alloc<char>( readSize.Prod() * elemSize, true );

		const Bound3i dataBound{ isect.min,isect.min + Vec3i{ readSize }.ToPoint3() };
		GenerateABCFlow( dataSize.x, dataSize.y, dataSize.z, dataBound, buf );
		auto writeCount = rs.WriteRegionNoBoundary( dataBound.min.ToVector3(), readSize, (unsigned char*)buf );
	}

}
