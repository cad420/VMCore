#include <gtest/gtest-death-test.h>
#include <gtest/gtest.h>
#include "VMCoreExtension/i3dblockfileplugininterface.h"
#include "VMCoreExtension/ipagefile.h"
#include "VMFoundation/blockarray.h"
#include "VMUtils/fmt.hpp"
#include "VMUtils/vmnew.hpp"
#include "VMat/geometry.h"
#include <VMFoundation/cachepolicy.h>
#include <cstring>
#include <VMFoundation/logger.h>
#include <VMFoundation/genericcache.h>
#include <VMFoundation/blockedgridvolumefile.h>
#include <VMFoundation/cachepolicy.h>
#include <VMFoundation/largevolumecache.h>
#include <VMFoundation/pluginloader.h>
#include <type_traits>
#include <random>

using namespace vm;
using namespace std;

int custom_log_level = 10;

/**
 * Volume data reading handler
 */

template <int nLogBlockSize, typename T, typename std::enable_if<sizeof( T ) == 1, char>::type = 0>
I3DBlockDataInterface *CreateTestBlock3DArray( const Size3 &blockDim, int padding )
{
	auto array = VM_NEW<GenericBlockPageFileAdapter<T, nLogBlockSize>>( blockDim.x, blockDim.y, blockDim.z, padding, nullptr );
	const auto blockCount = blockDim.Prod();
	constexpr auto blockSize = ( 1L << nLogBlockSize );
	const auto blockBytes = blockSize * blockSize * blockSize;
	std::unique_ptr<T[]> blockBuffer( new T[ blockBytes ] );
	for ( int i = 0; i < blockCount; i++ ) {
		std::memset( blockBuffer.get(), i % 256, blockBytes );
		array->SetBlockData( i, blockBuffer.get() );
	}
	return array;
}

/**
 * BRV file has a bad perfermance just for testing
 */
template <int nLogBlockSize, typename T, typename std::enable_if<sizeof( T ) == 1, char>::type = 0>
void CreateBRVFile( const std::string &fileName, const Block3DArray<T, nLogBlockSize> &data )
{
}

TEST( test_cachepolicy, listbasedlrucachepolicy_read_basic )
{
	Logger::SetLogLevel( LogLevel( custom_log_level - 1 ) );
	auto &pluginLoader = *PluginLoader::GetPluginLoader();
	const Size3 psize{ 8, 1, 1 };
	const Size3 vsize{ 4, 4, 4 };
	auto data = CreateTestBlock3DArray<5, char>( vsize, 0 );
	Ref<Block3DCache> cache = VM_NEW<Block3DCache>( data, [ &psize ]( I3DBlockDataInterface *data ) { return psize; } );

	ASSERT_EQ( cache->GetPhysicalPageCount(), psize.Prod() );
	ASSERT_EQ( cache->GetVirtualPageCount(), vsize.Prod() );

	LOG_INFO << fmt( "Physical cache block count: {}, virtual cache block count: {}", cache->GetPhysicalPageCount(), cache->GetVirtualPageCount() );

	for ( int i = 0; i < vsize.Prod(); i++ ) {
		VirtualMemoryBlockIndex index{ size_t( i ), (int)vsize.x, (int)vsize.y, (int)vsize.z };
		auto p = (const char *)cache->GetPage( index );
		ASSERT_EQ( i, (int)p[ 0 ] );
	}

	std::default_random_engine e;
	std::uniform_int_distribution<int> u( 0, vsize.Prod() - 1 );

	std::unique_ptr<char[]> page( new char[ data->GetPageSize() ] );

	const auto naccess = 10;
	const auto nwrtie = 10;

	for ( int i = 0; i < nwrtie; i++ ) {
		for ( int j = 0; j < naccess; j++ ) {
			// random access cache 10 times
			size_t rnd = u( e );
			VirtualMemoryBlockIndex index{ rnd, (int)vsize.x, (int)vsize.y, (int)vsize.z };
			auto val = (const char *)cache->GetPage( index );
			ASSERT_EQ( val[ 0 ], rnd );
		}
	}
}

TEST( test_cachepolicy, listbasedlrucachepolicy_write_through )
{
	Logger::SetLogLevel( LogLevel( custom_log_level - 1 ) );
	auto &pluginLoader = *PluginLoader::GetPluginLoader();
	const Size3 psize{ 8, 1, 1 };
	const Size3 vsize{ 4, 4, 4 };
	auto data = CreateTestBlock3DArray<5, char>( vsize, 0 );
	Ref<Block3DCache> cache = VM_NEW<Block3DCache>( data, [ &psize ]( I3DBlockDataInterface *data ) { return psize; } );

	ASSERT_EQ( cache->GetPhysicalPageCount(), psize.Prod() );
	ASSERT_EQ( cache->GetVirtualPageCount(), vsize.Prod() );

	LOG_INFO << fmt( "Physical cache block count: {}, virtual cache block count: {}", cache->GetPhysicalPageCount(), cache->GetVirtualPageCount() );

	for ( int i = 0; i < vsize.Prod(); i++ ) {
		VirtualMemoryBlockIndex index{ size_t( i ), (int)vsize.x, (int)vsize.y, (int)vsize.z };
		auto p = (const char *)cache->GetPage( index );
		ASSERT_EQ( i, (int)p[ 0 ] );
	}

	std::default_random_engine e;
	std::uniform_int_distribution<int> u( 0, vsize.Prod() - 1 );

	std::unique_ptr<char[]> page( new char[ data->GetPageSize() ] );

	const auto naccess = 10;
	const auto nwrtie = 10;

	for ( int i = 0; i < nwrtie; i++ ) {
		for ( int j = 0; j < naccess; j++ ) {
			// random access cache 10 times
			size_t rnd = u( e );
			VirtualMemoryBlockIndex index{ rnd, (int)vsize.x, (int)vsize.y, (int)vsize.z };
			cache->GetPage( index );
			LOG_CUSTOM( custom_log_level ) << "Access " << j << "th of " << naccess;
		}
		// write cache
		size_t write_addr = u( e );
		char write_val = u( e );

		std::memset( page.get(), write_val, data->GetPageSize() );
		VirtualMemoryBlockIndex write_index{ write_addr, (int)vsize.x, (int)vsize.y, (int)vsize.z };
		// write through
		LOG_CUSTOM( custom_log_level ) << "Writting: " << i << " of " << nwrtie;

		LOG_CUSTOM( custom_log_level ) << fmt( "Writing value: {}, va: {}", int( write_val ), write_addr );
		cache->Write( page.get(), write_addr, true );
		auto cache_value = (const char *)cache->GetPage( write_index );
		auto data_value = (const char *)data->GetPage( write_addr );
		ASSERT_EQ( (int)cache_value[ 0 ], (int)data_value[ 0 ] );
	}
}

TEST( test_cachepolicy, listbasedlrucachepolicy_write_back )
{
	auto &pluginLoader = *PluginLoader::GetPluginLoader();
	const Size3 psize{ 8, 1, 1 };
	const Size3 vsize{ 4, 4, 4 };
	auto data = CreateTestBlock3DArray<5, char>( vsize, 0 );
	Ref<Block3DCache> cache = VM_NEW<Block3DCache>( data, [ &psize ]( I3DBlockDataInterface *data ) { return psize; } );

	ASSERT_EQ( cache->GetPhysicalPageCount(), psize.Prod() );
	ASSERT_EQ( cache->GetVirtualPageCount(), vsize.Prod() );

	LOG_INFO << fmt( "Physical cache block count: {}, virtual cache block count: {}", cache->GetPhysicalPageCount(), cache->GetVirtualPageCount() );

	for ( int i = 0; i < vsize.Prod(); i++ ) {
		VirtualMemoryBlockIndex index{ size_t( i ), (int)vsize.x, (int)vsize.y, (int)vsize.z };
		auto p = (const char *)cache->GetPage( index );
		ASSERT_EQ( i, (int)p[ 0 ] );
	}

	std::default_random_engine e;
	std::uniform_int_distribution<int> u( 0, vsize.Prod() - 1 );

	std::unique_ptr<char[]> page( new char[ data->GetPageSize() ] );

	const auto naccess = 100;
	const auto nwrtie = 100;

	for ( int i = 0; i < nwrtie; i++ ) {
		for ( int j = 0; j < naccess; j++ ) {
			// random access cache 10 times
			size_t rnd = u( e );
			VirtualMemoryBlockIndex index{ rnd, (int)vsize.x, (int)vsize.y, (int)vsize.z };
			cache->GetPage( index );
			LOG_CUSTOM( custom_log_level ) << "Access " << j << "th of " << naccess;
		}
		// write cache
		size_t write_addr = u( e );
		char write_val = u( e );

		std::memset( page.get(), write_val, data->GetPageSize() );
		VirtualMemoryBlockIndex write_index{ write_addr, (int)vsize.x, (int)vsize.y, (int)vsize.z };
		// write through
		LOG_CUSTOM( custom_log_level ) << "Writting: " << i << " of " << nwrtie;
		cache->Write( page.get(), write_addr, false );

		LOG_CUSTOM( custom_log_level ) << fmt( "Writing value: {}, va: {}", int( write_val ), write_addr );
		cache->Flush( write_addr );
	}

	cache->Flush();
	for ( int i = 0; i < vsize.Prod(); i++ ) {
		size_t write_addr = i;
		VirtualMemoryBlockIndex write_index{ write_addr, (int)vsize.x, (int)vsize.y, (int)vsize.z };

		auto cache_value = (const char *)cache->GetPage( write_index );
		auto data_value = (const char *)data->GetPage( write_addr );

		ASSERT_EQ( (int)cache_value[ 0 ], (int)data_value[ 0 ] );
	}
}

TEST( test, writebackondist )
{

}
