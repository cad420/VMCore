#include <iostream>
#include <gtest/gtest.h>
#include <VMUtils/timer.hpp>
#include <VMFoundation/pluginloader.h>
#include <VMFoundation/logger.h>
#include <VMCoreExtension/ifilemappingplugininterface.h>
#include <fstream>
#include <random>
using namespace vm;

std::string fileName = "D:\\data\\test_filemapping_basic.bin";

template<typename T>
std::vector<T> CreateTestFile( const std::string &fileName ,size_t count)
{
	std::ofstream file( fileName, std::ios::binary|std::ios::trunc| std::ios::out);
	if (file.is_open() == false) {
		perror( "Erro on open file: " );
	}
	const size_t size = sizeof(T) * count;

	std::default_random_engine e;
	std::uniform_int_distribution<T> u;
	std::vector<T> res;
	for (int i = 0; i < size; i++) {
		auto a = u( e );
		res.push_back( a );
		file.write( (const char*)&a, sizeof(T) );
	}
	file.close();
	return res;
}

TEST( test_filemapping, windows_basic )
{
	auto file = PluginLoader::GetPluginLoader()->CreatePlugin<IMappingFile>( "windows" );
	ASSERT_TRUE( file );

	size_t count = 256;
	const size_t offset = 0;

	using TestType = int;

	auto res = CreateTestFile<TestType>( fileName , count);
	const size_t len = sizeof(TestType) * res.size();
	

	file->Open( fileName.c_str(), len, FileAccess::ReadWrite, MapAccess::ReadWrite );
	auto ptr = (int*)file->MemoryMap( offset, len );

	ASSERT_TRUE( ptr );


	for (int i = 0; i < count; i++) {
		ASSERT_EQ( res[ i ], ptr[ i ] );
	}

	std::default_random_engine e;
	std::uniform_int_distribution<TestType> u;
	std::vector<TestType> v;
	for (int i = 0; i < count; i++) {
		auto a = u( e );
		v.push_back( a );
		ptr[ i ] = a;
	}

	file->Flush( ptr, len, 0 );

	file->Close();

	std::ifstream fs;
	fs.open( fileName.c_str(), std::ios::binary | std::ios::in);
	if (fs.is_open() == false) {
		perror( "Erro on open file: " );
		ASSERT_TRUE( false );
	}
	ASSERT_TRUE( fs.is_open() );

	for (int i = 0; i < count; i++) {
		int a;
		fs.read((char*)&a,sizeof(int));
		ASSERT_EQ( a, v[i] );
	}

}
TEST( test_filemapping, windows_create_file )
{
	std::string fileName = "create_mapping_file_test";
	auto file = PluginLoader::GetPluginLoader()->CreatePlugin<IMappingFile>( "windows" );
	ASSERT_TRUE( file );
	const auto fileSize = 1024 * 1024 * 1;
	// TODO:: delete old file before opening
	auto ok = file->Open( fileName.c_str(), fileSize, FileAccess::ReadWrite, MapAccess::ReadWrite );
	ASSERT_TRUE( ok );

	file->Close();

	std::ifstream infile( fileName );
	ASSERT_TRUE( infile.is_open() );

}
