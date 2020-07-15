
#include <iostream>
#include <gtest/gtest.h>
#include <VMUtils/timer.hpp>
#include <VMFoundation/pluginloader.h>
#include "VMFoundation/largevolumecache.h"

TEST(test_blockedgridvolumefile,basic)
{
	using namespace vm;

	std::string fileName = R"(E:\Desktop\mixfrac.raw)";

	auto file = PluginLoader::GetPluginLoader()->CreatePlugin<I3DBlockFilePluginInterface>( ".brv" );

	
	//file->Open(R"(G:\mousehighres\lod3.brv)");
	file->Open(R"(C:\temp\lod5.brv)");
	{
		Timer::Scoped s( []( auto d ) {
			std::cout << d.s();
		} );
		
		const auto blockDim = file->Get3DPageCount();

		srand( time( 0 ) );

		std::cout << blockDim << std::endl;
		for ( int i = 0; i < 200; i++ ) {
			const auto idx = rand() % blockDim.Prod();
			const auto buf = file->GetPage( idx );
			std::cout << i << std::endl;
			//std::cout << vm::Dim( i, { blockDim.x, blockDim.y } ) << std::endl;
		}
	}
}
