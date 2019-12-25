
#include <iostream>

#include <VMFoundation/blockedgridvolumefile.h>
#include <VMUtils/ref.hpp>
#include <VMUtils/vmnew.hpp>
#include <VMUtils/timer.hpp>
#include <VMFoundation/pluginloader.h>
#include "VMat/numeric.h"
#include "VMFoundation/largevolumecache.h"

int main()
{
	using namespace vm;

	std::string fileName = R"(E:\Desktop\mixfrac.raw)";

	auto file = PluginLoader::GetPluginLoader()->CreatePlugin<I3DBlockFilePluginInterface>( ".brv" );
	

	file->Open(R"(G:\mousehighres\lod4.brv)");
	{
		Timer::Scoped s( []( auto d ) {
			std::cout << d.s();
		} );
		
		const auto blockDim = file->Get3DPageCount();

		std::cout << blockDim << std::endl;
		for ( int i = 0; i < blockDim.Prod(); i++ ) {
			const auto buf = file->GetPage( i );
			//std::cout << vm::Dim( i, { blockDim.x, blockDim.y } ) << std::endl;
		}
	}

	return 0;
}
