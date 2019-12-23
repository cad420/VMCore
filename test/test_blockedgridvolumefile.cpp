
#include <iostream>

#include <VMFoundation/blockedgridvolumefile.h>
#include <VMUtils/ref.hpp>
#include <VMUtils/vmnew.hpp>
#include <VMUtils/timer.hpp>
#include "VMat/numeric.h"

int main()
{
	using namespace vm;

	std::string fileName = R"(E:\Desktop\mixfrac.raw)";

	Ref<BlockedGridVolumeFile> file = VM_NEW<BlockedGridVolumeFile>( fileName, Size3{ 480, 720, 120 }, 1, 6, 2 );

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
