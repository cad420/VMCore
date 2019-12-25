
#include <VMFoundation/reflector.h>
#include <VMUtils/vmnew.hpp>

namespace vm
{
std::vector<std::string> VMCorePluginFactory::Keys() const
{
	return { ".vifo" };
}

IEverything *VMCorePluginFactory::Create( const std::string &key )
{
	if ( key == ".vifo" ) {
		return VM_NEW<BlockedGridVolumeFile>();
	}
	return nullptr;
}
}


EXPORT_PLUGIN_FACTORY_IMPLEMENT( vm::VMCorePluginFactory )
