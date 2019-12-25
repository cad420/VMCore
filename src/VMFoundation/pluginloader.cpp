
#include <VMFoundation/pluginloader.h>
#include <VMFoundation/libraryloader.h>

namespace vm
{
_Register__PluginFactory::_Register__PluginFactory( std::function<IPluginFactory *()> func )
{
	if ( func != nullptr ) {
		const auto iid = func()->GetIID();
		PluginLoader::GetPluginLoader()->factories[ iid ].push_back( func );
	}
}

PluginLoader *PluginLoader::GetPluginLoader()
{
	static PluginLoader pluginLoader;
	return &pluginLoader;
}

void PluginLoader::LoadPlugins( const std::string &directory )
{
	LibraryReposity::GetLibraryRepo()->AddLibraries( directory );
	for ( const auto &lib : LibraryReposity::GetLibraryRepo()->GetLibRepo() ) {
		void *sym = nullptr;
		if ( ( sym = lib.second->Symbol( "GetPluginFactoryInstance" ) ) != nullptr ) {
			const auto func = reinterpret_cast<FuncType>( sym );
			const auto iid = func()->GetIID();
			GetPluginLoader()->factories[ iid ].push_back( func );
		}
	}
}
}  // namespace vm
