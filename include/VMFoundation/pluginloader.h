

#pragma once

#include <unordered_map>
#include <functional>
#include <VMCoreExtension/plugin.h>
#include <VMCoreExtension/plugindef.h>
#include <VMFoundation/foundation_config.h>

namespace vm
{

class VMFOUNDATION_EXPORTS _Register__PluginFactory
{
public:
	_Register__PluginFactory( std::function<IPluginFactory *()> func );
};

class VMFOUNDATION_EXPORTS PluginLoader final
{
	std::unordered_map<std::string, std::vector<std::function<IPluginFactory *()>>> factories;
	// std::vector should be replaced by std::set
	friend class _Register__PluginFactory;

public:
	template <typename T>
	static T *CreatePlugin( const char *key )
	{
		const auto &f = PluginLoader::GetPluginLoader()->factories;
		auto iter = f.find( _iid_trait<T>::GetIID() );
		if ( iter == f.end() ) {
			return nullptr;
		}
		for ( const auto &fptr : iter->second ) {
			const char **keys = nullptr;
			const auto count = fptr()->Keys( keys );
			for ( int i = 0; i < count; i++ ) {
				if ( std::strcmp( keys[ i ], key ) == 0 ) {
					return ( dynamic_cast<T *>( fptr()->Create( key ) ) );
				}
			}
		}
		return nullptr;
	}

	static PluginLoader *GetPluginLoader();
	static void LoadPlugins( const std::string &directory );

private:
	PluginLoader() = default;
};

#define VM_REGISTER_INTERNAL_PLUGIN_IMPL( pluginFactoryTypeName ) \
	static vm::_Register__PluginFactory _##pluginFactoryTypeName__RegisterHelper( GetHelper__##pluginFactoryTypeName );

}  // namespace vm
