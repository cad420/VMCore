
#ifndef _PLUGINLOADER_H_
#define _PLUGINLOADER_H_

#include <unordered_map>
#include <functional>
#include <VMCoreExtension/plugin.h>
#include <VMCoreExtension/plugindef.h>
#include <VMFoundation/foundation_config.h>
#include <VMUtils/ref.hpp>
#include <VMUtils/ieverything.hpp>

namespace vm
{
class VMFOUNDATION_EXPORTS PluginLoader final
{
	std::unordered_map<std::string, std::vector<std::function<IPluginFactory *()>>> factories;

public:
	template <typename T>
	static T *CreatePlugin( const std::string &key )
	{
		const auto &f = PluginLoader::GetPluginLoader()->factories;
		auto iter = f.find( _iid_trait<T>::GetIID() );
		if ( iter == f.end() ) {
			return nullptr;
		}
		for ( const auto &fptr : iter->second ) {
			for ( const auto &k : fptr()->Keys() ) {
				if ( key == k ) {
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

}  // namespace ysl
#endif
