
#ifndef _PLUGINLOADER_H_
#define _PLUGINLOADER_H_

#include <unordered_map>
#include <functional>
#include <VMFoundation/plugin.h>
#include <VMFoundation/foundation_config.h>

namespace ysl
{
	class VMFOUNDATION_EXPORTS PluginLoader final
	{
		std::unordered_map<std::string, std::vector<std::function<IPluginFactory*()>>> factories;
	public:

		template<typename T> static 
		std::shared_ptr<T> CreatePlugin(const std::string & key)
		{
//			static_assert( std::is_base_of<IEverything, T>::value);
			
			const auto& f = PluginLoader::GetPluginLoader()->factories;
			auto iter = f.find(_iid_trait<T>::GetIID());
			if(iter == f.end())
			{
				return nullptr;
			}
			for(const auto & fptr:iter->second)
			{
				for(const auto & k:fptr()->Keys())
				{
					if(key == k)
					{
						//return Ref<T>(dynamic_cast<T>(fptr()->Create(key)));
						return Shared_Object_Dynamic_Cast<T>( fptr()->Create( key ) );
					}
				}
			}
			return nullptr;
		}
		
		static PluginLoader* GetPluginLoader();
		static void LoadPlugins(const std::string& directory);
	private:
		PluginLoader() = default;
	};

}
#endif