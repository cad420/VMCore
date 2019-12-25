
#ifndef _PLUGINDEF_H_
#define _PLUGINDEF_H_

#ifdef _WIN32
#define DLL_EXPORT __declspec(dllexport)
#else
#define DLL_EXPORT
#endif

#define DECLARE_PLUGIN_FACTORY( iid ) \
public:                               \
	std::string GetIID() const override { return iid; }


#define VM_REGISTER_PLUGIN_FACTORY_DECL(pluginFactoryTypeName)   \
	extern "C" ::vm::IPluginFactory *GetHelper__##pluginFactoryTypeName();

#define VM_REGISTER_PLUGIN_FACTORY_IMPL(pluginFactoryTypeName)   \
	::vm::IPluginFactory *GetHelper__##pluginFactoryTypeName()     \
	{                                                            \
	    static pluginFactoryTypeName factory;                    \
        return &factory;                                         \
	}

#define EXPORT_PLUGIN_FACTORY( pluginFactoryTypeName ) \
	extern "C" DLL_EXPORT ::vm::IPluginFactory *GetPluginFactoryInstance();

//#define EXPORT_PLUGIN_FACTORY_IMPLEMENT( pluginFactoryTypeName ) \
//	vm::IPluginFactory *GetPluginFactoryInstance()               \
//	{                                                            \
//		static pluginFactoryTypeName factory;                    \
//		return &factory;                                         \
//	}

#define EXPORT_PLUGIN_FACTORY_IMPLEMENT( pluginFactoryTypeName ) \
	::vm::IPluginFactory *GetPluginFactoryInstance()               \
	{                                                            \
	   return GetHelper__##pluginFactoryTypeName();              \
	}

#define DECLARE_PLUGIN_METADATA( pluginInterfaceTypeName, iid ) \
	template <>                                                 \
	struct _iid_trait<pluginInterfaceTypeName>                  \
	{                                                           \
		static const std::string GetIID() { return iid; }       \
	};

template <typename T>
struct _iid_trait;

#endif
