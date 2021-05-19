
#include <iostream>
#include <gtest/gtest.h>
#include <VMUtils/timer.hpp>
#include <VMFoundation/pluginloader.h>
#include "VMCoreExtension/i3dblockfileplugininterface.h"
#include "VMFoundation/largevolumecache.h"

TEST(test_blockedgridvolumefile,basic)
{
	using namespace vm;
	Ref<I3DBlockFilePluginInterface> file = PluginLoader::GetPluginLoader()->CreatePlugin<I3DBlockFilePluginInterface>( ".brv" );

}
