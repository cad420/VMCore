#include "VMCoreExtension/ipagefile.h"
#include "VMFoundation/blockarray.h"
#include "VMUtils/fmt.hpp"
#include "VMUtils/vmnew.hpp"
#include "VMat/geometry.h"
#include <VMFoundation/cachepolicy.h>
#include <cstring>
#include <gtest/gtest-death-test.h>
#include <gtest/gtest.h>
#include <VMFoundation/logger.h>
#include <VMFoundation/genericcache.h>

#include <VMFoundation/blockedgridvolumefile.h>
#include <VMFoundation/cachepolicy.h>
#include <VMFoundation/largevolumecache.h>

#include <VMFoundation/pluginloader.h>

#include <type_traits>

using namespace vm;
using namespace std;


/**
 * Volume data reading handler
 */
vector<Ref<Block3DCache>> SetupVolumeData(
  const vector<string> &fileNames,
  PluginLoader &pluginLoader,
  size_t availableHostMemoryHint )
{
	try {
		const auto lodCount = fileNames.size();
		vector<Ref<Block3DCache>> volumeData( lodCount );
		for ( int i = 0; i < lodCount; i++ ) {
			const auto cap = fileNames[ i ].substr( fileNames[ i ].find_last_of( '.' ) );
			auto p = pluginLoader.CreatePlugin<I3DBlockFilePluginInterface>( cap );
			if ( !p ) {
                LOG_CRITICAL<<fmt( "Failed to load plugin to read {} file", cap );
				exit( -1 );
			}
			p->Open( fileNames[ i ] );
			volumeData[ i ] = VM_NEW<Block3DCache>( p, [&availableHostMemoryHint]( I3DBlockFilePluginInterface *p ) {
				// this a
				const auto bytes = p->GetDataSizeWithoutPadding().Prod();
				size_t th = 2 * 1024 * 1024 * size_t( 1024 );  // 2GB as default
				if ( availableHostMemoryHint != 0 )
					th = availableHostMemoryHint;
				size_t d = 0;
				const auto pageSize = p->Get3DPageSize().Prod();
				if ( bytes < th ) {
					while ( d * d * d * pageSize < bytes ) d++;
				} else {
					while ( d * d * d * pageSize < th )
						d++;
				}
				return Size3{ d, d, d };
			} );
		}
		return volumeData;
	} catch ( std::runtime_error &e ) {
        LOG_CRITICAL<<e.what();
		return {};
	}
}

template<int nLogBlockSize, typename T, typename std::enable_if<sizeof(T)==1, char>::type = 0>
IPageFile * CreateTestBlock3DArray(const Size3 & blockDim, int padding){
  auto array = VM_NEW<GenericBlockPageFileAdapter<T, nLogBlockSize>>(blockDim.x, blockDim.y,blockDim.z,nullptr);
  const auto blockCount = blockDim.Prod();
  constexpr auto blockSize = (1L<<nLogBlockSize);
  const auto blockBytes = blockSize * blockSize * blockSize;
  std::unique_ptr<T[]> blockBuffer(new T[blockBytes]);
  for(int i = 0;i < blockCount;i++){
    std::memset(blockBuffer.get(), i % 256, blockBytes);
    array->SetBlockData(i, blockBuffer.get());
    std::cout<<blockBuffer[0]<<" "<<i<<std::endl;
  }
  return array;
}

/**
 * BRV file has a bad perfermance just for testing
 */
template<int nLogBlockSize, typename T,typename std::enable_if<sizeof(T)==1, char>::type = 0>
void CreateBRVFile(const std::string & fileName, const Block3DArray<T, nLogBlockSize>& data){

}

TEST( test_cachepolicy, listbasedlrucachepolicy )
{
  auto & pluginLoader = *PluginLoader::GetPluginLoader();
  auto policy = VM_NEW<ListBasedLRUCachePolicy>();
  auto cache = VM_NEW<BlockedGridVolumeFile>();
  auto data = CreateTestBlock3DArray<5, char>({2,2,2}, 0);
  for(int i = 0;i<8;i++){
    LOG_INFO<<((int8_t*)data->GetPage(i))[0];
  }
}
