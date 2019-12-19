
#ifndef _LARGEVOLUMECACHE_H_
#define _LARGEVOLUMECACHE_H_
#include "lvdreader.h"
#include <VMat/geometry.h>
#include <VMFoundation/virtualmemorymanager.h>
#include <VMFoundation/genericcache.h>
#include <VMUtils/ieverything.hpp>
#include <functional>

namespace vm
{
/**
	 * \brief This class is an adapter for the LVDReader.
	 */
class VMFOUNDATION_EXPORTS Disk3DPageAdapter : public AbstrMemoryCache
{
	LVDReader lvdReader;

public:
	Disk3DPageAdapter(::vm::IRefCnt *cnt, const std::string &fileName ) :
	  AbstrMemoryCache( cnt ),
	  lvdReader( fileName ) {}
	const void *GetPage( size_t pageID ) override { return lvdReader.ReadBlock( pageID ); }
	size_t GetPageSize() const override { return lvdReader.BlockSize(); }
	size_t GetPhysicalPageCount() const override { return lvdReader.BlockCount(); }
	size_t GetVirtualPageCount() const override { return lvdReader.BlockCount(); }

	int GetPadding() const { return lvdReader.GetBlockPadding(); }
	Size3 GetDataSizeWithoutPadding() const { return lvdReader.OriginalDataSize(); }
	Size3 Get3DPageSize() const
	{
		const std::size_t len = lvdReader.BlockSize();
		return Size3{ len, len, len };
	}
	int Get3DPageSizeInLog() const { return lvdReader.BlockSizeInLog(); }
	Size3 Get3DPageCount() const { return lvdReader.SizeByBlock(); }

private:
	void *GetPageStorage_Implement( size_t pageID ) override { return nullptr; }
};



class VMFOUNDATION_EXPORTS Block3DCache : public AbstrMemoryCache
{
	Size3 cacheDim;
	std::unique_ptr<IBlock3DArrayAdapter> m_volumeCache;

	Ref<I3DBlockFilePluginInterface> adapter;

	[[deprecated]] int blockCoordinateToBlockId( int xBlock, int yBlock, int zBlock ) const;
	
	void Create( I3DBlockFilePluginInterface * pageFile);
public:

	Block3DCache(::vm::IRefCnt *cnt, const std::string & fileName,std::function<Size3(I3DBlockFilePluginInterface*)> evaluator);
	Block3DCache( ::vm::IRefCnt *cnt, const std::string &fileName );

	Block3DCache( ::vm::IRefCnt *cnt, I3DBlockFilePluginInterface *pageFile, std::function<Size3( I3DBlockFilePluginInterface * )> evaluator );
	Block3DCache( ::vm::IRefCnt *cnt, I3DBlockFilePluginInterface *pageFile );

	void SetDiskFileCache( I3DBlockFilePluginInterface *diskCache );

	Size3 CPUCacheBlockSize() const;

	vm::Size3 CPUCacheSize() const;

	[[deprecated]] int Padding() const;
	[[deprecated]] Size3 DataSizeWithoutPadding() const;
	[[deprecated]] Size3 BlockDim() const;
	[[deprecated]] Size3 BlockSize() const;

	Size3 CacheBlockDim() const { return cacheDim; }
	size_t GetPhysicalPageCount() const override { return CacheBlockDim().Prod(); }
	size_t GetVirtualPageCount() const override { return BlockDim().Prod(); }
	size_t GetPageSize() const override { return BlockSize().Prod() * sizeof( char ); }
	
	const void *GetPage( int xBlock, int yBlock, int zBlock ) { return AbstrMemoryCache::GetPage( blockCoordinateToBlockId( xBlock, yBlock, zBlock ) ); }
	const void *GetPage( const VirtualMemoryBlockIndex &index ) { return GetPage( index.x, index.y, index.z ); };

protected:
	[[deprecated]] int GetLog() const;
	void *GetPageStorage_Implement( size_t pageID ) override;
};
}  // namespace ysl

#endif /*_LARGEVOLUMECACHE_H_*/