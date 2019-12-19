
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

class Disk3DPageAdapter__pImpl;

class VMFOUNDATION_EXPORTS Disk3DPageAdapter : public AbstrMemoryCache
{
	VM_DECL_IMPL( Disk3DPageAdapter )

public:
	Disk3DPageAdapter( ::vm::IRefCnt *cnt, const std::string &fileName );
	const void *GetPage( size_t pageID ) override;
	size_t GetPageSize() const override;
	size_t GetPhysicalPageCount() const override;
	size_t GetVirtualPageCount() const override;

	int GetPadding() const;
	Size3 GetDataSizeWithoutPadding() const;
	Size3 Get3DPageSize() const;
	int Get3DPageSizeInLog() const;
	Size3 Get3DPageCount() const;

private:
	void *GetPageStorage_Implement( size_t pageID ) override { return nullptr; }
};


class Block3DCache__pImpl;
class VMFOUNDATION_EXPORTS Block3DCache : public AbstrMemoryCache
{
	VM_DECL_IMPL( Block3DCache )

	[[deprecated]] int blockCoordinateToBlockId( int xBlock, int yBlock, int zBlock ) const;
	void Create( I3DBlockFilePluginInterface * pageFile);
public:

	Block3DCache(IRefCnt *cnt, const std::string & fileName,std::function<Size3(I3DBlockFilePluginInterface*)> evaluator);
	Block3DCache( IRefCnt *cnt, const std::string &fileName );

	Block3DCache( IRefCnt *cnt, I3DBlockFilePluginInterface * pageFile, std::function<Size3( I3DBlockFilePluginInterface * )> evaluator );
	Block3DCache( IRefCnt *cnt, I3DBlockFilePluginInterface * pageFile );

	void SetDiskFileCache( I3DBlockFilePluginInterface *diskCache );

	Size3 CPUCacheBlockSize() const;
	vm::Size3 CPUCacheSize() const;

	[[deprecated]] int Padding() const;
	[[deprecated]] Size3 DataSizeWithoutPadding() const;
	[[deprecated]] Size3 BlockDim() const;
	[[deprecated]] Size3 BlockSize() const;

	Size3 CacheBlockDim() const;
	size_t GetPhysicalPageCount() const override { return CacheBlockDim().Prod(); }
	size_t GetVirtualPageCount() const override { return BlockDim().Prod(); }
	size_t GetPageSize() const override { return BlockSize().Prod() * sizeof( char ); }
	
	const void *GetPage( int xBlock, int yBlock, int zBlock ) { return AbstrMemoryCache::GetPage( blockCoordinateToBlockId( xBlock, yBlock, zBlock ) ); }
	const void *GetPage( const VirtualMemoryBlockIndex &index ) { return GetPage( index.x, index.y, index.z ); };

	virtual ~Block3DCache();
protected:
	[[deprecated]] int GetLog() const;
	void *GetPageStorage_Implement( size_t pageID ) override;
};
}  // namespace ysl

#endif /*_LARGEVOLUMECACHE_H_*/