#pragma once

#include "VMCoreExtension/ipagefile.h"
#include "VMUtils/ieverything.hpp"
#include <VMCoreExtension/i3dblockfileplugininterface.h>
#include "blockarray.h"
#include <VMFoundation/logger.h>

namespace vm
{
/**
	 * \brief This is used to eliminate template parameters of \a Block3DArray
	 */
struct IBlock3DArrayAdapter
{
	virtual void *GetBlockData( size_t blockID ) = 0;
	virtual void *GetRawData() = 0;
	virtual ~IBlock3DArrayAdapter() = default;
};

template <typename T, int log>
class GenericBlockCache : public vm::Block3DArray<T, log>,
  public IBlock3DArrayAdapter
{
public:
	//GenericBlockCache( int w, int h, int d, T *data ) :
	// Block3DArray<T, log>( w, h, d, data ) {}
	GenericBlockCache( int xb, int yb, int zb, T *data ) :
	  Block3DArray<T, log>( xb * ( 1 << log ), yb * ( 1 << log ), zb * ( 1 << log ), data ) {}
	void *GetBlockData( size_t blockID ) override { return reinterpret_cast<void *>( Block3DArray<T, log>::BlockData( blockID ) ); }

	void *GetRawData() override { return Block3DArray<T, log>::Data(); }

};

template <typename T, int log>
class GenericBlockPageFileAdapter final: public vm::Block3DArray<T, log>, public EverythingBase<I3DBlockDataInterface>
{
  int m_padding = 0;
public:
	GenericBlockPageFileAdapter(IRefCnt* cnt, int xb, int yb, int zb, int padding, T *data) :m_padding(padding),
	  Block3DArray<T, log>( xb * ( 1 << log ), yb * ( 1 << log ), zb * ( 1 << log ), data ),EverythingBase<I3DBlockDataInterface>(cnt) {}

	const void *GetPage( size_t pageID )override{ return reinterpret_cast<void *>( Block3DArray<T, log>::BlockData( pageID ) ); }

	void Flush() override{}

	void UnlockPage(size_t pageID){}

	void Write( const void *page, size_t pageID, bool flush )override
	{
		Block3DArray<T, log>::SetBlockData( pageID, (const T*)page );
	}

	void Flush( size_t pageID ) override {  }

	size_t GetPageSize()const override {return (1L<<log) * (1L<<log) * (1L<<log) * sizeof(T);}

	size_t GetPhysicalPageCount() const override{return Block3DArray<T,log>::BlockCount();}

	size_t GetVirtualPageCount() const override{return GetPhysicalPageCount();}

	int GetPadding() const override { return m_padding; }
	Size3 GetDataSizeWithoutPadding() const override
	{
		const auto pageSize = Get3DPageSize();
		const size_t padding = GetPadding();
		return pageSize-Size3{padding,padding,padding};
	}
	Size3 Get3DPageSize() const override
	{
	  const auto s = 1ULL<<Get3DPageSizeInLog();
		return { s, s, s };
	}
	int Get3DPageSizeInLog() const override { return static_cast<int>(log); }
	Size3 Get3DPageCount() const override
	{
		return Size3{ (size_t)Block3DArray<T,log>::BlockWidth(), (size_t)Block3DArray<T,log>::BlockHeight(), (size_t)Block3DArray<T,log>::BlockDepth()};
	}
};


template <typename T>
using GenericBlock16Cache = GenericBlockCache<T, 4>;
template <typename T>
using GenericBlock32Cache = GenericBlockCache<T, 5>;
template <typename T>
using GenericBlock64Cache = GenericBlockCache<T, 6>;
template <typename T>
using GenericBlock128Cache = GenericBlockCache<T, 7>;
template <typename T>
using GenericBlock256Cache = GenericBlockCache<T, 8>;
template <typename T>
using GenericBlock512Cache = GenericBlockCache<T, 9>;
template <typename T>
using GenericBlock1024Cache = GenericBlockCache<T, 10>;

using Int8Block16Cache = GenericBlock16Cache<char>;
using Int8Block32Cache = GenericBlock32Cache<char>;
using Int8Block64Cache = GenericBlock64Cache<char>;
using Int8Block128Cache = GenericBlock128Cache<char>;
using Int8Block256Cache = GenericBlock256Cache<char>;
using Int8Block512Cache = GenericBlock512Cache<char>;
using Int8Block1024Cache = GenericBlock1024Cache<char>;
}  // namespace vm
