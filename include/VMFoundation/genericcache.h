#pragma once

#include "VMCoreExtension/ipagefile.h"
#include "VMUtils/ieverything.hpp"
#include "blockarray.h"

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
class GenericBlockPageFileAdapter : public vm::Block3DArray<T, log>, public EverythingBase<IPageFile>
{
public:
	GenericBlockPageFileAdapter(IRefCnt* cnt, int xb, int yb, int zb, T *data ) :
	  Block3DArray<T, log>( xb * ( 1 << log ), yb * ( 1 << log ), zb * ( 1 << log ), data ),EverythingBase<IPageFile>(cnt) {}

	const void *GetPage( size_t pageID )override{ return reinterpret_cast<void *>( Block3DArray<T, log>::BlockData( pageID ) ); }

	void Flush() override{}

	void Write( const void *page, size_t pageID, bool flush )override{}

	void Flush( size_t pageID )override{}

	size_t GetPageSize()const override {return (1L<<log) * (1L<<log) * (1L<<log) * sizeof(T);}

	size_t GetPhysicalPageCount() const override{return Block3DArray<T,log>::BlockCount();}

	size_t GetVirtualPageCount() const override{return GetPhysicalPageCount();}
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
