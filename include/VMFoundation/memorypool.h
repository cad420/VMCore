
#pragma once
#include <VMUtils/common.h>
#include "VMUtils/concepts.hpp"

namespace vm
{
class MemoryPool__pImpl;

class MemoryPool:NoCopy
{
	VM_DECL_IMPL( MemoryPool )
public:
	MemoryPool( size_t maxSize );

	void *Alloc(size_t size);
	void *AlignAlloc( size_t size, size_t align );
	void Free( void *ptr );

	bool IsFull() const;
	bool IsEmpty() const;

	size_t GetMaxSize() const;
	size_t GetFreeSize() const;
	size_t GetUsedSize() const;
	
	~MemoryPool();

	template<typename T>
	T *Alloc( size_t n )
	{
		T *p = Alloc(sizeof(T) * n);
		for ( size_t i = 0; i < n; i++ ) 
		{
			new (p + i * sizeof(T)) T;
		}
		return p;
	}

	template<typename T, typename ...Agrs>
	T *Alloc( size_t n, Agrs &&... agrs)
	{
		T *p = Alloc( sizeof( T ) * n );
		for ( size_t i = 0; i < n; i++ ) {
			new ( p + i * sizeof( T ) ) T(std::forward<Agrs>( agrs )...);
		}
		return p;
	}

	template<typename T>
	void Free( T * ptr,size_t n )
	{
		for ( size_t i = 0; i < n; i++ ) {
			reinterpret_cast<T *>( ptr + i * sizeof( T ) )->~T();
		}
		Free( ptr );
	}
	
};


class SafeMemoryPool__pImpl;

class SafeMemoryPool:NoCopy
{
	VM_DECL_IMPL( SafeMemoryPool )
public:

	
	SafeMemoryPool( size_t maxSize );

	void * SafeAlloc( size_t size );
	void SafeFree( void *ptr );

	bool IsFull() const;
	bool IsEmpty() const;

	size_t GetMaxSize() const;
	size_t GetFreeSize() const;
	size_t GetUsedSize() const;

	~SafeMemoryPool();

	template <typename T>
	T *SafeAlloc( size_t n )
	{
		T *p = SafeAlloc( sizeof( T ) * n );
		for ( size_t i = 0; i < n; i++ ) {
			new ( p + i * sizeof( T ) ) T;
		}
		return p;
	}

	template <typename T,typename ...Args>
	T *SafeAlloc( size_t n, Args&&... args )
	{
		T *p = SafeAlloc( sizeof( T ) * n );
		for ( size_t i = 0; i < n; i++ ) {
			new ( p + i * sizeof( T ) ) T(std::forward<Args>( args )...);
		}
		return p;
	}

	template<typename T>
	void SafeFree( T * ptr, size_t n)
	{
		for ( size_t i = 0; i < n; i++ ) {
			reinterpret_cast<T *>( ptr + i * sizeof( T ) )->~T();
		}
		SafeFree( ptr );
	}
};

}  // namespace vm