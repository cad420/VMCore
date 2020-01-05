
#include <VMFoundation/memorypool.h>
#include <VMFoundation/memoryallocationtracker.h>
#include <mutex>

namespace vm
{
class MemoryPool__pImpl
{
	VM_DECL_API( MemoryPool )
public:
	MemoryPool__pImpl( MemoryPool *api, size_t maxSize ) :
	  q_ptr( api ),
	  tracker( maxSize ),ptr(new unsigned char[maxSize]) {}
	MemoryAllocationTracker tracker;
	std::unique_ptr<unsigned char[]> ptr;
};

MemoryPool::MemoryPool( size_t maxSize ):d_ptr( new MemoryPool__pImpl(this,maxSize) )
{
	
}

void * MemoryPool::Alloc( size_t size )
{
	VM_IMPL( MemoryPool )
	const auto pos = _->tracker.Allocate( size + sizeof( Allocation ), alignof( std::max_align_t ) );
	if ( !pos.IsValid() ) 
		return nullptr;
	auto &posAlloc = *( reinterpret_cast<Allocation *>( _->ptr.get() + pos.UnalignedOffset ) );
	posAlloc = pos;
	return _->ptr.get() + sizeof(Allocation);
}

void MemoryPool::Free( void *ptr )
{
	VM_IMPL( MemoryPool )
	auto &posAlloc = *( reinterpret_cast<Allocation *>( (uint8_t*)ptr - sizeof( Allocation ) ) );
	_->tracker.Free(std::move(posAlloc));
}

bool MemoryPool::IsFull() const
{
	const auto _ = d_func();
	return _->tracker.IsFull();
}

bool MemoryPool::IsEmpty() const
{
	const auto _ = d_func();
	return _->tracker.IsEmpty();
}

size_t MemoryPool::GetMaxSize() const
{
	const auto _ = d_func();
	return _->tracker.GetMaxSize();
}

size_t MemoryPool::GetFreeSize() const
{
	const auto _ = d_func();
	return _->tracker.GetFreeSize();
}

size_t MemoryPool::GetUsedSize() const
{
	const auto _ = d_func();
	return _->tracker.GetUsedSize();
}

MemoryPool::~MemoryPool()
{
}





class SafeMemoryPool__pImpl
{
	VM_DECL_API( SafeMemoryPool )
public:
	SafeMemoryPool__pImpl( SafeMemoryPool *api, size_t maxSize ) :
	  q_ptr( api ),
	  tracker( maxSize ) {}
    MemoryAllocationTracker tracker;
	std::unique_ptr<unsigned char[]> ptr;
	std::mutex ptrMut;
};

SafeMemoryPool::SafeMemoryPool( size_t maxSize ):d_ptr( new SafeMemoryPool__pImpl(this,maxSize) )
{
}

void * SafeMemoryPool::SafeAlloc( size_t size )
{
	VM_IMPL( SafeMemoryPool )

	_->ptrMut.lock();
	const auto pos = _->tracker.Allocate( size + sizeof( Allocation ), alignof( std::max_align_t ) );
	_->ptrMut.unlock();

	if ( !pos.IsValid() )
		return nullptr;
	auto &posAlloc = *( reinterpret_cast<Allocation *>( _->ptr.get() + pos.UnalignedOffset ) );
	posAlloc = pos;
	return _->ptr.get() + sizeof( Allocation );
}

void SafeMemoryPool::SafeFree( void *ptr )
{
	VM_IMPL( SafeMemoryPool )
	auto &posAlloc = *( reinterpret_cast<Allocation *>( (uint8_t*)(ptr) - sizeof( Allocation ) ) );

	std::lock_guard<std::mutex> lk( _->ptrMut );
	_->tracker.Free( std::move( posAlloc ) );
}

bool SafeMemoryPool::IsFull() const
{
	const auto _ = d_func();
	return _->tracker.IsFull();
}

bool SafeMemoryPool::IsEmpty() const
{
	const auto _ = d_func();
	return _->tracker.IsEmpty();
}

size_t SafeMemoryPool::GetMaxSize() const
{
	
	const auto _ = d_func();
	return _->tracker.GetMaxSize();
}

size_t SafeMemoryPool::GetFreeSize() const
{
	const auto _ = d_func();
	return _->tracker.GetFreeSize();
}

size_t SafeMemoryPool::GetUsedSize() const
{
	const auto _ = d_func();
	return _->tracker.GetUsedSize();
}

SafeMemoryPool::~SafeMemoryPool()
{
}


}  // namespace vm
