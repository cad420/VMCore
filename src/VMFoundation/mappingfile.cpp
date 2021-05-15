
//#include <VMUtils/vmnew.hpp>
#define _HAS_STD_BYTE 0 // bullshit on windows : byte is ambiguous
#include <VMFoundation/pluginloader.h>
#include "mappingfile.h"


#include <unordered_set>
#include <VMUtils/vmnew.hpp>
#include <VMFoundation/logger.h>

#ifdef _WIN32

#include <Windows.h>
namespace vm
{
class WindowsFileMapping__pImpl
{
	VM_DECL_API( WindowsFileMapping )
public:
	WindowsFileMapping__pImpl( WindowsFileMapping *api ) :
	  q_ptr( api ) {}

	HANDLE f = nullptr;
	HANDLE mapping = nullptr;
	FileAccess fileFlag;
	MapAccess mapFlag;
	void *addr = nullptr;
	std::unordered_set<unsigned char *> mappedPointers;
	void PrintLastErrorMsg()
	{
		DWORD dw = GetLastError();
		char msg[ 512 ];
		//LPWSTR;
		FormatMessage(
		  FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM,
		  NULL,
		  dw,
		  MAKELANGID( LANG_NEUTRAL, SUBLANG_DEFAULT ),
		  msg,
		  0, NULL );
		printf( "Last Error Code: [%d]\n", dw, msg );
	}
};


WindowsFileMapping::WindowsFileMapping( ::vm::IRefCnt *cnt ) :
  EverythingBase<IMappingFile>( cnt ), d_ptr( new WindowsFileMapping__pImpl(this) )
{
}

bool WindowsFileMapping::Open( const std::string &fileName, size_t fileSize, FileAccess fileFlags,
							   MapAccess mapFlags )
{
	VM_IMPL( WindowsFileMapping );
	bool newCreated = false;
	DWORD dwAttrib = GetFileAttributes( fileName.c_str() );
	_->fileFlag = fileFlags;
	_->mapFlag = mapFlags;
	if ( !( dwAttrib != INVALID_FILE_ATTRIBUTES && 0 == ( FILE_ATTRIBUTE_DIRECTORY & dwAttrib ) ) ) {
		newCreated = true;
	}

	//enum class FileAccess
	//{
	//	Read = GENERIC_READ,
	//	Write = GENERIC_WRITE,
	//};
	//enum class MapAccess
	//{
	//	ReadOnly = PAGE_READONLY,
	//	ReadWrite = PAGE_READWRITE
	//};

	int flags = 0;
	if ( fileFlags == FileAccess::Read )
		flags = GENERIC_READ;
	if ( fileFlags == FileAccess::Write )
		flags = GENERIC_WRITE;
	if ( fileFlags == FileAccess::ReadWrite )
		flags = GENERIC_READ | GENERIC_WRITE;

	_->f = CreateFile( TEXT( fileName.c_str() ),
					flags,
					0,
					NULL,
					OPEN_ALWAYS,
					FILE_ATTRIBUTE_NORMAL,
					NULL );

	if ( _->f == INVALID_HANDLE_VALUE ) {
		printf( "Create file failed:" );
		_->PrintLastErrorMsg();
		return false;
	}

	/// NOTE: LARGE_INTEGER is a signed 64bit integer, but fileSize is an unsigned 64-bit integer
	LARGE_INTEGER fs;
	fs.QuadPart = fileSize;

	if ( newCreated )  // Specified the file size for file mapping
	{
		SetFilePointer( _->f, fs.LowPart, &fs.HighPart, FILE_BEGIN );
		if ( !SetEndOfFile( _->f ) ) {
			printf( "Set end of file failed:" );
			_->PrintLastErrorMsg();
			return false;
		}
	}

	int mapflags = 0;
	if ( mapFlags == MapAccess::ReadOnly )
		mapflags = PAGE_READONLY;
	if ( mapFlags == MapAccess::ReadWrite )
		mapflags = PAGE_READWRITE;

	_->mapping = CreateFileMapping( _->f,
								 NULL,
								 mapflags,
								 fs.HighPart,
								 fs.LowPart,
								 NULL );

	if (_->mapping == nullptr ) {
		printf( "Create file mapping failed" );
		_->PrintLastErrorMsg();
		return false;
	}
	return true;
}

unsigned char *WindowsFileMapping::FileMemPointer( unsigned long long offset, std::size_t size )
{
	VM_IMPL( WindowsFileMapping );
	LARGE_INTEGER os;
	os.QuadPart = offset;

	int flags = 0;
	if ( _->mapFlag == MapAccess::ReadOnly )
		flags |= FILE_MAP_READ;
	if ( _->mapFlag == MapAccess::ReadWrite )
		flags |= FILE_MAP_WRITE;

	auto addr = (unsigned char *)MapViewOfFile( _->mapping,
												flags,
												os.HighPart,
												os.LowPart,
												static_cast<SIZE_T>( size ) );

	if ( !addr ) {
		printf( "MapViewOfFile failed " );
		_->PrintLastErrorMsg();
		return nullptr;
	}

	_->mappedPointers.insert( addr );
	return addr;
}

void WindowsFileMapping::DestroyFileMemPointer( unsigned char *addr )
{
	VM_IMPL( WindowsFileMapping );
	auto it = _->mappedPointers.find( addr );
	if ( it != _->mappedPointers.end() ) {
		UnmapViewOfFile( (LPVOID)addr );
		_->mappedPointers.erase( it );
	}
}

bool WindowsFileMapping::Flush()
{
	LOG_CRITICAL<<"WindowsFileMapping::Flush | Not implement yet.";
	return false;
}
bool WindowsFileMapping::Flush(void * ptr, size_t len, int flags) {
	VM_IMPL( WindowsFileMapping )
	auto res = FlushViewOfFile( ptr, len );
	if(res == 0){
		_->PrintLastErrorMsg();
	}
	return res;
}


bool WindowsFileMapping::Close()
{
	//for (auto & addr : mappedPointers)
	//	WindowsFileMapping::DestroyFileMemPointer(addr);
	VM_IMPL( WindowsFileMapping )
	CloseHandle( _->f );
	CloseHandle( _->mapping );
	return true;
}

WindowsFileMapping::~WindowsFileMapping()
{
	WindowsFileMapping::Close();
}
}  // namespace vm

std::vector<std::string> WindowsFileMappingFactory::Keys() const
{
	return { "windows" };
}

vm::IEverything *WindowsFileMappingFactory::Create( const std::string &key )
{
	return VM_NEW<vm::WindowsFileMapping>();
}

VM_REGISTER_PLUGIN_FACTORY_IMPL( WindowsFileMappingFactory )
VM_REGISTER_INTERNAL_PLUGIN_IMPL( WindowsFileMappingFactory )

#else

#include <set>
#include <memory>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>

namespace vm
{
bool LinuxFileMapping::Open( const std::string &fileName, size_t fileSize, FileAccess fileFlags, MapAccess mapFlags )
{
	Close();

	fileAccess = fileFlags;
	mapAccess = mapFlags;

	int fileflags = 0;
	if ( fileFlags == FileAccess::Read )
		fileflags |= O_RDONLY;
	if ( fileFlags == FileAccess::Write )
		fileflags |= O_WRONLY;
	if ( fileFlags == FileAccess::ReadWrite )
		fileflags |= O_RDWR;

	fileflags |= O_CREAT;  // If file don't exist, create a new one.
	fd = open64( fileName.c_str(), fileflags, 0777 );

	if ( -1 == fd ) {
		//throw std::runtime_error("can not open file");
		return false;
	}

	return true;
}
unsigned char *LinuxFileMapping::FileMemPointer( unsigned long long offset, size_t size )
{
	int prot = 0;
	if ( mapAccess == MapAccess::ReadOnly )
		prot = PROT_READ;
	if ( mapAccess == MapAccess::ReadWrite )
		prot = PROT_READ | PROT_WRITE;
	void *ptr = mmap( nullptr, size, prot, MAP_SHARED, fd, offset );
	return reinterpret_cast<unsigned char *>( ptr );
}
void LinuxFileMapping::DestroyFileMemPointer( unsigned char *addr )
{
	for ( auto it = ptrs.begin(); it != ptrs.end(); ) {
		if ( it->first == addr ) {
			munmap( it->first, it->second );
			it = ptrs.erase( it );
		} else {
			++it;
		}
	}
}
bool LinuxFileMapping::Flush()
{
	LOG_CRITICAL<<"LinuxFileMapping::Flush | Not implement yet.";
	return false;
}

bool LinuxFileMapping::Flush(void * ptr, size_t len, int flags) {
	LOG_CRITICAL<<"LinuxFileMapping::Flush(void* ptr, size_t, int) | Not implement yet.";
	return false;
}

bool LinuxFileMapping::Close()
{
	for ( auto it = ptrs.begin(); it != ptrs.end(); ) {
		munmap( it->first, it->second );
		it = ptrs.erase( it );
	}
	if ( fd != -1 ) {
		close( fd );
		fd = -1;
	}
	return true;
}

LinuxFileMapping::~LinuxFileMapping()
{
	Close();
}

}  // namespace vm

std::vector<std::string> LinuxFileMappingFactory::Keys() const
{
	return { "linux" };
}

vm::IEverything *LinuxFileMappingFactory::Create( const std::string &key )
{
	return VM_NEW<vm::LinuxFileMapping>();
}
VM_REGISTER_PLUGIN_FACTORY_IMPL( LinuxFileMappingFactory )
VM_REGISTER_INTERNAL_PLUGIN_IMPL( LinuxFileMappingFactory )

#endif
