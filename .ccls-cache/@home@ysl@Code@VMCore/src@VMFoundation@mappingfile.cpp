
//#include <VMUtils/vmnew.hpp>
#include <VMFoundation/pluginloader.h>
#include "mappingfile.h"
#include <VMUtils/vmnew.hpp>

#ifdef _WIN32
namespace vm
{

void WindowsFileMapping::PrintLastErrorMsg()
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

bool WindowsFileMapping::Open( const std::string &fileName, size_t fileSize, FileAccess fileFlags,
							   MapAccess mapFlags )
{
	bool newCreated = false;
	DWORD dwAttrib = GetFileAttributes( fileName.c_str() );
	fileFlag = fileFlags;
	mapFlag = mapFlags;
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

	f = CreateFile( TEXT( fileName.c_str() ),
					flags,
					0,
					NULL,
					OPEN_ALWAYS,
					FILE_ATTRIBUTE_NORMAL,
					NULL );

	if ( f == INVALID_HANDLE_VALUE ) {
		printf( "Create file failed:" );
		PrintLastErrorMsg();
		return false;
	}

	/// NOTE: LARGE_INTEGER is a signed 64bit integer, but fileSize is an unsigned 64-bit integer
	LARGE_INTEGER fs;
	fs.QuadPart = fileSize;

	if ( newCreated )  // Specified the file size for file mapping
	{
		SetFilePointer( f, fs.LowPart, &fs.HighPart, FILE_BEGIN );
		if ( !SetEndOfFile( f ) ) {
			printf( "Set end of file failed:" );
			PrintLastErrorMsg();
			return false;
		}
	}

	int mapflags = 0;
	if ( mapFlags == MapAccess::ReadOnly )
		mapflags = PAGE_READONLY;
	if ( mapFlags == MapAccess::ReadWrite )
		mapflags = PAGE_READWRITE;

	mapping = CreateFileMapping( f,
								 NULL,
								 mapflags,
								 fs.HighPart,
								 fs.LowPart,
								 NULL );

	if ( mapping == nullptr ) {
		printf( "Create file mapping failed" );
		PrintLastErrorMsg();
		return false;
	}
	return true;
}

unsigned char *WindowsFileMapping::FileMemPointer( unsigned long long offset, std::size_t size )
{
	LARGE_INTEGER os;
	os.QuadPart = offset;

	int flags = 0;
	if ( mapFlag == MapAccess::ReadOnly )
		flags |= FILE_MAP_READ;
	if ( mapFlag == MapAccess::ReadWrite )
		flags |= FILE_MAP_WRITE;

	auto addr = (unsigned char *)MapViewOfFile( mapping,
												flags,
												os.HighPart,
												os.LowPart,
												static_cast<SIZE_T>( size ) );

	if ( !addr ) {
		printf( "MapViewOfFile failed " );
		PrintLastErrorMsg();
		return nullptr;
	}

	mappedPointers.insert( addr );
	return addr;
}

void WindowsFileMapping::DestroyFileMemPointer( unsigned char *addr )
{
	auto it = mappedPointers.find( addr );
	if ( it != mappedPointers.end() ) {
		UnmapViewOfFile( (LPVOID)addr );
		mappedPointers.erase( it );
	}
}

bool WindowsFileMapping::WriteCommit()
{
	return true;
}

bool WindowsFileMapping::Close()
{
	//for (auto & addr : mappedPointers)
	//	WindowsFileMapping::DestroyFileMemPointer(addr);
	CloseHandle( f );
	CloseHandle( mapping );
	return true;
}

WindowsFileMapping::~WindowsFileMapping()
{
	WindowsFileMapping::Close();
}
}

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
bool LinuxFileMapping::WriteCommit()
{
	return true;
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

vm::IEverything * LinuxFileMappingFactory::Create( const std::string &key )
{
	return VM_NEW<vm::LinuxFileMapping>();
}
VM_REGISTER_PLUGIN_FACTORY_IMPL( LinuxFileMappingFactory )
VM_REGISTER_INTERNAL_PLUGIN_IMPL( LinuxFileMappingFactory )


#endif
