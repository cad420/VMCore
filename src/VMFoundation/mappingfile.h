#pragma once

/*
 * Only for internal use
 */

#include <VMCoreExtension/ifilemappingplugininterface.h>
#include <VMCoreExtension/plugin.h>
#include <VMUtils/common.h>

#ifdef _WIN32
namespace vm
{
class WindowsFileMapping__pImpl;
class WindowsFileMapping : public EverythingBase<IMappingFile>
{
	VM_DECL_IMPL( WindowsFileMapping )
public:
	WindowsFileMapping( ::vm::IRefCnt *cnt );
	bool Open( const std::string &fileName, size_t fileSize, FileAccess fileFlags, MapAccess mapFlags ) override;
	unsigned char *MemoryMap( unsigned long long offset, std::size_t size ) override;
	void MemoryUnmap( unsigned char *addr ) override;
	bool Flush() override;
	bool Flush(void * ptr, size_t len, int flags) override;
	bool Close() override;
	~WindowsFileMapping();
};

}  // namespace vm

class WindowsFileMappingFactory : public vm::IPluginFactory
{
	DECLARE_PLUGIN_FACTORY( "vmcore.imappingfile" )
public:
	std::vector<std::string> Keys() const override;
	::vm::IEverything *Create( const std::string &key ) override;
};

VM_REGISTER_PLUGIN_FACTORY_DECL( WindowsFileMappingFactory )

//EXPORT_PLUGIN_FACTORY( WindowsFileMappingFactory )

#else

#include <set>

namespace vm
{
class LinuxFileMapping : public ::vm::EverythingBase<IMappingFile>
{
	std::unordered_map<unsigned char *,size_t> mappedPointers;
	int fd = -1;
	FileAccess fileAccess;
	MapAccess mapAccess;
	size_t fileSize = 0;

public:
	LinuxFileMapping( ::vm::IRefCnt *cnt ) :
	  ::vm::EverythingBase<IMappingFile>( cnt ) {}
	bool Open( const std::string &fileName, size_t fileSize, FileAccess fileFlags, MapAccess mapFlags ) override;
	unsigned char *MemoryMap( unsigned long long offset, size_t size ) override;
	void MemoryUnmap( unsigned char *addr ) override;
	bool Flush() override;
	bool Flush(void * ptr, size_t len, int flags) override;
	bool Close() override;
	~LinuxFileMapping();
};
}  // namespace vm

class LinuxFileMappingFactory : public vm::IPluginFactory
{
	DECLARE_PLUGIN_FACTORY( "vmcore.imappingfile" )
public:
	std::vector<std::string> Keys() const override;
	::vm::IEverything *Create( const std::string &key ) override;
};
VM_REGISTER_PLUGIN_FACTORY_DECL( LinuxFileMappingFactory )

#endif
