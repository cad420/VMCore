#pragma once

/*
 * Only for internal use
 */

#include <VMCoreExtension/ifilemappingplugininterface.h>
#include <VMCoreExtension/plugin.h>

#ifdef _WIN32

#include <Windows.h>
#include <unordered_set>

namespace vm
{
class WindowsFileMapping : public ::vm::EverythingBase<IMappingFile>
{
	HANDLE f = nullptr;
	HANDLE mapping = nullptr;
	FileAccess fileFlag;
	MapAccess mapFlag;
	void *addr = nullptr;
	std::unordered_set<unsigned char *> mappedPointers;
	void PrintLastErrorMsg();

public:
	WindowsFileMapping( ::vm::IRefCnt *cnt ) :
	  ::vm::EverythingBase<IMappingFile>( cnt ) {}
	bool Open( const std::string &fileName, size_t fileSize, FileAccess fileFlags, MapAccess mapFlags ) override;
	unsigned char *FileMemPointer( unsigned long long offset, std::size_t size ) override;
	void DestroyFileMemPointer( unsigned char *addr ) override;
	bool WriteCommit() override;
	bool Close() override;
	~WindowsFileMapping();
};

}  // namespace vm

class WindowsFileMappingFactory : public vm::IPluginFactory
{
	DECLARE_PLUGIN_FACTORY( "vmcore.imappingfile" )
	std::vector<std::string> Keys() const override;
	vm::IEverything *Create( const std::string &key ) override;
};

VM_REGISTER_PLUGIN_FACTORY_DECL( WindowsFileMappingFactory )

//EXPORT_PLUGIN_FACTORY( WindowsFileMappingFactory )

#else

#include <set>

namespace vm
{
class LinuxFileMapping : public ::vm::EverythingBase<IMappingFile>
{
	std::set<std::pair<void *, size_t>> ptrs;
	int fd = -1;
	FileAccess fileAccess;
	MapAccess mapAccess;

public:
	LinuxFileMapping( ::vm::IRefCnt *cnt ) :
	  ::vm::EverythingBase<IMappingFile>( cnt ) {}
	bool Open( const std::string &fileName, size_t fileSize, FileAccess fileFlags, MapAccess mapFlags ) override;
	unsigned char *FileMemPointer( unsigned long long offset, size_t size ) override;
	void DestroyFileMemPointer( unsigned char *addr ) override;
	bool WriteCommit() override;
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
