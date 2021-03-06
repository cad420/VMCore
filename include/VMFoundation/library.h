
#pragma once

#include <string>
#include <VMFoundation/foundation_config.h>

namespace vm
{
class VMFOUNDATION_EXPORTS Library
{
public:
	Library( const std::string &name );
	void *Symbol( const std::string &name ) const;
	void Close();
	~Library();

private:
	void *lib;
};
}  // namespace ysl
