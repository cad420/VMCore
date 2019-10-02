
#ifndef _LIBRARY_H_
#define _LIBRARY_H_
#include <string>
#include <VMFoundation/foundation_config.h>

namespace ysl
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

#endif