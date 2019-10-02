
#ifndef _PLUGIN_H_
#define _PLUGIN_H_
#include <vector>
#include <string>
#include <VMUtils/ieverything.hpp>

namespace ysl
{
class IPluginFactory
{
public:
	virtual std::vector<std::string> Keys() const = 0;
	virtual ::vm::IEverything *Create( const std::string &key ) = 0;
	virtual std::string GetIID() const = 0;
	virtual ~IPluginFactory() = default;
};
using FuncType = IPluginFactory *(*)();
}  // namespace ysl
#endif