
#pragma once

#include <vector>
#include <string>
#include <VMUtils/ieverything.hpp>

namespace vm
{
class IPluginFactory
{
public:
	virtual std::vector<std::string> Keys() const = 0;
	virtual IEverything *Create( const std::string &key ) = 0;
	virtual std::string GetIID() const = 0;
	virtual ~IPluginFactory() = default;
};
using FuncType = IPluginFactory *(*)();
}  // namespace vm
