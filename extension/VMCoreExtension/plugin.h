
#pragma once

#include <vector>
#include <string>
#include <VMUtils/ieverything.hpp>

namespace vm
{
class IPluginFactory
{
public:
	virtual int Keys(const char ** keys) const = 0;
	virtual IEverything *Create( const char *key ) = 0;
	virtual std::string GetIID() const = 0;
	virtual ~IPluginFactory() = default;
};
using FuncType = IPluginFactory *(*)();
}  // namespace vm
