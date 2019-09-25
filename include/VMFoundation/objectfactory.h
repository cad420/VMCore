#ifndef _OBJECTFACTORY_H_
#define _OBJECTFACTORY_H_
#include <memory>
#include <string>
#include <functional>
#include <unordered_map>

#include <VMFoundation/foundation_config.h>


namespace ysl
{

	class Object;

	struct VMFOUNDATION_EXPORTS IObjectFactory
	{
		virtual ~IObjectFactory() = default;
		template<typename T>
		std::unique_ptr<T> CreateInstance(const std::string & uid)
		{
			auto ptr = CreateInstance_Implement(uid).release();
			auto pp = Object_Dynamic_Cast<T>(ptr);
			return std::unique_ptr<T>(pp);
		}
	protected:
		virtual std::unique_ptr<Object> CreateInstance_Implement(const std::string & uid) = 0;
	};

	class VMFOUNDATION_EXPORTS ObjectFactory : public IObjectFactory
	{
	public:
		ObjectFactory() = default;
		ObjectFactory(const ObjectFactory &) = delete;
		ObjectFactory & operator=(const ObjectFactory &) = delete;
		virtual ~ObjectFactory() = default;
		bool RegisterClass(const std::string& uid, std::function<std::unique_ptr<Object>()> ctor);
		void UnregisterClass(const std::string& uid);
	protected:
		std::unique_ptr<Object> CreateInstance_Implement(const std::string& uid) override;
	private:
		std::unordered_map<std::string, std::function<std::unique_ptr<Object>()>> constructors;
	};
}

#ifdef __cplusplus
extern "C"
{
#endif
VMFOUNDATION_EXPORTS ysl::IObjectFactory *GetObjectFactory();
#ifdef __cplusplus
}
#endif


#endif
