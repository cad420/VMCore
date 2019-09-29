
#ifndef _IPAGEFILE_H_
#define _IPAGEFILE_H_

#include <VMUtils/ieverything.hpp>

namespace ysl
{
class IPageFile : public ::vm::IEverything
{
public:
	virtual ~IPageFile() = default;
	/**
				 * \brief Get the page give by \a pageID. If the page does not exist in the cache, it will be swapped in.
				 * \note The page data pointed by the  pointer returned by the function is only valid at current call.
				 * It could be invalid when next call because its data has been swapped out.
				 */
	virtual const void *GetPage( size_t pageID ) = 0;

	/**
		 * \brief Returns the page size by bytes
		 */
	virtual size_t GetPageSize() const = 0;
	/**
		 * \brief
		 * \return
		 */
	virtual size_t GetPhysicalPageCount() const = 0;
	/**
		 * \brief
		 * \return
		 */
	virtual size_t GetVirtualPageCount() const = 0;

protected:
};
}  // namespace ysl
#endif