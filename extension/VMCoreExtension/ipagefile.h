
#pragma once

#include "ifile.h"

namespace vm
{
class IPageFile : public IFile
{
public:
	virtual ~IPageFile() = default;
	/**
	 * \brief Get the page give by \a pageID. If the page does not exist in the cache, it will be swapped in.
	 * \note The page data pointed by the  pointer returned by the function is locked. \refitem UnlockPage should be
	 * called after using it.
	 *
	 * \sa UnlockPage
	 */
	virtual const void *GetPage( size_t pageID ) = 0;

	/**
	 * @brief If the buffer returned by \refitem GetPage is no long used.
	 * This function should be called to allow the page given by \a pageID could be swapped out.
	 * This funciont could be called many times. It remains no effects when the page is already unlocked.
	 * @param pageID
	 *
	 * \sa GetPage
	 */
	virtual void UnlockPage(size_t pageID) = 0;

	virtual void Flush() = 0;

	virtual void Write( const void *page, size_t pageID, bool flush ) = 0;

	virtual void Flush( size_t pageID ) = 0;


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
}  // namespace vm