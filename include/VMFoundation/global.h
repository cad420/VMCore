#ifndef _COMMON_H_
#define _COMMON_H_

#include <memory>

namespace ysl
{
class Object;

template <typename Ty>
Ty *Object_Dynamic_Cast( Object *obj );

template <typename Ty>
const Ty *Object_Dynamic_Cast( const Object *obj );

template <typename Ty>
Ty *Object_Static_Cast( Object *obj );

template <typename Ty>
const Ty *Object_Static_Cast( const Object *obj );

template <typename Ty>
std::shared_ptr<Ty> Shared_Object_Dynamic_Cast( const std::shared_ptr<Object> &obj );

template <typename Ty>
std::shared_ptr<Ty> Shared_Object_Static_Cast( const std::shared_ptr<Object> &obj );

}  // namespace ysl
#endif
