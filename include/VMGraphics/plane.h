
#ifndef _PLANE_H_
#define _PLANE_H_
#include "shape.h"
#include "graphics_config.h"

namespace vm
{
class VMGRAPHICS_EXPORTS Plane
{
	Float mOrigin = 0;
	Vector3f mNormal;

public:
	Plane() {}
	Plane( const Vector3f &n, Float o ) :
	  mOrigin( o ), mNormal( n ) { assert( n.IsNull() == false ); }
	Plane( const Vector3f &n, const Point3f &p ) :
	  mOrigin( vm::Dot( n, p ) ), mNormal( n ) { assert( n.IsNull() == false ); }
	bool Intersect( const Ray &ray, Float *t ) const;
	Vector3f GetNormal() const { return mNormal; }
	Float GetOrigin() const { return mOrigin; }
	Float GetDistance( const Point3f &pos );
	bool IsValid() const { return mNormal.IsNull() == false; }
};
}  // namespace ysl
#endif