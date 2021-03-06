#ifndef SHAPE_H
#define SHAPE_H

#include <memory>

#include <VMat/geometry.h>
#include "interaction.h"
#include "material.h"
#include "graphics_config.h"

namespace vm
{
class Interaction;
class VMGRAPHICS_EXPORTS Shape
{
	std::shared_ptr<Material> m_material;

public:
	Shape( std::shared_ptr<Material> m ) :
	  m_material( m ) {}
	virtual Bound3f bound() const = 0;
	virtual Float area() const = 0;
	virtual bool intersect( const Ray &ray, Float *t, Interaction *iterac ) const = 0;
	void setMaterial( std::shared_ptr<Material> m ) { m_material = m; }
	const std::shared_ptr<Material> &getMaterial() const { return m_material; }
	virtual Interaction sample( const Point2f &u ) const { return Interaction(); }
	virtual Float pdf( const Interaction &isect ) const { return 1 / area(); }
	virtual Interaction sample( const Interaction &ref, const Point2f &u ) const { return sample( u ); }
	virtual Float pdf( const Interaction &ref, const Vector3f &wi ) const
	{
		Ray ray( wi, ref.m_p );
		Float tHit;
		Interaction isect;
		if ( intersect( ray, &tHit, &isect ) == false )
			return 0;
		Float pdf = ( ref.m_p - isect.m_p ).LengthSquared() / std::abs( Vector3f::Dot( isect.m_norm, -wi ) * area() );
		return pdf;
	}

	virtual ~Shape(){};
};

}  // namespace ysl

#endif	// SHAPE_H
