#include <VMat/geometry.h>
#include <VMat/transformation.h>
#include <VMGraphics/camera.h>

FocusCamera::FocusCamera( const vm::Point3f &position, vm::Vector3f up, const vm::Point3f &center ) :
  m_position( position ),
  m_front( center - position ),
  m_worldUp( up ),
  m_movementSpeed( SPEED ),
  m_mouseSensitivity( SENSITIVITY ),
  m_center( center ),
  m_zoom( ZOOM )
{
	m_right = vm::Vector3f::Cross( m_front, m_worldUp );
	m_up = vm::Vector3f::Cross( m_right, m_front );
	//updateCameraVectors(QVector3D(0,1,0),QVector3D(0,0,0),0);
}

void FocusCamera::setCenter( const vm::Point3f &center )
{
	m_center = center;
	m_front = ( m_center - m_position ).Normalized();
	m_right = vm::Vector3f::Cross( m_front, m_worldUp ).Normalized();
	m_up = vm::Vector3f::Cross( m_right, m_front ).Normalized();
}

void FocusCamera::movement( const vm::Vector3f &direction, float deltaTime )
{
	const auto velocity = m_movementSpeed * direction * deltaTime;
	m_position += velocity;
}

void FocusCamera::rotation( float xoffset, float yoffset )
{
	xoffset *= m_mouseSensitivity;
	yoffset *= m_mouseSensitivity;
	const auto theta = 4.0 * ( std::fabs( xoffset ) + std::fabs( yoffset ) );
	const auto v = ( ( m_right * xoffset ) + ( m_up * yoffset ) );
	const auto axis = vm::Vector3f::Cross( v, -m_front ).Normalized();
	updateCameraVectors( axis, theta );
}

void FocusCamera::processMouseScroll( float yoffset )
{
	if ( m_zoom >= 1.0f && m_zoom <= 45.0f )
		m_zoom -= yoffset;
	if ( m_zoom <= 1.0f )
		m_zoom = 1.0f;
	if ( m_zoom >= 45.0f )
		m_zoom = 45.0f;
}

void FocusCamera::updateCameraVectors( const vm::Vector3f &axis, double theta )
{
	vm::Transform rotation;
	rotation.SetRotate( axis, theta );
	vm::Transform translation;
	translation.SetTranslate( -m_center.x, -m_center.y, -m_center.z );
	m_position = translation.Inversed() * ( rotation * ( translation * m_position ) );
	m_front = ( rotation * m_front.Normalized() );
	m_up = ( rotation * m_up.Normalized() );
	m_right = vm::Vector3f::Cross( m_front, m_up );
	m_up = vm::Vector3f::Cross( m_right, m_front );
	m_front.Normalize();
	m_right.Normalize();
	m_up.Normalize();
}
