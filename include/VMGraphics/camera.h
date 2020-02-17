#ifndef CAMERA_H
#define CAMERA_H

#include <VMat/geometry.h>
#include <VMat/transformation.h>
#include <VMGraphics/graphics_config.h>


class VMGRAPHICS_EXPORTS FocusCamera
{
	static constexpr float YAW = -90.0f;
	static constexpr float PITCH = 0.0f;
	static constexpr float SPEED = 2.5f;
	static constexpr float SENSITIVITY = 0.1f;
	static constexpr float ZOOM = 45.0f;

	// Camera Attributes
	vm::Point3f m_position;
	vm::Vector3f m_front;
	vm::Vector3f m_up;
	vm::Vector3f m_right;
	vm::Vector3f m_worldUp;
	vm::Point3f m_center;

	// Camera options
	float m_movementSpeed;
	float m_mouseSensitivity;
	float m_zoom;

public:
	// Constructor with vectors
	FocusCamera( const vm::Point3f &position = { 0.0f, 0.0f, 0.0f }, vm::Vector3f up = { 0.0f, 1.0f, 0.0f },
				 const vm::Point3f &center = { 0, 0, 0 } );

	vm::Vector3f front() const { return m_front; }
	vm::Vector3f right() const { return m_right; }
	vm::Vector3f up() const { return m_up; }

	// Returns the view matrix calculated using Euler Angles and the LookAt Matrix
	vm::Transform view() const
	{
		vm::Transform vi;
		vi.SetLookAt( m_position, m_position + m_front, m_up );
		return vi;
	}

	vm::Point3f position() const { return m_position; }

	vm::Point3f center() const { return m_center; }

	void setCenter( const vm::Point3f &center );

	void movement( const vm::Vector3f &direction, float deltaTime );

	void rotation( float xoffset, float yoffset );

	void processMouseScroll( float yoffset );

private:
	void updateCameraVectors( const vm::Vector3f &axis, double theta );
};

#endif	// CAMERA_H
