#ifndef CAMERA_H
#define CAMERA_H

#include <VMat/geometry.h>
#include <VMat/transformation.h>
#include <VMUtils/json_binding.hpp>
#include <VMGraphics/graphics_config.h>

#include <vector>


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
namespace vm
{


class VMGRAPHICS_EXPORTS LookAtTransform
{
	static constexpr float YAW = -90.0f;
	static constexpr float PITCH = 0.0f;
	static constexpr float SPEED = 1.0f;
	static constexpr float SENSITIVITY = .1f;
	static constexpr float ZOOM = 45.0f;

	// Camera Attributes
	Point3f m_position;
	Vector3f m_front;
	Vector3f m_up;
	Vector3f m_right;  // Redundant
	Vector3f m_worldUp;
	//Point3f m_center;

	// Camera options
	float m_movementSpeed;
	float m_mouseSensitivity;
	float m_zoom;

public:
	// Constructor with vectors
	LookAtTransform( const Point3f &position = { 0.0f, 0.0f, 0.0f }, Vector3f up = { 0.0f, 1.0f, 0.0f },
					   const Point3f &center = { 0, 0, 0 } );
	LookAtTransform( const Point3f &position, Vector3f up,
					   const Vector3f &front );

	Vector3f GetFront() const { return m_front; }
	void SetFront( const Vector3f &front ) { m_front = front.Normalized(); }
	Vector3f GetRight() const { return m_right; }
	Vector3f GetUp() const { return m_up; }
	void Update( const Point3f &position, Vector3f worlUp,
					   const Point3f &center );
	void Update( const Point3f &position, Vector3f worldUp, const Vec3f &front );
	// Returns the view matrix calculated using Euler Angles and the LookAt Matrix
	Transform LookAt() const;
	Point3f GetPosition() const { return m_position; }
	void SetPosition( const Point3f &pos );
	//Point3f GetCenter() const { return m_center; }
	void SetCenter( const Point3f &center );
	void Move( const Vector3f &direction, float deltaTime );
	void Rotate( float xOffset, float yOffset, const Point3f &center );
	void ProcessMouseScroll( float yOffset );
	void Rotate( const Vector3f &axis, double theta, const Point3f &center );

private:
};

class VMGRAPHICS_EXPORTS ViewingTransform
{
public:
    ViewingTransform() = default;
	ViewingTransform( const Point3f &position,
			const Vector3f &up,
			const Point3f &center );

	Transform ViewMatrix() const { return lookAtTransform.LookAt(); }

	LookAtTransform & GetViewMatrixWrapper() { return lookAtTransform; }

	const LookAtTransform & GetViewMatrixWrapper() const { return lookAtTransform; }

	void SetViewMatrixWrapper( const LookAtTransform & lookAtTransform ) { this->lookAtTransform = lookAtTransform; }

	void SetProjectionMatrix( const Transform & projection );

	const Transform & ProjectionMatrix() const;

	Transform ProjectViewMatrix() const { return ( projTransform ) * ViewMatrix(); }

	Transform GetPerspectiveMatrix() { return projTransform; }

	const Transform & GetPerspectiveMatrix() const { return projTransform; }

	void SetPerspectiveMatrix( const Transform  &persp ) { projTransform = persp ; }




	void SetCamera( const LookAtTransform & lookAtTransform, const Transform & projTransform );


	void SetCamera( const Point3f &position, Vector3f worlUp,
					const Point3f &center, float nearPlane, float farPlane, float aspectRatio, float fov );

	void SetCamera( const Point3f &position, Vector3f worlUp,
					const Vector3f &front, float nearPlane, float farPlane, float aspectRatio, float fov );


	float GetFov() const { return fov; }

	void SetFov( float fov )
	{
		this->fov = Clamp( fov, 1.f, 89.f );
		UpdateProjMatrix();
	}

	float GetNearPlane() const { return nearPlan; }

	void SetNearPlane( float np )
	{
		nearPlan = np;
		UpdateProjMatrix();
	}

	float GetFarPlane() const { return farPlan; }

	void SetFarPlane( float fp )
	{
		farPlan = fp;
		UpdateProjMatrix();
	}

	float GetAspectRatio() const { return aspectRatio; }

	void SetAspectRatio( float aspect ) { aspectRatio = aspect; }

	std::vector<Point3f> GetFrustumLines() const;

private:
	void UpdateProjMatrix() { projTransform.SetGLPerspective( fov, aspectRatio, nearPlan, farPlan ); }
	LookAtTransform lookAtTransform;
	Transform projTransform;
	float fov = 60;
	float aspectRatio = 1024.0 / 768.0;
	float nearPlan = 0.01;
	float farPlan = 1000;
};

VMGRAPHICS_EXPORTS ViewingTransform ConfigCamera( const std::string &jsonFileName );
VMGRAPHICS_EXPORTS void SaveCameraAsJson( const ViewingTransform & camera, const std::string &jsonFileName );


}

#endif	// CAMERA_H
