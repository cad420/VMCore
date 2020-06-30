#include <VMat/geometry.h>
#include <VMat/transformation.h>
#include <VMGraphics/camera.h>
#include <fstream>
#include <sstream>

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


namespace vm{
LookAtTransform::LookAtTransform( const Point3f &position,const  Vector3f &worldUp, const Point3f &center ) :
  m_position( position ),
  m_front( center - position ),
  m_worldUp( worldUp ),
  m_movementSpeed( SPEED ),
  m_mouseSensitivity( SENSITIVITY ),
  m_zoom( ZOOM )
{
	m_right = Vector3f::Cross( m_front, m_worldUp ).Normalized();
	m_up = Vector3f::Cross( m_right, m_front ).Normalized();
	//updateCameraVectors(QVector3D(0,1,0),QVector3D(0,0,0),0);
}

	LookAtTransform::LookAtTransform( const Point3f &position, Vector3f up, const Vector3f &front ):
  m_position( position ),
  m_front( front.Normalized() ),
  m_worldUp( up ),
  m_movementSpeed( SPEED ),
  m_mouseSensitivity( SENSITIVITY ),
  m_zoom( ZOOM )
{
	
}

void LookAtTransform::Update( const Point3f &position, Vector3f worldUp, const Point3f & center )
{
	m_position = position;
	m_front = (center - position).Normalized();
	m_worldUp = worldUp.Normalized();
	m_right = Vector3f::Cross( m_front, m_worldUp ).Normalized();
	m_up = Vector3f::Cross( m_right, m_front ).Normalized();
}

void LookAtTransform::Update( const Point3f &position, Vector3f worldUp, const Vec3f &front )
{
	m_position = position;
	m_front = front.Normalized();
	m_worldUp = worldUp.Normalized();
	m_right = Vector3f::Cross( m_front, m_worldUp ).Normalized();
	m_up = Vector3f::Cross( m_right, m_front ).Normalized();
}

Transform LookAtTransform::LookAt() const
{
	Transform vi;
	vi.SetLookAt( m_position, m_position + m_front, m_up );
	return vi;
}

void LookAtTransform::SetPosition( const Point3f &pos )
{
	m_position = pos;
}

void LookAtTransform::SetCenter( const Point3f &center )
{
	//m_center = center;
	m_front = ( center - m_position ).Normalized();
	m_right = Vector3f::Cross( m_front, m_worldUp ).Normalized();
	m_up = Vector3f::Cross( m_right, m_front ).Normalized();
}

void LookAtTransform::Move( const Vector3f &direction, float deltaTime )
{
	const auto velocity = m_movementSpeed * direction * deltaTime;
	m_position += velocity;
}

void LookAtTransform::Rotate( float xoffset, float yoffset, const Point3f &center )
{
	xoffset *= m_mouseSensitivity;
	yoffset *= m_mouseSensitivity;
	const auto theta = 4.0 * ( std::fabs( xoffset ) + std::fabs( yoffset ) );
	const auto v = ( ( m_right * xoffset ) + ( m_up * yoffset ) );
	const auto axis = Vector3f::Cross( v, -m_front ).Normalized();
	Rotate( axis, theta,center );
}

void LookAtTransform::ProcessMouseScroll( float yoffset )
{
	if ( m_zoom >= 1.0f && m_zoom <= 45.0f )
		m_zoom -= yoffset;
	if ( m_zoom <= 1.0f )
		m_zoom = 1.0f;
	if ( m_zoom >= 45.0f )
		m_zoom = 45.0f;
}

void LookAtTransform::Rotate( const Vector3f &axis, double theta, const Point3f &center )
{
	Transform rotation;
	rotation.SetRotate( axis, theta );
	Transform translation;
	translation.SetTranslate( -center.x, -center.y, -center.z );
	m_position = translation.Inversed() * ( rotation * ( translation * m_position ) );
	m_front = ( rotation * m_front.Normalized() );
	m_up = ( rotation * m_up.Normalized() );
	m_right = Vector3f::Cross( m_front, m_up );
	m_up = Vector3f::Cross( m_right, m_front );
	m_front.Normalize();
	m_right.Normalize();
	m_up.Normalize();

}

ViewingTransform::ViewingTransform( const Point3f &position, const Vector3f &up, const Point3f &center )
{
	lookAtTransform = LookAtTransform( position, up, center );
	projTransform.SetGLPerspective( fov, aspectRatio, nearPlan, farPlan );
}

void ViewingTransform::SetProjectionMatrix( const Transform &projection )
{
	( this->projTransform ) = projection;
}

const Transform &ViewingTransform::ProjectionMatrix() const
{
	return projTransform;
}

void ViewingTransform::SetCamera( const LookAtTransform & lookAtTransform, const Transform & projTransform )
{
	this->lookAtTransform = lookAtTransform;
	this->projTransform = projTransform;
}

void ViewingTransform::SetCamera( const Point3f &position, Vector3f worlUp, const Point3f &center,
						float nearPlane, float farPlane, float aspectRatio, float fov )
{
	lookAtTransform.Update( position, worlUp, center );

	this->nearPlan = nearPlane;
	this->farPlan = farPlane;
	this->aspectRatio = aspectRatio;
	this->fov = fov;

	Transform perspectiveMatrix;
	perspectiveMatrix.SetGLPerspective( fov, aspectRatio, nearPlane, farPlane );
	SetProjectionMatrix( perspectiveMatrix );
}

void ViewingTransform::SetCamera( const Point3f &position, Vector3f worlUp, const Vector3f & front,
						float nearPlane, float farPlane, float aspectRatio, float fov )
{
	lookAtTransform.Update( position, worlUp, front );

	this->nearPlan = nearPlane;
	this->farPlan = farPlane;
	this->aspectRatio = aspectRatio;
	this->fov = fov;

	Transform perspectiveMatrix;
	perspectiveMatrix.SetGLPerspective( fov, aspectRatio, nearPlane, farPlane );
	SetProjectionMatrix( perspectiveMatrix );
}

std::vector<Point3f> ViewingTransform::GetFrustumLines() const{
/*
			 *
			 * 7------6
			 * |\	 /|
			 * | 3--2 |
			 * | |	| |
			 * | 0--1 |
			 * 4------5
			 *
			 * Frustum
			 */

	const auto pos = lookAtTransform.GetPosition();
	const auto direction = lookAtTransform.GetFront().Normalized();
	const auto right = lookAtTransform.GetRight().Normalized();
	const auto up = lookAtTransform.GetUp().Normalized();
	const auto farPlane = ( GetFarPlane() + GetNearPlane() ) / 2.0;
	const auto nearPlane = ( GetNearPlane() );
	const auto tanfov2 = 2.0 * std::tan( GetFov() * Pi / 180 / 2.0 );
	const auto atanfov2 = GetAspectRatio() * tanfov2;
	const auto nearCenter = pos + direction * nearPlane;
	const auto farCenter = pos + direction * farPlane;
	const auto nearHalfHeight = tanfov2 * nearPlane;
	const auto nearHalfWidth = atanfov2 * nearPlane;
	const auto farHalfHeight = tanfov2 * farPlane;
	const auto farHalfWidth = atanfov2 * farPlane;

	//lines->GetBufferObject()->Resize(8 * sizeof(Point3f));
	std::vector<Point3f> ptr( 8 );
	//auto ptr = reinterpret_cast<Point3f*>(lines->GetBufferObject()->LocalData());

	ptr[ 0 ] = nearCenter + nearHalfWidth * ( -right ) + nearHalfHeight * ( -up );
	ptr[ 1 ] = nearCenter + nearHalfWidth * ( right ) + nearHalfHeight * ( -up );
	ptr[ 2 ] = nearCenter + nearHalfWidth * ( right ) + nearHalfHeight * ( up );
	ptr[ 3 ] = nearCenter + nearHalfWidth * ( -right ) + nearHalfHeight * ( up );

	ptr[ 4 ] = farCenter + farHalfWidth * ( -right ) + farHalfHeight * ( -up );
	ptr[ 5 ] = farCenter + farHalfWidth * ( right ) + farHalfHeight * ( -up );
	ptr[ 6 ] = farCenter + farHalfWidth * ( right ) + farHalfHeight * ( up );
	ptr[ 7 ] = farCenter + farHalfWidth * ( -right ) + farHalfHeight * ( up );
	//auto lines = MakeVMRef<ArrayFloat3>();
	//lines->GetBufferObject()->SetLocalData(ptr, sizeof(ptr));
	return ptr;
}

struct ViewMatrixJSONStruct : json::Serializable<ViewMatrixJSONStruct> {
	VM_JSON_FIELD( std::vector<float>, pos );
	VM_JSON_FIELD( std::vector<float>, up );
	VM_JSON_FIELD( std::vector<float>, front );
};

struct PerspMatrixJSONStruct : json::Serializable<PerspMatrixJSONStruct>
{
	VM_JSON_FIELD( float, fov );
	VM_JSON_FIELD( float, nearPlane );
	VM_JSON_FIELD( float, farPlane );
	VM_JSON_FIELD( float, aspectRatio );
};

struct CameraJSONStruct : json::Serializable<CameraJSONStruct>
{
	VM_JSON_FIELD( ViewMatrixJSONStruct, viewMatrix );
	VM_JSON_FIELD( PerspMatrixJSONStruct, perspectiveMatrix );
};

ViewingTransform ConfigCamera( const std::string &jsonFileName ){
	//using namespace rapidjson;
	//std::ifstream ifs( jsonFileName );
	//rapidjson::IStreamWrapper isw( ifs );
	//Document d;
	//d.ParseStream( isw );

	std::ifstream json( jsonFileName );
	CameraJSONStruct JSON;
	json >> JSON;

	// Camera Params
	// View Matrix: up front center
	// Perspective Matrix: fov, nearPlane, farPlane, aspectRatio

	Point3f pos;
	Vector3f up;
	Vector3f front;

	front.x = JSON.viewMatrix.front[ 0 ];
	front.y = JSON.viewMatrix.front[ 1 ];
	front.z = JSON.viewMatrix.front[ 2 ];

	pos.x = JSON.viewMatrix.pos[ 0 ];
	pos.y = JSON.viewMatrix.pos[ 1 ];
	pos.z = JSON.viewMatrix.pos[ 2 ];

	up.x = JSON.viewMatrix.up[ 0 ];
	up.y = JSON.viewMatrix.up[ 1 ];
	up.z = JSON.viewMatrix.up[ 2 ];

	const float fov = JSON.perspectiveMatrix.fov;
	const float nearPlane = JSON.perspectiveMatrix.nearPlane;
	const float farPlane = JSON.perspectiveMatrix.farPlane;
	const float aspectRatio = JSON.perspectiveMatrix.aspectRatio;

	Transform perspectiveMatrix;
	perspectiveMatrix.SetGLPerspective( fov, aspectRatio, nearPlane, farPlane );

	ViewingTransform camera;
	camera.SetCamera( pos, up, front, nearPlane, farPlane, aspectRatio, fov );
	return camera;
}
void SaveCameraAsJson( const ViewingTransform & camera, const std::string &jsonFileName ){
	auto pos = camera.GetViewMatrixWrapper().GetPosition();
	auto front = camera.GetViewMatrixWrapper().GetFront();
	auto up = camera.GetViewMatrixWrapper().GetUp();

	auto fov = camera.GetFov();
	auto nearPlane = camera.GetNearPlane();
	auto farPlane = camera.GetFarPlane();
	auto aspectRatio = camera.GetAspectRatio();

	CameraJSONStruct JSON;

	JSON.viewMatrix.front[ 0 ] = front.x;
	JSON.viewMatrix.front[ 1 ]= front.y;
	JSON.viewMatrix.front[ 2 ]= front.z;

	JSON.viewMatrix.pos[ 0 ]=pos.x;
	JSON.viewMatrix.pos[ 1 ]=pos.y;
	JSON.viewMatrix.pos[ 2 ]=pos.z;

	JSON.viewMatrix.up[ 0 ]=up.x;
	JSON.viewMatrix.up[ 1 ]=up.y;
	JSON.viewMatrix.up[ 2 ]=up.z;

	JSON.perspectiveMatrix.fov = fov;
	JSON.perspectiveMatrix.nearPlane = nearPlane;
	JSON.perspectiveMatrix.farPlane = farPlane;
	JSON.perspectiveMatrix.aspectRatio = aspectRatio;

	std::ofstream json( jsonFileName );
	json::Writer writer;
	json << writer.write(JSON);
	json.close();
}

}
