#pragma once

#include <VMat/transformation.h>
#include <VMat/geometry.h>
#include <VMUtils/common.h>
#include "film.h"

namespace vm
{

struct CameraSample
{
	Point2f FilmSample;
	Point2f LensSample;
	Float Time;
};

/**
 * @brief The base class of all camera. It only contains a \a CameraToWorld transform used to transform the 
 * final ray generated by a specified camera into the scene space. So a derived camera only need to take the
 * responsibility to generate the ray in the camera space.
 * 
 */
class Camera
{
public:
	Camera( const Transform &cameraToWorld,Film * film) :
	  CameraToWorld( cameraToWorld ),film(film){}
	virtual Float GenerateRay( const CameraSample &sample, Ray *ray ) const = 0;
	virtual Float GenerateDifferentialRay(const CameraSample & sample,DifferentialRay * ray)const;
	Transform CameraToWorld;  // It transform the ray from camera space to world space, i.e. The inverse of LookAt transform
	Film * film = nullptr;
};

/**
 * @brief 
 * @note Projective camera here is not like 3d transform in OpenGL or other graphics APIs in some aspects.
 * The x and y range of NDC in this camera is [0,1], which is OpenGL is [-1,1].
 * 
 * Raster space is almost like NDC except the x and y coordinates range from (0,0) to (resolution.x, resolution.y)
 * 
 * The order of the transform is Screen --> Raster --> Camera --> World. The constructor gives a screen bound(start from upper-left coner)
 * 
 */
class ProjectiveCamera : public Camera
{
public:
	ProjectiveCamera( const Transform &cameraToWorld, const Transform &cameraToScreen,const Bound2f & screenSize,Float shutterOpen,Float shutterClose,Float aperture,Float focalDistance,Film * film );

protected:
	Transform ScreenToRaster;  // Could be evaluated by the screen size and the file resolution directly
	Transform RasterToScreen;  // Inverse of ScreenToRaster
	Transform CameraToScreen;  // The P matrix in MVP transformation.  It depends on the projection style( perspective or orthognal)
	Transform RasterToCamera;  // rasterToCamera transform the point on the canvas to the space where the camera is (camera space)
	Float ShutterOpen,ShutterClose;
	Float Aperture,FocalDistance;
};

class OrthographicCamera : public ProjectiveCamera
{
public:
	OrthographicCamera( const Transform &cameraToWorld,
	 const Bound2f &screenSize,
	 Float shutterOpen, 
	 Float shutterClose,
	 Float aperture,
	 Float focalDistance,
	 Film *film );
	Float GenerateRay( const CameraSample &sample, Ray * ray ) const override;
	Float GenerateDifferentialRay(const CameraSample & sample,DifferentialRay * ray)const override;
	private:
	Vec3f dxCamera,dyCamera;
};

class PerspectiveCamera : public ProjectiveCamera
{
public:
	PerspectiveCamera( const Transform &cameraToWorld, 
	const Bound2f &screenSize,
	Float fov,
	Float shutterOpen,
	Float shutterClose,
	Float aperture,
	Float focalDistance,
	Film *film );
	Float GenerateRay( const CameraSample &sample, Ray * ray ) const override;
	Float GenerateDifferentialRay(const CameraSample &sample,DifferentialRay * ray)const override;
};



}  // namespace 