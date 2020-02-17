
#include <VMGraphics/pbrcamera.h>
#include <VMat/arithmetic.h>
#include <cassert>

namespace vm
{

Float Camera::GenerateDifferentialRay(const CameraSample & sample,DifferentialRay * ray)const
{
		Float wt = GenerateRay(sample,ray);
		CameraSample shift = sample;
		shift.FilmSample.x++;
		Ray r;
		Float wtx = GenerateRay(shift,&r);
		if(wtx == 0)return 0;
		ray->Ox = r.o;
		ray->Dx = r.d;
		shift.FilmSample.x--;
		shift.FilmSample.y++;
		Float wty = GenerateRay(shift,&r);
		if(wty == 0)return 0;
		ray->Oy = r.o;
		ray->Dy = r.d;
		ray->Differential = true;
		return wt;

}

ProjectiveCamera::ProjectiveCamera( const Transform &cameraToWorld,
									const Transform &cameraToScreen,
									const Bound2f &screenBound,
									Float shutterOpen,Float shutterClose,Float aperture,Float focal,
									Film *film ):
  Camera( cameraToWorld, film ), 
  CameraToScreen( cameraToScreen ),
  ShutterOpen(shutterOpen),
  ShutterClose(shutterClose),
  Aperture(aperture),
  FocalDistance(focal)
{
	ScreenToRaster = Scale( film->Resolution.x, film->Resolution.y, 1.0 ) *
					 Scale( 1 / ( screenBound.max.x - screenBound.min.x ), 1 / ( screenBound.max.y - screenBound.min.y ), 1.0 ) *
					 Translate( -screenBound.min.x, -screenBound.max.y, 0 );
	RasterToScreen = ScreenToRaster.Inversed();
	RasterToCamera = cameraToScreen.Inversed() * RasterToScreen;
}

OrthographicCamera::OrthographicCamera( const Transform &cameraToWorld,
										const Bound2f &screenSize,
									Float shutterOpen,Float shutterClose,Float aperture,Float focal,
										Film *film ):
  ProjectiveCamera( cameraToWorld, Orthographic( 0, 1 ), screenSize,shutterOpen,shutterClose,aperture,focal, film )
{	
	dxCamera = RasterToCamera * Vec3f(1,0,0);
	dyCamera = RasterToCamera * Vec3f(0,1,0);
}

Float OrthographicCamera::GenerateRay( const CameraSample &sample, Ray *ray ) const
{
	const auto pFilm = Point3f( sample.FilmSample.x, sample.FilmSample.y, 0 );
	const auto pCamera = RasterToCamera * pFilm;
	assert(ray);
	*ray = Ray( Vector3f( 0, 0, 1 ), pCamera ); // along z-axis

	if(Aperture > 0){
		const auto lenPos = Aperture * concentricDiskSample(sample.LensSample); // sampling a point in the concentric disk which the center is (0, 0)
		Point3f focus = (*ray)(FocalDistance / ray->d.z);
		ray->o = Point3f(lenPos.x,lenPos.y);
		ray->d = (focus - ray->o).Normalized();
	}

	ray->Time = Lerp(sample.Time,ShutterOpen,ShutterClose);
	*ray = CameraToWorld * ( *ray );
	return 1;
}


Float OrthographicCamera::GenerateDifferentialRay(const CameraSample & sample,DifferentialRay * ray)const
{
	const auto filmPos = Point3f(sample.FilmSample.x,sample.FilmSample.y,0);
	const auto camera = RasterToCamera*(filmPos);

	*ray = DifferentialRay(Vec3f(0,0,1),camera);
	if(Aperture > 0)
	{
		const auto lenPos = Aperture * concentricDiskSample(sample.LensSample); // sampling a point in the concentric disk which the center is (0, 0)
		Point3f focus = (*ray)(FocalDistance / ray->d.z);
		ray->o = Point3f(lenPos.x,lenPos.y);
		ray->d = (focus - ray->o).Normalized();

	}
	if(Aperture > 0){
		const auto lensPos = Aperture * concentricDiskSample(sample.LensSample);
		const auto t = FocalDistance / ray->d.z;

		auto focus = camera + dxCamera + (t * Vec3f(0,0,1));

		ray->Ox = Point3f(lensPos.x,lensPos.y,0);
		ray->Dx = (focus - ray->Ox).Normalized();

		focus = camera + dyCamera + (t * Vec3f(0,0,1));
		ray->Oy = Point3f(lensPos.x,lensPos.y,0);
		ray->Dy = (focus - ray->Oy).Normalized();
	}else
	{
		ray->Ox = ray->o + dxCamera;
		ray->Oy = ray->o + dyCamera;
		ray->Dx = ray->Dy = ray->d;
	}
	ray->Time = Lerp(sample.Time,ShutterOpen,ShutterClose);
	ray->Differential = true;
	*ray = CameraToWorld * (*ray);
	return 1; 
}

PerspectiveCamera::PerspectiveCamera( const Transform &cameraToWorld,
									  const Bound2f &screenSize, Float fov,
									 Float shutterOpen,Float shutterClose,Float aperture,Float focal,
									  Film *film ) :
  ProjectiveCamera( cameraToWorld, Perspective( fov, 0.0001, 1000.f ), screenSize,shutterOpen,shutterClose,aperture,focal, film )
{
}

Float PerspectiveCamera::GenerateRay( const CameraSample &sample, Ray *ray ) const
{
	const auto pFilm = Point3f( sample.FilmSample.x, sample.FilmSample.y, 0 );
	const auto pCamera = RasterToCamera * pFilm;
	assert(ray);
	*ray = Ray( Vector3f( pCamera ).Normalized(), { 0, 0, 0 } );

	*ray = CameraToWorld * ( *ray );
	return 1;
}

}  // namespace vm
