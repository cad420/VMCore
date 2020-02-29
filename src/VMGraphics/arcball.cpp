
#include <VMGraphics/arcball.h>
#include <cmath>

namespace vm{
    class Arcball__pImpl{
        VM_DECL_API(Arcball);
    public:
        Arcball__pImpl(float radius,const Point3f &center,Arcball * api):q_ptr(api),radius(radius),center(center){}
        inline bool inner(float x,float y)
        {
            return (x * x + y * y) <= radius * radius/ 2;
        }
        inline float Z(float x,float y)
        {
            if(inner(x,y))
                return std::sqrt(radius * radius - (x*x+y*y));
            else
                return radius * radius /(2*std::sqrt(x*x + y * y));
        }

        inline void Track(int x1,int y1,int x2,int y2,Vector3f & axis,float & radians)
        {
            const auto tx1 = x1-center.x, ty1 = y1-center.y;
            const auto tx2 = x2-center.x, ty2 = y2- center.y;
            const auto v1 = Vector3f{tx1,ty1,Z(tx1,ty1)+center.z}.Normalized();
            const auto v2 = Vector3f{tx2,ty2,Z(tx2,ty2)+center.z}.Normalized();
            axis = Vector3f::Cross(v2,v1);
            radians = std::acos(Vector3f::Dot(v1,v2));
        }

        float radius = 0.f;
        Point3f center={0.f,0.f,0.f};
    };
   
    Arcball::Arcball(float radius,const Point3f & center):d_ptr(new Arcball__pImpl(radius,center,this)){
    }
    void Arcball::SetRotationCenter(const Point3f & center){
        VM_IMPL(Arcball);
        _->center = center;
    }
    void Arcball::SetArcballRadius(float radius){
        VM_IMPL(Arcball);
        _->radius = radius;
    }
    void Arcball::SetArcballAndRadius(const Point3f & center,float radius)
    {
        VM_IMPL(Arcball);
        _->center = center;
        _->radius = radius;
    }

    Vector3f Arcball::GetRotationAxis(const Point2i & pos1,const Point2i & pos2)const{
        Vector3f axis;
        float radians;
        d_ptr->Track(pos1.x,pos1.y,pos2.x,pos2.y,axis,radians);
        return axis;
    }
    float Arcball::GetRotationRadians(const Point2i & pos1,const Point2i & pos2)const{
        Vector3f axis;
        float radians;
        d_ptr->Track(pos1.x,pos1.y,pos2.x,pos2.y,axis,radians);

        return radians;
    }
    Vector4f Arcball::GetAxisAndRadians(const Point2i & pos1,const Point2i & pos2)const{
        Vector3f axis;
        float radians;
        d_ptr->Track(pos1.x,pos1.y,pos2.x,pos2.y,axis,radians);
        return Vector4f(axis,radians);
    }

    Transform Arcball::GetRotationTransform(const Point2i & pos1,const Point2i & pos2)const{
        const auto p = GetAxisAndRadians(pos1,pos2);
        return Rotate({p.x,p.y,p.z},p.w);
    }
}