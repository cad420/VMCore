#pragma once
#include <VMUtils/common.h>
#include <VMat/transformation.h>
namespace vm{
    class Arcball__pImpl;
    class Arcball{
        VM_DECL_IMPL(Arcball);
        public:
        Arcball(float radius,const Point3f & center);
        void SetRotationCenter(const Point3f & center);
        void SetArcballRadius(float radius);
        void SetArcballAndRadius(const Point3f & center,float radius);

        Vector3f GetRotationAxis(const Point2i & pos1,const Point2i & pos2)const;
        float GetRotationRadians(const Point2i & pos1,const Point2i & pos2)const;
        Vector4f GetAxisAndRadians(const Point2i & pos1,const Point2i & pos2)const;
        Transform GetRotationTransform(const Point2i & pos1,const Point2i & pos2)const;
    };
}
