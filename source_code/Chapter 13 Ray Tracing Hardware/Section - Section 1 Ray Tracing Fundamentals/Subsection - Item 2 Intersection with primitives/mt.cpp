struct Hit { bool hit; float t,u,v; };
Hit intersectTriangle(const Vec3& o, const Vec3& d, 
                      const Vec3& v0, const Vec3& v1, const Vec3& v2,
                      bool cullBackface=false) {
  Vec3 e1 = v1 - v0; Vec3 e2 = v2 - v0;
  Vec3 pvec = cross(d, e2);
  float det = dot(e1, pvec);
  if (cullBackface) { if (det < 1e-8f) return {false,0,0,0}; }
  else            { if (fabs(det) < 1e-8f) return {false,0,0,0}; }
  float invDet = 1.0f / det;
  Vec3 tvec = o - v0;
  float u = dot(tvec, pvec) * invDet;
  if (u < 0.0f || u > 1.0f) return {false,0,0,0};
  Vec3 qvec = cross(tvec, e1);
  float v = dot(d, qvec) * invDet;
  if (v < 0.0f || u + v > 1.0f) return {false,0,0,0};
  float t = dot(e2, qvec) * invDet;
  if (t <= 0.0f) return {false,0,0,0};
  return {true, t, u, v};
}