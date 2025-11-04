#include 
using Mat4 = float[16];

// perspective: fovY radians, aspect = w/h, near n, far f
void perspective(Mat4 M, float fovY, float aspect, float n, float f) {
  float y = 1.0f / tanf(fovY * 0.5f);
  float x = y / aspect;
  // column-major OpenGL-style
  M[0]=x; M[4]=0; M[8]=0;             M[12]=0;
  M[1]=0; M[5]=y; M[9]=0;             M[13]=0;
  M[2]=0; M[6]=0; M[10]=(f+n)/(n-f);  M[14]=(2*f*n)/(n-f);
  M[3]=0; M[7]=0; M[11]=-1;           M[15]=0;
}

// orthographic: left/right, bottom/top, near/far
void ortho(Mat4 M, float l, float r, float b, float t, float n, float f) {
  M[0]=2.0f/(r-l); M[4]=0;          M[8]=0;          M[12]=-(r+l)/(r-l);
  M[1]=0;          M[5]=2.0f/(t-b); M[9]=0;          M[13]=-(t+b)/(t-b);
  M[2]=0;          M[6]=0;          M[10]=2.0f/(n-f);M[14]=-(f+n)/(f-n);
  M[3]=0;          M[7]=0;          M[11]=0;         M[15]=1;
}