class FloatPoint;
class EDfield {
  int resolution;
  int scale; //aim to operate on the square [-scale/2,scale/2]^2
  vector<vector<float> > rf;
public:
  void convolve_gaussian(vector<float> &a, float sigma, int n);
  EDfield (int resolution, int scale, float sigma, float amp);
  EDfield() {}
  void stretch(FloatPoint &p);
};

class FloatPoint {
public:
  float x;
  float y;
  FloatPoint() {}
  FloatPoint(float x, float y) : x(x), y(y) {}
  void flip_horizontal() {
    x=-x;
  }
  void stretch_x(float alpha) {
    x*=(1+alpha);
  }
  void stretch_y(float alpha) {
    y*=(1+alpha);
  }
  void rotate(float angle) {
    float c=cos(angle);
    float s=sin(angle);
    float xx=+x*c+y*s;
    float yy=-x*s+y*c;
    x=xx;
    y=yy;
  }
  void slant_x(float alpha) {
    y+=alpha*x;
  }
  void slant_y(float alpha) {
    x+=alpha*y;
  }
  void stretch4(float cxx, float cxy, float cyx, float cyy) {
    float tx=x;
    float ty=y;
    x=(1+cxx)*tx+cxy*ty;
    y=(1+cyy)*ty+cyx*tx;
  }
  void stretch(EDfield& f) {
    f.stretch(*this);
  }
};

template<class T> T bilinearInterpolation(FloatPoint& p, T* array, int xSize, int ySize) {
  //Treat array as a rectangle [0,xSize]*[0,ySize]. Associate each value of the array with the centre of the corresponding unit square.
  int ix=floor(p.x-0.5);
  int iy=floor(p.y-0.5);
  float rx=p.x-0.5-ix;
  float ry=p.y-0.5-iy;
  T c00=0, c01=0, c10=0, c11=0;
  if (ix>=0 && ix<xSize && iy>=0 && iy<ySize) c00=array[ix+iy*xSize];
  ix++;
  if (ix>=0 && ix<xSize && iy>=0 && iy<ySize) c10=array[ix+iy*xSize];
  iy++;
  if (ix>=0 && ix<xSize && iy>=0 && iy<ySize) c11=array[ix+iy*xSize];
  ix--;
  if (ix>=0 && ix<xSize && iy>=0 && iy<ySize) c01=array[ix+iy*xSize];
  return (1-rx)*(1-ry)*c00+(1-rx)*ry*c01+rx*(1-ry)*c10+rx*ry*c11;
}

template<class T> T bilinearInterpolationScaled(FloatPoint p, T* array, int xSize, int ySize, float xMin, float yMin, float xMax,  float yMax) {
  p.x=(p.x-xMin)*xSize/(xMax-xMin);
  p.y=(p.y-yMin)*ySize/(yMax-yMin);
  return bilinearInterpolation<T>(p,array,xSize,ySize);
}


void EDfield::convolve_gaussian(vector<float> &a, float sigma, int n) { //inplace
  vector<float> b(n*n,0);
  for (int i=0;i<n;i++) {
    for (int j=0; j<n; j++) {
      for (int k=max<int>(0,j-3*sigma);k<=min<int>(n-1,j+3*sigma);k++)
        b[i*n+j]+=a[i*n+k]*exp(-(k-j)*(k-j)/2.0f/sigma/sigma)/sigma/0.82; //Gives EDfield components mean absolute magnitude amp.
    }
  }
  for (int i=0;i<n;i++) {
    for (int j=0; j<n; j++) {
      a[i*n+j]=0;
      for (int k=max<int>(0,i-3*sigma);k<=min<int>(n-1,i+3*sigma);k++)
        a[i*n+j]+=b[k*n+j]*exp(-(k-i)*(k-i)/2.0f/sigma/sigma);
    }
  }
}
EDfield::EDfield (int resolution, int scale, float sigma, float amp) :
  resolution(resolution), scale(scale) {
  RNG rng;
  rf.resize(2);
  for (int k=0; k<2; k++) {
    rf[k].resize(resolution*resolution);
    for (int i=0;i<resolution;i++) {
      for (int j=0; j<resolution; j++) {
        rf[k][i*resolution+j]=rng.uniform(-amp,amp);
      }
    }
    convolve_gaussian(rf[k], sigma, resolution);
  }
}
void EDfield::stretch(FloatPoint &p) {
  float dx=bilinearInterpolationScaled<float>(p, &rf[0][0], resolution, resolution, -0.6*scale, -0.6*scale, 0.6*scale, 0.6*scale);
  float dy=bilinearInterpolationScaled<float>(p, &rf[1][1], resolution, resolution, -0.6*scale, -0.6*scale, 0.6*scale, 0.6*scale);
  p.x+=dx;
  p.y+=dy;
}

//Use to precalculate a large number of EDfield objects
class EDfields {
  int resolution, scale;
  float sigma, amp;
  boost::thread_group tg;
  void t(int j) {
    for (int i=j;i<edf.size();i+=6)
      edf[i]=EDfield(resolution,scale,sigma,amp);
  }
public:
  vector<EDfield> edf;
  EDfields(int n, int resolution, int scale, float sigma, float amp)
    : resolution(resolution), scale(scale), sigma(sigma), amp(amp) {
    edf.resize(n);
    for (int i=0; i<6; i++)
      tg.add_thread(new boost::thread(boost::bind(&EDfields::t,this,i)));
    tg.join_all();
  }
};
