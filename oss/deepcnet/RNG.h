//#include<sys/time.h>
boost::mutex RNGseedGeneratorMutex;
boost::mt19937 RNGseedGenerator;

class RNG {
  //  timespec ts;
public:
  boost::mt19937 gen;
  RNG() {
    RNGseedGeneratorMutex.lock();
    gen.seed(RNGseedGenerator()+startEpoch);
    RNGseedGeneratorMutex.unlock();
    // clock_gettime(CLOCK_REALTIME, &ts);
    // gen.seed(ts.tv_nsec);
  }
  int randint(int n) {
    return gen()%n;
  }
  float uniform(float a=0, float b=1) {
    unsigned int k=gen();
    return a+(b-a)*k/4294967296.0;
  }
  int bernoulli(float p) {
    if (uniform()<p)
      return 1;
    else
      return 0;
  }
  template <typename T>
  int index(vector<T> &v) {
    if (v.size()==0) cout << "RNG::index called for empty vector!";
    return gen()%v.size();
  }
  vector<int> NchooseM(int n, int m) {
    vector<int> ret(m);
    int ctr=m;
    for(int i=0;i<n;i++)
      if (rand()<ctr*1.0/(n-i)) ret[--ctr]=i;
    return ret;
  }
};
