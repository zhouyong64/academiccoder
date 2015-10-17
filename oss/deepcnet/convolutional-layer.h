const int thisManyThreads = 512;

enum sigmoidType             {NOSIGMOID,   LOGISTIC,   RECTIFIEDLINEAR,   TANH,   SOFTMAX,   BLOCKY };
const char *sigmoidNames[] ={"NOSIGMOID", "LOGISTIC", "RECTIFIEDLINEAR", "TANH", "SOFTMAX", "BLOCKY"};
//   _____ _    _ _____            ______ ______    _____
//  / ____| |  | |  __ \   /\     |  ____/ /  _ \  |  __ \
// | |    | |  | | |  | | /  \    | |__ / /| |_) | | |__) | __ ___  _ __
// | |    | |  | | |  | |/ /\ \   |  __/ / |  _ <  |  ___/ '__/ _ \| '_ \
// | |____| |__| | |__| / ____ \  | | / /  | |_) | | |   | | | (_) | |_) |
//  \_____|\____/|_____/_/    \_\ |_|/_/   |____/  |_|   |_|  \___/| .__/
//                                                                 | |
//                                                                 |_|

//kMaxout==1 when applying a nonlinear function (applied after maxout anyway)
__global__ void dSigmoidLogistic(float* g, int count, int nOut) {
  for(int i=threadIdx.x; i<count; i+=thisManyThreads) {
    for (int k=0;k<nOut;k++) {
      g[i*nOut+k]=1/(1+expf(-g[i*nOut+k]));
    }
  }
}
__global__ void dSigmoidRectifiedLinear(float* g, int count, int nOut) {
  for(int i=threadIdx.x; i<count; i+=thisManyThreads) {
    for (int k=0;k<nOut;k++) {
      g[i*nOut+k]=(g[i*nOut+k]>0)?g[i*nOut+k]:0;
    }
  }
}
__global__ void dSigmoidBlocky(float* g, int count, int nOut) {
  for(int i=threadIdx.x; i<count; i+=thisManyThreads) {
    for (int k=0;k<nOut;k++) {
      g[i*nOut+k]=
        (g[i*nOut+k]>1 )?
        1 :
        (( g[i*nOut+k]< -1 )?
         -1 :
         g[i*nOut+k] );
    }
  }
}
__global__ void dSigmoidTanh(float* g, int count, int nOut) {
  for(int i=threadIdx.x; i<count; i+=thisManyThreads) {
    for (int k=0;k<nOut;k++) {
      g[i*nOut+k]=tanhf(g[i*nOut+k]);
    }
  }
}
__global__ void dSigmoidSoftmax(float* g, int count, int nOut) {
  for(int i=threadIdx.x; i<count; i+=thisManyThreads) {
    float acc=0.0f;
    float mx=0.0f;
    for (int k=0;k<nOut;k++)
      if (g[i*nOut+k]>mx) mx=g[i*nOut+k];
    for (int k=0;k<nOut;k++) {
      g[i*nOut+k]=expf(g[i*nOut+k]-mx);
      acc+=g[i*nOut+k];}
    for (int k=0;k<nOut;k++) {
      g[i*nOut+k]=g[i*nOut+k]/acc;
    }
  }
}


//SOFTMAX only occurs at the top layer;
//derivative contained in calculation of initial d_delta.
__global__ void dSigmoidBackpropLogistic(float* d, float* g, int count, int N) {
  for(int i=0; i<count; i++) {
    for (int j=threadIdx.x; j<N; j+=thisManyThreads)
      d[i*N+j]*=g[i*N+j]*(1-g[i*N+j]);
  }
}
__global__ void dSigmoidBackpropRectifiedLinear(float* d, float* g, int count, int N) {
  for(int i=0; i<count; i++) {
    for (int j=threadIdx.x; j<N; j+=thisManyThreads)
      d[i*N+j]*=((g[i*N+j]>0)?1:0);
  }
}
__global__ void dSigmoidBackpropBlocky(float* d, float* g, int count, int N) {
  for(int i=0; i<count; i++) {
    for (int j=threadIdx.x; j<N; j+=thisManyThreads)
      d[i*N+j]*=((g[i*N+j]>-1 && g[i*N+j] <1)?1:0);
  }
}
__global__ void dSigmoidBackpropTanh(float* d, float* g, int count, int N) {
  for(int i=0; i<count; i++) {
    for (int j=threadIdx.x; j<N; j+=thisManyThreads)
      d[i*N+j]*=(1+g[i*N+j])*(1-g[i*N+j]);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void dColumnSum
(float* matrix, float* target, int nRows, int nColumns) {
  for (int col=threadIdx.x;col<nColumns;col+=thisManyThreads)
    for (int row=0; row<nRows; row++)
      target[col]+=matrix[row*nColumns+col];}

////////////////////////////////////////////////////////////////////////////////////////////////
//Calculate softmaxProbability(i) - indicator(i=label)
// for i=0,1,...N-1 with N the number of character classes.
__global__ void dDerivativeOfCostWRTpreSoftmaxTopLevelWeights
(int batchSize, float* topDelta, float* topGrid,
 int* labels, int N) {
  for (int k=0;k<batchSize;k++) {
    for(int i=threadIdx.x;i<N;i+=thisManyThreads)
      topDelta[k*N+i]+=topGrid[k*N+i];
    if (threadIdx.x==0)
      topDelta[k*N+labels[k]]-=1;
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////
__global__ void dGradientDescent //momentum
(float* d_delta, float* d_momentum, float* d_weights, int N,
 float learningRate, float momentumDecayRate, float weightDecayRate) {
  for(int i = threadIdx.x;i<N;i+=thisManyThreads) {
    d_momentum[i]=d_momentum[i]*(1-momentumDecayRate)+d_delta[i]*momentumDecayRate;
    d_weights[i]-=learningRate*(d_momentum[i]+d_weights[i]*weightDecayRate);
    d_delta[i]=0;
  }
}
///////////////////////////////////////////////////////////////////////////////////////////////////
//   _____                      _       _   _                   _ _
//  / ____|                    | |     | | (_)                 | | |
// | |     ___  _ ____   _____ | |_   _| |_ _  ___  _ __   __ _| | |     __ _ _   _  ___ _ __
// | |    / _ \| '_ \ \ / / _ \| | | | | __| |/ _ \| '_ \ / _` | | |    / _` | | | |/ _ \ '__|
// | |___| (_) | | | \ V / (_) | | |_| | |_| | (_) | | | | (_| | | |___| (_| | |_| |  __/ |
//  \_____\___/|_| |_|\_/ \___/|_|\__,_|\__|_|\___/|_| |_|\__,_|_|______\__,_|\__, |\___|_|
//                                                                             __/ |
//                                                                            |___/
class ConvolutionalLayer {
public:
  int filterSize;
  int poolSize;
  int s0,s1,s2;
  int nIn;
  int nOut;
  float dropoutProbability;
  int kMaxout;
  sigmoidType sigmoid;
  int fs2;
  int ps2;
  vector<float> W;
  vector<float> B;
  float* d_W;
  float* d_B;
  float* d_momentumW;
  float* d_momentumB;

  void loadWeightsFromStream(ifstream &f) {
    W.resize(filterSize*filterSize*nIn*nOut*kMaxout);
    B.resize(nOut*kMaxout);
    f.read((char*)&W[0],sizeof(float)*W.size());
    f.read((char*)&B[0],sizeof(float)*B.size());
    h2dMemcopy<float>(&W[0],d_W,W.size());
    h2dMemcopy<float>(&B[0],d_B,B.size()); }
  void putWeightsToStream(ofstream &f)  {
    d2hMemcopy<float>(d_W,&W[0],W.size());
    d2hMemcopy<float>(d_B,&B[0],B.size());
    f.write((char*)&W[0],sizeof(float)*W.size());
    f.write((char*)&B[0],sizeof(float)*B.size()); }

  ConvolutionalLayer
  (int fs, int ps, int s0, int s1, int s2, int in, int out, sigmoidType sig, float dropoutProbability=0, int kMaxout=1) :
    filterSize(fs), poolSize(ps), s0(s0), s1(s1), s2(s2), nIn(in), nOut(out), sigmoid(sig), dropoutProbability(dropoutProbability), kMaxout(kMaxout) {
    RNG rng;
    fs2=filterSize*filterSize;
    ps2=poolSize*poolSize;
    float fanIn=nIn*fs2;
    float fanOut=nOut*fs2*1.0f/ps2;
    float scale=pow(6.0f/(fanIn+fanOut),0.5f);

    B.resize(nOut*kMaxout,0);
    d_B=d_allocateArrayFromVector<float>(B,__FILE__,__LINE__);
    d_momentumB=d_allocateArrayZeroed<float>(B.size(),__FILE__,__LINE__);

    W.resize(filterSize*filterSize*nIn*nOut*kMaxout);
    for (int i=0; i<W.size(); i++)
      W[i]=rng.uniform(-scale,scale);
    d_W=d_allocateArrayFromVector<float>(W,__FILE__,__LINE__);
    d_momentumW=d_allocateArrayZeroed<float>(W.size(),__FILE__,__LINE__);
  }


  void applyDerivatives(float* d_deltaW, float* d_deltaB, float learningRate, float momentumDecayRate, float weightDecayRate) {
    dGradientDescent<<<1,thisManyThreads>>>(d_deltaW, d_momentumW, d_W, W.size(), learningRate, momentumDecayRate, weightDecayRate);
    dGradientDescent<<<1,thisManyThreads>>>(d_deltaB, d_momentumB, d_B, B.size(), learningRate, momentumDecayRate, weightDecayRate);
  }
};
