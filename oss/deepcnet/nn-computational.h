template <typename t> __global__ void dReplicateArray(t* d_src, t* d_dest, int size, int nCopies) {
  for (int i=threadIdx.x;i<nCopies;i+=thisManyThreads) {
    for (int j=0;j<size;j++) d_dest[i*size+j]=d_src[j];
  }
}
///////////////////////////////////////////////////////////////////////////////////////////////
__global__ void dPropForwardToMatrixMultiplyInput
(float* d_features, float* d_sgemm, int* rules, int count, int nIn, int fs2) {
  for (int i = threadIdx.x;i<count*fs2;i+=thisManyThreads) {
    for (int k=0;k<nIn;k++) {
      d_sgemm[i*nIn+k]=d_features[rules[i]*nIn+k]; }
  }
}

__global__ void dPropBackwardFromMatrixMultiplyOutput
(float* d_deltaGrid, float* d_sgemm, int* rules, int count, int nIn, int fs2) {
  for (int i = threadIdx.x;i<count*fs2;i+=thisManyThreads) {
    for (int k=0;k<nIn;k++) {
      atomicAdd(d_deltaGrid+rules[i]*nIn+k,d_sgemm[i*nIn+k]); }
  }
}

__global__ void dDropoutFeatures
(float* d_features, int* d_featureSampleNumbers,
 int count, int nIn, float* d_featureWeight) {
  for (int i=threadIdx.x; i<count*nIn; i+=thisManyThreads) {
    int item=d_featureSampleNumbers[i/nIn];
    d_features[i]*=d_featureWeight[item*nIn+(i%nIn)];
  }
}

__global__ void dClassify
(float* d_features, int* d_predictions, int batchSize, int nOut) {
  for (int i = threadIdx.x;i<batchSize;i+=thisManyThreads) {
    int prediction=0;
    float maxP=d_features[i*nOut];
    for (int k=1;k<nOut;k++) {
      if (d_features[i*nOut+k]>maxP) {
        prediction=k;
        maxP=d_features[i*nOut+k];
      }
    }
    d_predictions[i]=prediction;
  }
}
///////////////////////////////////////////////////////////////////////////////////////////////
//count is a multiple of nOut
__global__ void dMaxout
(float* g1a, float* g1b, int count, int kMaxout, unsigned char* d_choice) {
  for (int i =  threadIdx.x;i<count; i+=thisManyThreads) {
    g1b[i]=g1a[i*kMaxout];
    d_choice[i]=0;
    for (int j=1;j<kMaxout;j++)
      if (g1b[i]<g1a[i*kMaxout+j]) {
        g1b[i]=g1a[i*kMaxout+j];
        d_choice[i]=j;
      }
  }
}

__global__ void dMaxoutBackprop
(float* d1a, float* d1b, int count, int kMaxout, unsigned char* d_choice) {
  for (int i=threadIdx.x; i<count; i+=thisManyThreads)
    d1a[i*kMaxout+d_choice[i]]=d1b[i];
}


///////////////////////////////////////////////////////////////////////////////////////////////
__global__ void dMaxPool
(float* g1, float* g2, int* rules, int count, int ps2, int nOut, unsigned char* d_choice) {
  for (int i =  threadIdx.x;i<count; i+=thisManyThreads) {
    for (int j = 0;j<nOut;j++) {
      g2[i*nOut+j]=g1[rules[i*ps2+0]*nOut+j];
      d_choice[i*nOut+j]=0;
      for (int k=1;k<ps2;k++)
        if (g2[i*nOut+j]<g1[rules[i*ps2+k]*nOut+j]) {
          g2[i*nOut+j]=g1[rules[i*ps2+k]*nOut+j];
          d_choice[i*nOut+j]=k;
        }
    }
  }
}

__global__ void dMaxPoolBackprop
(int* rules, float* d1, float* d2, int count, int ps2, int nOut, unsigned char* d_choice) {
  for (int i=threadIdx.x; i<count; i+=thisManyThreads)
    for (int j=0; j<nOut; j++)
      d1[rules[i*ps2+d_choice[i*nOut+j]]*nOut+j]=d2[i*nOut+j];
}

///////////////////////////////////////////////////////////////////////////////////////////////
//   _____                             _             _
//  / ____|                           | |           (_)
// | (___  _ __   __ _ _ __ ___  ___  | | ___   __ _ _  ___
//  \___ \| '_ \ / _` | '__/ __|/ _ \ | |/ _ \ / _` | |/ __|
//  ____) | |_) | (_| | |  \__ \  __/ | | (_) | (_| | | (__
// |_____/| .__/ \__,_|_|  |___/\___| |_|\___/ \__, |_|\___|
//        | |                                   __/ |
//        |_|                                  |___/

class ConvolutionalComputationalLayerInterface {
public:
  float* d_features;
  batchType type;
  int batchSize;
  vector<int> featureSampleNumbers;
  vector<int> backgroundNullVectorNumbers;
  vector<vector<int> > grids;
  int count;
  ConvolutionalComputationalLayerInterface () : count(0) {}
};


class ConvolutionalComputationalLayerBase {
public:
  RNG rng;
  ConvolutionalLayer& L;
  int level;
  float* d_deltaW;//     (d cost)/(d W)
  float* d_deltaB;//     (d cost)/(d B)
  ConvolutionalComputationalLayerInterface &input;
  ConvolutionalComputationalLayerInterface middle;
  ConvolutionalComputationalLayerInterface output;
  unsigned char* d_maxPoolChoice;
  float* d_featuresToMaxout;
  unsigned char* d_maxoutChoice;
  int* d_featureSampleNumbers; //Used for
  float* d_featureWeight;        //dropout

  ConvolutionalComputationalLayerBase
  (ConvolutionalLayer &L, int level, ConvolutionalComputationalLayerInterface &input) :
    L(L), level(level), input(input) {}
  virtual void initialize() {}
  virtual void copyDataToGPU() {}
  virtual void forwards() {}
  virtual void backwards(float* &d_delta) {}
  virtual void applyDerivatives(float learningRate, float momentumDecayRate, float weightDecayRate) {}
  virtual void cleanUp() {}
};

class ConvolutionalComputationalLayer: public ConvolutionalComputationalLayerBase {
public:
  float* d_sgemm;
  int* d_cRules;
  int* d_pRules;
  vector<int> cRules;
  vector<int> pRules;

  ConvolutionalComputationalLayer
  (ConvolutionalLayer &L, int level, ConvolutionalComputationalLayerInterface &input) :
    ConvolutionalComputationalLayerBase(L,level,input) {
    d_deltaB=d_allocateArrayZeroed<float>(L.B.size(),__FILE__,__LINE__);
    d_deltaW=d_allocateArrayZeroed<float>(L.W.size(),__FILE__,__LINE__);
  }

  ~ConvolutionalComputationalLayer() {
    cudaFree(d_deltaW);
    cudaFree(d_deltaB); }


  bool nullVectorSurvivesConvolution(int item) {
    for (int i=0; i<L.s1;i++) {
      for (int j=0; j<L.s1;j++) {
        int ctr=0;
        for (int ii=0;ii<L.filterSize;ii++) {
          for (int jj=0;jj<L.filterSize;jj++) {
            int n0=(i+ii)*L.s0+(j+jj);
            if (input.grids[item][n0]==input.backgroundNullVectorNumbers[item])
              ctr++;
          }
        }
        if (ctr==L.fs2) return true;
      }
    }
    return false;
  }
  void gridConvolutionRules(vector<int>& g0, vector<int>& g1, int bg0, int bg1) {
    if (bg1 != -1)
      for (int i=0; i<L.fs2; i++)
        cRules.push_back(bg0);
    g1.resize(L.s1*L.s1,bg1);
    for (int i=0;i<L.s1;i++) {
      for (int j=0;j<L.s1;j++) {
        int n1=i*L.s1+j;
        for (int ii=0;ii<L.filterSize;ii++) {
          for (int jj=0;jj<L.filterSize;jj++) {
            int n0=(i+ii)*L.s0+(j+jj);
            if (g0[n0]!=bg0 && g1[n1]==bg1)
              g1[n1]=middle.count++;
          }
        }
        if (g1[n1]!=bg1) {
          for (int ii=0;ii<L.filterSize;ii++) {
            for (int jj=0;jj<L.filterSize;jj++) {
              int n0=(i+ii)*L.s0+(j+jj);
              cRules.push_back(g0[n0]);
            }
          }
        }
      }
    }
  }
  bool nullVectorSurvivesPooling(int item) {
    for (int i=0; i<L.s2;i++) {
      for (int j=0; j<L.s2;j++) {
        int ctr=0;
        for (int ii=0;ii<L.poolSize;ii++) {
          for (int jj=0;jj<L.poolSize;jj++) {
            int n1=(i*L.poolSize+ii)*L.s1+(j*L.poolSize+jj);
            //            cout << level << " " << n1 << " " << middle.grids[item][n1] << " " << middle.backgroundNullVectorNumbers[item]<< " " << ctr << endl;
            if (middle.grids[item][n1]==middle.backgroundNullVectorNumbers[item])
              ctr++;
          }
        }
        if (ctr==L.ps2) return true;
      }
    }
    return false;
  }
  void gridPoolingRules(vector<int>& g1, vector<int>& g2, int bg1, int bg2) {
    if (bg2 != -1)
      for (int i=0; i<L.ps2; i++)
        pRules.push_back(bg1);
    g2.resize(L.s2*L.s2,bg2);
    for (int i=0;i<L.s2;i++) {
      for (int j=0;j<L.s2;j++) {
        int n2=i*L.s2+j;
        for (int ii=0;ii<L.poolSize;ii++) {
          for (int jj=0;jj<L.poolSize;jj++) {
            int n1=(i*L.poolSize+ii)*L.s1+(j*L.poolSize+jj);
            if (g1[n1]!=bg1 && g2[n2]==bg2)
              g2[n2]=output.count++;
          }
        }
        if (g2[n2]!=bg2) {
          for (int ii=0;ii<L.poolSize;ii++) {
            for (int jj=0;jj<L.poolSize;jj++) {
              int n1=(i*L.poolSize+ii)*L.s1+(j*L.poolSize+jj);
              pRules.push_back(g1[n1]);
            }
          }
        }
      }
    }
  }
  void initialize() {
    cRules.reserve(L.fs2*input.batchSize*L.s1*L.s1);
    pRules.reserve(L.ps2*input.batchSize*L.s2*L.s2);
    output.type=middle.type=input.type;
    output.batchSize=middle.batchSize=input.batchSize; //All the same
    middle.backgroundNullVectorNumbers.resize(middle.batchSize);
    output.backgroundNullVectorNumbers.resize(output.batchSize);
    middle.grids.resize(middle.batchSize);
    output.grids.resize(output.batchSize);
    for (int item=0; item<input.batchSize; item++) {
      if (nullVectorSurvivesConvolution(item))
        middle.backgroundNullVectorNumbers[item]=middle.count++;
      else
        middle.backgroundNullVectorNumbers[item]=-1;
      gridConvolutionRules(input.grids[item],
                           middle.grids[item],
                           input.backgroundNullVectorNumbers[item],
                           middle.backgroundNullVectorNumbers[item]);
      while (middle.featureSampleNumbers.size() < middle.count)
        middle.featureSampleNumbers.push_back(item);

      if (nullVectorSurvivesPooling(item))
        output.backgroundNullVectorNumbers[item]=output.count++;
      else
        output.backgroundNullVectorNumbers[item]=-1;
      gridPoolingRules(middle.grids[item],
                       output.grids[item],
                       middle.backgroundNullVectorNumbers[item],
                       output.backgroundNullVectorNumbers[item]);
      while (output.featureSampleNumbers.size() < output.count)
        output.featureSampleNumbers.push_back(item);
    }
    // size_t Free, Total;
    // cudaError_t result=cudaMemGetInfo(&Free, &Total);
    // cout << Free<< " " << Total<<endl;
    // cout << cRules.size()/L.fs2 << " "  << pRules.size()/L.ps2<<endl;
    // cout <<input.count <<" " << middle.count << " " << output.count <<endl;
  }

  void copyDataToGPU() {
    d_cRules=d_allocateArrayFromVector<int>(cRules,__FILE__,__LINE__);
    d_pRules=d_allocateArrayFromVector<int>(pRules,__FILE__,__LINE__);
    d_sgemm=d_allocateArray<float>(middle.count*L.fs2*L.nIn,__FILE__,__LINE__);
    d_featuresToMaxout=d_allocateArray<float>(middle.count*L.nOut*L.kMaxout,__FILE__,__LINE__);
    if (L.kMaxout>1) {
      d_maxoutChoice=d_allocateArray<unsigned char>(middle.count*L.nOut,__FILE__,__LINE__);
      middle.d_features=d_allocateArray<float>(middle.count*L.nOut,__FILE__,__LINE__);
    } else
      middle.d_features=d_featuresToMaxout;
    if (L.poolSize>1) {
      d_maxPoolChoice=d_allocateArray<unsigned char>(output.count*L.nOut,__FILE__,__LINE__);
      output.d_features=d_allocateArray<float>(output.count*L.nOut,__FILE__,__LINE__);
    } else
      output.d_features=middle.d_features;
  }

  void forwards() {
    //Dropout
    if (L.dropoutProbability>0) {
      vector<float> featureWeights(input.batchSize*L.nIn,1-L.dropoutProbability);
      if (input.type==TRAINBATCH)
        for (int i=0;i<featureWeights.size(); i++)
          featureWeights[i]=rng.bernoulli(1-L.dropoutProbability);
      d_featureWeight=d_allocateArrayFromVector<float>(featureWeights,__FILE__,__LINE__);
      d_featureSampleNumbers=d_allocateArrayFromVector<int>(input.featureSampleNumbers,__FILE__,__LINE__);
      dDropoutFeatures<<<1,thisManyThreads>>>
        (input.d_features, d_featureSampleNumbers, input.count, L.nIn, d_featureWeight);
    }

    //convolution
    dPropForwardToMatrixMultiplyInput<<<1,thisManyThreads>>>
      (input.d_features, d_sgemm, d_cRules, middle.count,L.nIn,L.fs2);

    dReplicateArray<float><<<1,thisManyThreads>>>       //set bias
      (L.d_B, d_featuresToMaxout, L.nOut*L.kMaxout,middle.count);

    d_rowMajorSGEMM_alphaAB_betaC(cublasHandle,
                                  d_sgemm, L.d_W, d_featuresToMaxout,
                                  middle.count, L.fs2*L.nIn, L.nOut*L.kMaxout,
                                  1.0f, 1.0f);
    //Maxout
    if (L.kMaxout>1)
      dMaxout<<<1,thisManyThreads>>>
        (d_featuresToMaxout, middle.d_features, middle.count*L.nOut, L.kMaxout, d_maxoutChoice);

    //maxpooling
    if (L.poolSize>1)
      dMaxPool<<<1,thisManyThreads>>>
        (middle.d_features,output.d_features,d_pRules,output.count,L.ps2,L.nOut,d_maxPoolChoice);
    switch(L.sigmoid) {
    case RECTIFIEDLINEAR: dSigmoidRectifiedLinear<<<1,thisManyThreads>>> (output.d_features,output.count,L.nOut); break;
    case LOGISTIC:     dSigmoidLogistic    <<<1,thisManyThreads>>> (output.d_features,output.count,L.nOut); break;
    case BLOCKY:       dSigmoidBlocky      <<<1,thisManyThreads>>> (output.d_features,output.count,L.nOut); break;
    case TANH:         dSigmoidTanh        <<<1,thisManyThreads>>> (output.d_features,output.count,L.nOut); break;
    case SOFTMAX:      dSigmoidSoftmax     <<<1,thisManyThreads>>> (output.d_features,output.count,L.nOut); break;
    }
  }

  void backwards(float* &d_delta)  {
    //  cout << "B" << L.nIn<<endl;
    //adjust d_delta post to pre sigmoid using outputGrid. Does nothing to top softmax layer.
    switch(L.sigmoid) {
    case RECTIFIEDLINEAR: dSigmoidBackpropRectifiedLinear<<<1,thisManyThreads>>> (d_delta, output.d_features, output.count, L.nOut); break;
    case LOGISTIC:     dSigmoidBackpropLogistic    <<<1,thisManyThreads>>> (d_delta, output.d_features, output.count, L.nOut); break;
    case BLOCKY:       dSigmoidBackpropBlocky      <<<1,thisManyThreads>>> (d_delta, output.d_features, output.count, L.nOut); break;
    case TANH:         dSigmoidBackpropTanh        <<<1,thisManyThreads>>> (d_delta, output.d_features, output.count, L.nOut); break;
    }

    //Undo max-pooling.
    if (L.poolSize>1) {
      float* d_delta_=d_allocateArrayZeroed<float>(middle.count*L.nOut,__FILE__,__LINE__);
      dMaxPoolBackprop<<<1,thisManyThreads>>>
        (d_pRules, d_delta_, d_delta, output.count, L.ps2, L.nOut, d_maxPoolChoice);
      cudaFree(d_delta);
      d_delta=d_delta_;
    }
    //Undo maxout
    if (L.kMaxout>1) {
      float* d_delta_=d_allocateArrayZeroed<float>(middle.count*L.nOut*L.kMaxout,__FILE__,__LINE__);
      dMaxoutBackprop<<<1,thisManyThreads>>>
        (d_delta_, d_delta, middle.count*L.nOut, L.kMaxout, d_maxoutChoice);
      cudaFree(d_delta);
      d_delta=d_delta_;
    }

    //calculate d_deltaB
    dColumnSum<<<1,thisManyThreads>>>
      (d_delta,d_deltaB,middle.count,L.nOut*L.kMaxout);

    //Calculate delta_W
    d_rowMajorSGEMM_alphaAtB_betaC(cublasHandle,
                                   d_sgemm, d_delta, d_deltaW,
                                   L.nIn*L.fs2, middle.count, L.nOut*L.kMaxout,
                                   1.0, 0.0);

    if (level>0) {
      //Undo convolution
      float* d_deltaSgemm;
      d_deltaSgemm=d_allocateArray<float>(middle.count*L.nIn*L.fs2,__FILE__,__LINE__);
      d_rowMajorSGEMM_alphaABt_betaC(cublasHandle,
                                     d_delta, L.d_W, d_deltaSgemm,
                                     middle.count,L.nOut*L.kMaxout,L.nIn*L.fs2,
                                     1.0, 0.0);
      cudaFree(d_delta);
      d_delta=d_allocateArrayZeroed<float>(input.count*L.nIn,__FILE__,__LINE__);
      dPropBackwardFromMatrixMultiplyOutput<<<1,thisManyThreads>>>
        (d_delta, d_deltaSgemm,  d_cRules, middle.count, L.nIn, L.fs2);
      cudaFree(d_deltaSgemm);
      //Dropout
      if (L.dropoutProbability>0) {
        dDropoutFeatures<<<1,thisManyThreads>>>
          (d_delta, d_featureSampleNumbers, input.count, L.nIn, d_featureWeight);
      }
    } else
      cudaFree(d_delta);
  }
  void applyDerivatives(float learningRate, float momentumDecayRate, float weightDecayRate)
  {
    L.applyDerivatives(d_deltaW, d_deltaB, learningRate, momentumDecayRate, weightDecayRate);
  }
  void cleanUp()
  {
    cudaFree(d_cRules);
    cudaFree(d_pRules);
    cudaFree(d_sgemm);
    cRules.clear();
    pRules.clear();
    cudaFree(d_featuresToMaxout);
    if (L.kMaxout>1) {
      cudaFree(d_maxoutChoice);
      cudaFree(middle.d_features);
    }
    if (L.poolSize>1) {
      cudaFree(d_maxPoolChoice);
      cudaFree(output.d_features);
    }
    middle.featureSampleNumbers.clear();
    middle.backgroundNullVectorNumbers.clear();
    middle.grids.clear();
    middle.count=0;
    output.featureSampleNumbers.clear();
    output.backgroundNullVectorNumbers.clear();
    output.grids.clear();
    output.count=0;
    if (L.dropoutProbability>0) {
      cudaFree(d_featureWeight);
      cudaFree(d_featureSampleNumbers);
    }
  }
};
