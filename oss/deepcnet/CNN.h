#include <algorithm>
#include <assert.h>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <unistd.h>
#include <vector>
#include <boost/assign/list_of.hpp>
#include <boost/bind.hpp>
#include <boost/format.hpp>
#include <boost/random.hpp>
#include <boost/thread.hpp>
using namespace std;
using namespace boost::assign;
#include "RNG.h"

enum batchType {TRAINBATCH, TESTBATCH, UNLABELLEDBATCH};
class SparseCnnInterface {
public:
  batchType type; //TRAINBATCH, TESTBATCH or UNLABELLEDBATCH
  int batchSize;
  int count; // Number of feature vectors, _including_ zero-vectors
  vector<int> featureSampleNumbers; //length count, numbers from 0 to batchSize-1.
  vector<float> features; // size count*nInputFeatures,
  vector<int> labels; // only used if labels are known, i.e. during training and testing

  // Each vector<int> represents an NxN array
  vector<vector<int> > grids;
  vector<int> backgroundNullVectorNumbers;
  int nMistakes;
  vector<vector<int> > topGuesses; //used when labels are unknown; find 10 best matches

  SparseCnnInterface (batchType type) : type(type), batchSize(0), count(0), nMistakes(0) {}
};

class Picture {
public:
  virtual void codifyInputData (SparseCnnInterface &interface)=0;
  virtual Picture* distort () {return this;}
  int label; //-1 for unknown
  virtual ~Picture() {}
};

vector<Picture*> trainCharacters;
vector<Picture*> testCharacters;
void loadData(); //Application specific loading mechanism
void replaceTestSetWithValidationSet(float p = 0.8) {
  RNG rng;
  rng.gen.seed(0);
  while (testCharacters.size()>0) { //Delete the test set
    delete testCharacters.back();
    testCharacters.pop_back();
  }
  vector<Picture*> c(trainCharacters); //Split the training set into
  trainCharacters.clear();             //training and validation sets.
  while (c.size()>0) {
    if (rng.uniform()<p)
      trainCharacters.push_back(c.back());
    else
      testCharacters.push_back(c.back());
    c.pop_back();
  }
  cout << "Replacing test set with validation set.\n";
}
void smallerTestSet(float p = 0.03) {
  RNG rng;
  rng.gen.seed(0);
  vector<Picture*> c(testCharacters);
  testCharacters.clear();
  while (c.size()>0) {
    if (rng.uniform()<p)
      testCharacters.push_back(c.back());
    else
      delete c.back();
    c.pop_back();
  }
  cout << "Reducing test set size.\n";
}


vector<float> regularizingConstants;
void calculateRegularizingConstants(int nInputFeatures) {
  cout << "Using " << trainCharacters.size() << " training samples to calculate regularizing constants." << endl;
  RNG rng;
  SparseCnnInterface interface(TRAINBATCH);
  regularizingConstants.resize(nInputFeatures,1.0f);//Assume initially empty.
  for (int i=0;i<10000;i++)
    trainCharacters[rng.index(trainCharacters)]->codifyInputData(interface);
  for (int i=0; i<nInputFeatures; i++) {
    regularizingConstants[i]=0;
    for (int j=0; j<interface.count; j++)
      regularizingConstants[i]=
        max(abs(interface.features[i+j*nInputFeatures]),
            regularizingConstants[i]);
  }
  cout << "Regularizing constants: ";
  for (int i=0; i<nInputFeatures; i++)
    cout << regularizingConstants[i] << " ";
  cout << endl;
}

#include "cuda.h"
#include <cublas_v2.h>
#include "device-memory-functions.h"
#include "convolutional-layer.h"
boost::mutex CNNmutex;
cublasHandle_t cublasHandle;
class CNN {
public:
  vector<ConvolutionalLayer> L;
  float learningRate;
  float momentumDecayRate;
  float weightDecayRate;
  deque<float> trainError;
  deque<float> testError;
  int nInputFeatures;
  int nInputSpatialSize;
  int epoch;

  void saveWeights() {
    char filename[100];
    sprintf(filename,weightFileNameFormat,epoch);
    ofstream f;
    f.open(filename,ios::out | ios::binary);
    for (int i=0; i<L.size(); i++)
      L[i].putWeightsToStream(f);
    f.write((char*)&regularizingConstants[0],sizeof(float)*nInputFeatures);
    f.close();
  }
  void loadWeights() {
    char filename[100];
    sprintf(filename,weightFileNameFormat,epoch);
    ifstream f;
    f.open(filename,ios::in | ios::binary);
    if (!f) {
      cout <<"Cannot find " << filename << endl;
      exit(EXIT_FAILURE);
    }
    cout << "Loading network parameters from " << filename << endl;
    for (int i=0; i<L.size(); i++)
      L[i].loadWeightsFromStream(f);
    regularizingConstants.reserve(nInputFeatures);
    f.read((char*)&regularizingConstants[0],sizeof(float)*nInputFeatures);
    f.close();
  }
  CNN(int nInputFeatures, int nInputSpatialSize, float learningRate, float momentumDecayRate, float weightDecayRate, int epoch) :
    nInputFeatures(nInputFeatures),
    nInputSpatialSize(nInputSpatialSize),
    learningRate(learningRate),
    momentumDecayRate(momentumDecayRate),
    weightDecayRate(weightDecayRate),
    epoch(epoch) {
    cublasStatus_t ret = cublasCreate(&cublasHandle);
    if (ret != CUBLAS_STATUS_SUCCESS) {
      cout << "cublasCreate returned error code" << ret << endl;
      exit(EXIT_FAILURE);
    }
    loadData();
#ifdef VALIDATION
    replaceTestSetWithValidationSet();
#endif
#ifdef SMALLERTESTSET
    smallerTestSet();
#endif
    cout << "size0\t#in\tdropout\tFilter\tsize1\tkMaxout\tPool\tsize2\t#out\tFunction\n";
  }
  ~CNN() {
    cublasDestroy(cublasHandle);
    for (int i=0; i<trainCharacters.size(); i++)
      delete trainCharacters[i];
    trainCharacters.resize(0);
    for (int i=0; i<testCharacters.size(); i++)
      delete testCharacters[i];
    testCharacters.resize(0);
  }
  void addLayer(int filterSize, int poolSize, int nFilters, sigmoidType sigmoidFunction, float dropoutProbability=0, int kMaxout=1) {
    if (kMaxout>1)
      sigmoidFunction=NOSIGMOID;
    int s0, nIn;
    if (L.size()==0) {
      s0=nInputSpatialSize;
      nIn=nInputFeatures;
    } else {
      s0=L.back().s2;
      nIn=L.back().nOut;
    }
    if (filterSize>s0) {
      cout << "filterSize is too big for this layer!"<<endl;
      exit(EXIT_FAILURE);
    }
    if ((s0-filterSize+1)%poolSize!=0) {
      cout << "poolSize does not divide the size of the output of the filters for this layer!"<<endl;
      exit(EXIT_FAILURE);
    }
    L.push_back(ConvolutionalLayer(filterSize,
                                   poolSize,
                                   s0, s0-filterSize+1, (s0-filterSize+1)/poolSize,
                                   nIn, nFilters,
                                   sigmoidFunction,
                                   dropoutProbability,
                                   kMaxout));
    cout << L.back().s0 << "\t"
         << L.back().nIn << "\t"
         << L.back().dropoutProbability << "\t"
         << L.back().filterSize << "\t"
         << L.back().s1 << "\t"
         << L.back().kMaxout << "\t"
         << L.back().poolSize << "\t"
         << L.back().s2 << "\t"
         << L.back().nOut << "\t"
         << sigmoidNames[sigmoidFunction] << "\n";
  }
  void initialize() {
    if (epoch>0)
      loadWeights();
    else {
      cout << "Initialized network parameters using the uniform distribution." << endl;
      calculateRegularizingConstants(nInputFeatures);
    }
  }
};

const sigmoidType baseSigmoidType=RECTIFIEDLINEAR;
class DeepCNet : public CNN {
public:
  DeepCNet(int l, int k, int nInputFeatures, int nOutputClasses, float learningRate,
           float momentumDecayRate, float weightDecayRate, int epoch,
           vector<float> dropoutProbabilities = vector<float>(),
           vector<int> kMaxouts = vector<int>()) :
    CNN(nInputFeatures, 3*(1<<l), learningRate, momentumDecayRate, weightDecayRate, epoch) {
    if (scale_N!=nInputSpatialSize) {
      cout << "scale_N should be " << 3*(1<<l) << endl;
      exit(EXIT_FAILURE);
    }
    if (dropoutProbabilities.empty()) dropoutProbabilities.resize(l+2,0);
    if (dropoutProbabilities.size()!=l+2) {
      cout << "Need " << l+2<< " dropout probabilities." << endl;
      exit(EXIT_FAILURE);
    }
    if (kMaxouts.empty()) kMaxouts.resize(l+2,1);
    if (kMaxouts.size()!=l+2) {
      cout << "Need " << l+2<< " kMaxout values." << endl;
      exit(EXIT_FAILURE);
    }
    addLayer(3, 2, k, baseSigmoidType,dropoutProbabilities[0],kMaxouts[0]);
    for (int i=2; i<=l; i++)
      addLayer(2, 2, i*k, baseSigmoidType,dropoutProbabilities[i-1],kMaxouts[i-1]);
    addLayer(2,1,(l+1)*k,
             baseSigmoidType
             //TANH
             ,dropoutProbabilities[l],kMaxouts[l]);
    addLayer(1,1,nOutputClasses, SOFTMAX,dropoutProbabilities[l+1],kMaxouts[l+1]);
    initialize();
    cout << "DeepCNet(" << l << "," << k << ")" << endl;
  }
};
class DeepDeepCNet : public CNN {
public:
  DeepDeepCNet(int l, int k, int nInputFeatures, int nOutputClasses, float learningRate,
               float momentumDecayRate, float weightDecayRate, int epoch,
               vector<float> dropoutProbabilities = vector<float>(),
               vector<int> kMaxouts = vector<int>()) :
    CNN(nInputFeatures, 3*(1<<l), learningRate, momentumDecayRate, weightDecayRate, epoch) {
    if (scale_N!=nInputSpatialSize) {
      cout << "scale_N shoule be " << 3*(1<<l) << endl;
      exit(EXIT_FAILURE);
    }
    if (dropoutProbabilities.empty()) dropoutProbabilities.resize(l+3,0);
    if (dropoutProbabilities.size()!=l+3) {
      cout << "Need " << l+3<< " dropout probabilities." << endl;
      exit(EXIT_FAILURE);
    }
    if (kMaxouts.empty()) kMaxouts.resize(l+3,1);
    if (kMaxouts.size()!=l+3) {
      cout << "Need " << l+3<< " kMaxout values." << endl;
      exit(EXIT_FAILURE);
    }
    addLayer(3, 2, k, baseSigmoidType,dropoutProbabilities[0],kMaxouts[0]);
    for (int i=2; i<=l; i++)
      addLayer(2, 2, i*k, baseSigmoidType,dropoutProbabilities[i-1],kMaxouts[i-1]);
    addLayer(2,1,(l+1)*k,TANH,dropoutProbabilities[l],kMaxouts[l]);
    addLayer(1,1,(l+2)*k,TANH,dropoutProbabilities[l+1],kMaxouts[l+1]);
    addLayer(1,1,nOutputClasses, SOFTMAX,dropoutProbabilities[l+2],kMaxouts[l+2]);
    initialize();
    cout << "DeepDeepCNet(" << l << "," << k << ")" << endl;
  }
};
class FlatDeepCNet : public CNN {
public:
  FlatDeepCNet(int l, int k, int nInputFeatures, int nOutputClasses, float learningRate,
               float momentumDecayRate, float weightDecayRate, int epoch,
               vector<float> dropoutProbabilities = vector<float>(),
               vector<int> kMaxouts = vector<int>()) :
    CNN(nInputFeatures, 3*(1<<l), learningRate, momentumDecayRate, weightDecayRate, epoch) {
    if (scale_N!=nInputSpatialSize) {
      cout << "scale_N shoule be " << 3*(1<<l) << endl;
      exit(EXIT_FAILURE);
    }
    if (dropoutProbabilities.empty()) dropoutProbabilities.resize(l+2,0);
    if (dropoutProbabilities.size()!=l+2) {
      cout << "Need " << l+2<< " dropout probabilities." << endl;
      exit(EXIT_FAILURE);
    }
    if (kMaxouts.empty()) kMaxouts.resize(l+2,1);
    if (kMaxouts.size()!=l+2) {
      cout << "Need " << l+2<< " kMaxout values." << endl;
      exit(EXIT_FAILURE);
    }
    addLayer(3, 2, k, baseSigmoidType,dropoutProbabilities[0],kMaxouts[0]);
    for (int i=2; i<=l; i++)
      addLayer(2, 2, k, baseSigmoidType,dropoutProbabilities[i-1],kMaxouts[i-1]);
    addLayer(2,1,k,baseSigmoidType,dropoutProbabilities[l],kMaxouts[l]);
    addLayer(1,1,nOutputClasses, SOFTMAX,dropoutProbabilities[l+1],kMaxouts[l+1]);
    initialize();
    cout << "FlatDeepCNet(" << l << "," << k << ")" << endl;
  }
};




class LeNet5 : public CNN { //scale_N=28 or 32
public:
  LeNet5(int sizeMultiplier, int nInputFeatures, int nOutputClasses, float learningRate, float momentumDecayRate, float weightDecayRate, int epoch) :
    CNN(nInputFeatures, scale_N, learningRate, momentumDecayRate, weightDecayRate, epoch) {
    addLayer(5,2, 2*sizeMultiplier,baseSigmoidType);
    addLayer(5,2, 5*sizeMultiplier,baseSigmoidType);
    addLayer(L.back().s2,1,50*sizeMultiplier,TANH);
    addLayer(1,1,nOutputClasses,SOFTMAX);
    initialize();
    cout << "LeNet5: sizeMultiplier = " << sizeMultiplier << endl;
  }
};

class LeNet7 : public CNN {//scale_N=96
public:
  LeNet7(int sizeMultiplier, int nInputFeatures, int nOutputClasses, float learningRate, float momentumDecayRate, float weightDecayRate, int epoch) :
    CNN(nInputFeatures, scale_N, learningRate, momentumDecayRate, weightDecayRate, epoch) {
    addLayer(5,4, 1*sizeMultiplier,baseSigmoidType);
    addLayer(6,3, 3*sizeMultiplier,baseSigmoidType);
    addLayer(6,1,12*sizeMultiplier,TANH);
    addLayer(1,1,60*sizeMultiplier,TANH);
    addLayer(1,1,nOutputClasses,SOFTMAX);
    initialize();
    cout << "LeNet7: sizeMultiplier = " << sizeMultiplier << endl;
  }
};

#include "nn-computational.h"

class ComputationalCNN {
public:
  CNN &nn;
  vector<ConvolutionalComputationalLayerBase*> CL;
  SparseCnnInterface* data;
  ConvolutionalComputationalLayerInterface input;

  ComputationalCNN(CNN &nn) : nn(nn) {
    CL.push_back(new ConvolutionalComputationalLayer(nn.L[0],0, input));
    for (int l=1;l<nn.L.size();l++)
      CL.push_back(new ConvolutionalComputationalLayer(nn.L[l],l,CL[l-1]->output));
  }
  void buildSparseProcessingRulesOnCPU() {
    input.type=data->type;
    input.batchSize=data->batchSize;
    input.featureSampleNumbers=data->featureSampleNumbers;
    input.backgroundNullVectorNumbers=data->backgroundNullVectorNumbers;
    input.grids=data->grids;//Big copy. Turn input.grids into a pointer.
    for (int l=0; l<nn.L.size(); l++)
      CL[l]->initialize();
  }
  void copySparseDataToGPU() {
    input.d_features=d_allocateArrayFromVector<float>(data->features,__FILE__,__LINE__);
    for (int l=0;l<nn.L.size();l++)
      CL[l]->copyDataToGPU();
  }
  void forwardPropagate()
  {
    for (int l=0;l<nn.L.size();l++)
      CL[l]->forwards();
  }


  void test() {
    int* d_predictions=d_allocateArray<int>(data->batchSize,__FILE__,__LINE__);
    dClassify<<<1,thisManyThreads>>>
      (CL[CL.size()-1]->output.d_features, d_predictions,
       data->batchSize, nn.L[nn.L.size()-1].nOut);
    vector<int>predictions(data->batchSize);
    d2hMemcopy<int>(d_predictions,&predictions[0],data->batchSize);
    cudaFree(d_predictions);
    for (int i=0;i<data->batchSize;i++)
      data->nMistakes+=(predictions[i]!=data->labels[i]);
  }

  //   data->topGuesses.resize(data->batchSize);
  //   for (int i=0;i<data->batchSize;i++) {
  //     int prediction;
  //     cublasIsamax(cublasHandle, nn.L[nn.L.size()-1].nOut,
  //                  CL[CL.size()-1]->output.d_features+i*nn.L[nn.L.size()-1].nOut,
  //                  1,&prediction);
  //     prediction--; //Fortran indexing!
  //     data->topGuesses[i].push_back(prediction);
  //     data->nMistakes+=(prediction!=data->labels[i]);
  //   }
  // }
  void findTopTenGuesses() {
    // //   for (int batchItem=0; batchItem < data->batchSize; batchItem++) {
    // //     data->topGuesses[batchItem].resize(10,0);
    // //     for (int i=0;i<10;i++) {
    // //       float* x=&CL[CL.size()-1]->output.d_features[0]+data->batchItem*nn.L[nn.L.size()-1].nOut;
    // //       float mx=x[0];
    // //       for (int j=1;j<nn.L[nn.L.size()-1].nOut;j++) {
    // //         if (x[j]>mx) {data->topGuesses[batchItem][i]=j;mx=x[j];}}
    // //       x[data->topGuesses[batchItem][i]]-=1;
    // //     }
    // //   }
  }

  void backwardPropagate() {
    //top layer: d Cost / d SoftmaxInput
    int* d_labels;
    float* d_delta;  //freed by the last call to backwards
    d_delta=d_allocateArrayZeroed<float>
      (CL[nn.L.size()-1]->output.count*nn.L[nn.L.size()-1].nOut,__FILE__,__LINE__);
    d_labels=d_allocateArrayFromVector<int>(data->labels,__FILE__,__LINE__);
    dDerivativeOfCostWRTpreSoftmaxTopLevelWeights<<<1,thisManyThreads>>>
      (data->batchSize, d_delta, CL[CL.size()-1]->output.d_features, d_labels, nn.L[nn.L.size()-1].nOut);
    cudaFree(d_labels);
    for (int l=CL.size()-1;l>=0;l--)
      CL[l]->backwards(d_delta);
  }

  void applyDerivatives() {
    for (int l=0;l<CL.size();l++)
      CL[l]->applyDerivatives(nn.learningRate*exp(-learningRateDecayRate*nn.epoch)
                              , nn.momentumDecayRate, nn.weightDecayRate);
    // CL[l]->applyDerivatives(nn.learningRate/(1+nn.epoch/learningRateDecayRate)
    //                         , nn.momentumDecayRate, nn.weightDecayRate);
  }
  void cleanUp() {
    cudaFree(input.d_features);
    for (int l=0;l<nn.L.size();l++)
      CL[l]->cleanUp();
  }
  void processBatch(SparseCnnInterface *d) {
    data=d;
    input.count=data->count;
    buildSparseProcessingRulesOnCPU();
    copySparseDataToGPU();
    forwardPropagate();
    if (data->type == UNLABELLEDBATCH)
      findTopTenGuesses();
    else
      test();
    if (data->type ==  TRAINBATCH) {
      backwardPropagate();
      boost::mutex::scoped_lock lock(CNNmutex);
      applyDerivatives();
      if (++nn.epoch%1000==0)
        nn.saveWeights();
    }
    cleanUp();
  }
};
