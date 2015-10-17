const int nCharacters=10;
const int nInputFeatures=1;
const int scale_N=96;
const int startEpoch=0*1000;
const char weightFileNameFormat[]="mnist_epoch-%d.cnn";
const float learningRate=0.0003;
const float learningRateDecayRate=0;
const int trainingBatchSize=100;
#define MNIST
#define ACTION train_test(6000,6000)
#include "CNN.h"
#include "OfflineGrid.h"
DeepCNet cnn(5,10,nInputFeatures,nCharacters,learningRate,0.1,0,startEpoch);

Picture* OfflineGridPicture::distort() {
  OfflineGridPicture* pic=new OfflineGridPicture(*this);
  RNG rng;
  pic->jiggle(rng,2);
  return pic;
}
#include "run.h"
