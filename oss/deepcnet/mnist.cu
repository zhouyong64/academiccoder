//0.31% test error after 600,000 batches=1000 epochs (training error on jiggled data 0.15%)
const int nCharacters=10;
const int nInputFeatures=1;
const int scale_N=96;
const int startEpoch=0*1000;
const char weightFileNameFormat[]="mnist_epoch-%d.cnn";
const float learningRate=0.01;
const float learningRateDecayRate=0.00001;
const int trainingBatchSize=100;
#define MNIST
#define ACTION train_test(6000,6000)
#include "CNN.h"
#include "OfflineGrid.h"
DeepCNet cnn(5,60,nInputFeatures,nCharacters,learningRate,0.1,0,startEpoch
             ,list_of(0.0)(0.0)(0.0)(0.5)(0.5)(0.5)(0.5)
             );

Picture* OfflineGridPicture::distort() {
  OfflineGridPicture* pic=new OfflineGridPicture(*this);
  RNG rng;
  pic->jiggle(rng,2);
  return pic;
}
#include "run.h"
