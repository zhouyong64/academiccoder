const int scale_N=192;
const int scale_n=64;
const float delta=scale_n/5;
const float learningRate=0.001;
const int trainingBatchSize=100;
const float learningRateDecayRate=0;
const int startEpoch=0*1000;
const int nIteratedIntegrals=2;
const int nInputFeatures=(2<<nIteratedIntegrals)-1;
const int nCharacters=183;
const char weightFileNameFormat[]="assamese_epoch-%d.cnn";
// Number of writers to use for training, leaving
// 45 minus ASSAMESETRAININGSAMPLES writers for a test set
#define ASSAMESETRAININGSAMPLES 36
#define ACTION train_test()
#include "CNN.h"
#include "Online.h"
DeepCNet cnn(6,50,nInputFeatures,nCharacters,learningRate,0.1,0.0,startEpoch);
Picture* OnlinePicture::distort() {
  RNG rng;
  OnlinePicture* pic=new OnlinePicture(*this);
  jiggleStrokes(pic->ops,rng,1);
  stretchXY(pic->ops,rng,0.3);
  int r=rng.randint(3);
  if (r==0) rotate(pic->ops,rng,0.3);
  if (r==1) slant_x(pic->ops,rng,0.3);
  if (r==2) slant_y(pic->ops,rng,0.3);
  jiggleCharacter(pic->ops,rng,12);
  return pic;
}
#include "run.h"
