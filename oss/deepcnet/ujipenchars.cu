const char weightFileNameFormat[]="ujipenchars_epoch-%d.cnn";
const int scale_N=48;
const int scale_n=16;
const float delta=scale_n/5;
const float learningRate=0.001;
const float learningRateDecayRate=0;
const int trainingBatchSize=100;
const int startEpoch=0*1000;
const int nIteratedIntegrals=2;
const int nInputFeatures=(2<<nIteratedIntegrals)-1;
const int nCharacters=26; //26 for lower/upper case, 10 for digits
#define ACTION train_test()
#include "CNN.h"
#define UJIPENCHARS 0 /* 0 for lower case, 1 for upper case, 2 for digits*/
#include "Online.h"
DeepCNet cnn(4,20,nInputFeatures,nCharacters,learningRate,0.1,0.0,startEpoch);
Picture* OnlinePicture::distort() {
  RNG rng;
  OnlinePicture* pic=new OnlinePicture(*this);
  jiggleStrokes(pic->ops,rng,1);
  stretchXY(pic->ops,rng,0.3);
  int r=rng.randint(3);
  if (r==0) rotate(pic->ops,rng,0.3);
  if (r==1) slant_x(pic->ops,rng,0.3);
  if (r==2) slant_y(pic->ops,rng,0.3);
  jiggleCharacter(pic->ops,rng,8);
  return pic;
}
#include "run.h"
