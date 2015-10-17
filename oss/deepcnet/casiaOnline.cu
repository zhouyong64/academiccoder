//Test error: 3.57501% after 630k batches. Test error:  6.29656%
const int scale_N=192;
const int scale_n=64;
const float delta=scale_n/5;
const float learningRate=0.003;
const int trainingBatchSize=100;
const float learningRateDecayRate=1.0/200000;
const int startEpoch=0*1000;
const int nIteratedIntegrals=2;
const int nInputFeatures=(2<<nIteratedIntegrals)-1;
const int nCharacters=3755;
const char weightFileNameFormat[]="CASIAONLINE_epoch-%d.cnn";
#define ACTION train_test(9000,45000)
#include "CNN.h"
#define CASIA10
#include "Online.h"


DeepCNet cnn(6,100,nInputFeatures,nCharacters,learningRate,0.1,0,startEpoch
             ,list_of(0.0)(0.0)(0.0)(0.1)(0.2)(0.3)(0.4)(0.5)
             );
Picture* OnlinePicture::distort() {
  OnlinePicture* pic=new OnlinePicture(*this);
  RNG rng;
  jiggleStrokes(pic->ops,rng,1);
  stretchXY(pic->ops,rng,0.3);
  int r=rng.randint(3);
  if (r==0) rotate(pic->ops,rng,0.3);
  if (r==1) slant_x(pic->ops,rng,0.3);
  if (r==2) slant_y(pic->ops,rng,0.3);
  jiggleCharacter(pic->ops,rng,10);
  return pic;
}
#include "run.h"
