// Y. LeCun, F.J. Huang, L. Bottou,
// Learning Methods for Generic Object Recognition with Invariance to Pose and Lighting.
// IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR) 2004
#ifdef NORBSMALL
#define NORBSETBACKGROUNDTOZERO 1
void loadNorbSmallPics(string filename, vector<Picture*> &characters) {
  unsigned char p1[9216], p2[9216]; //96*96=9216
  ifstream f(filename.c_str(),ios::binary);
  f.seekg (24, f.beg);
  for (int datum=0;datum<24300;datum++) {
    OfflineGridPicture* character = new OfflineGridCharacter(96,96);

    f.read((char*)p1,9216);
    f.read((char*)p2,9216);

#ifdef NORBSETBACKGROUNDTOZERO
    int histogram1[256], histogram2[256];
    for (int i=0;i<256;i++) histogram1[i]=histogram2[i]=0;
    for (int i=0;i<9216;i++) histogram1[p1[i]]++;
    for (int i=0;i<9216;i++) histogram2[p2[i]]++;
    int mode1=0, mode2=0, count1=0, count2=0;
    for (int i=0;i<256;i++) if(histogram1[i]>count1) {
        count1=histogram1[i];
        mode1=i;
      }
    for (int i=0;i<256;i++) if(histogram2[i]>count2) {
        count2=histogram2[i];
        mode2=i;
      }
#endif

    for (int i=0;i<9216;i++) {
      character.bitmap[i     ]=p1[i];
      character.bitmap[i+9216]=p2[i];
#ifdef NORBSETBACKGROUNDTOZERO
      character.bitmap[i     ]-=mode1;
      character.bitmap[i+9216]-=mode2;
      if (abs(character.bitmap[i     ])<5) character.bitmap[i     ]=0;
      if (abs(character.bitmap[i+9216])<5) character.bitmap[i+9216]=0;
#endif
    }
    characters.push_back(character);
  }
}

void loadNorbSmallLabels(string filename, vector<Picture*> characters) {
  ifstream f(filename.c_str());
  f.seekg(20);
  for (int datum=0;datum<24300;datum++) {
    int l;
    f.read((char*)&l,4);
    characters[datum]->label=l;
  }
}

void loadData() {
  string trainP("Data/NORB_small/smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat");
  string trainL("Data/NORB_small/smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat");
  string testP("Data/NORB_small/smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat");
  string testL("Data/NORB_small/smallnorb-5x01235x9x18x6x2x96x96-testing-cat.mat");
  loadNorbSmallPics(trainP, trainCharacters);
  loadNorbSmallLabels(trainL, trainLabels);
  loadNorbSmallPics(testP, testCharacters);
  loadNorbSmallLabels(testL, testLabels);
  cout << trainCharacters.size()<< " " << trainLabels.size() << " "
       << testCharacters.size() << " " << testLabels.size() << endl; //24300 24300 24300 24300
}
#endif
