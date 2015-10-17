using namespace std;

#ifdef UJIPENCHARS
#if UJIPENCHARS == 0
char characterset[]="abcdefghijklmnopqrstuvwxyz";
#endif
#if UJIPENCHARS == 1
char characterset[]="ABCDEFGHIJKLMNOPQRSTUVWXYZ";
#endif
#if UJIPENCHARS == 2
char characterset[]="0123456789";
#endif

void loadData() {
  char* filename="Data/UJIpenchars2/ujipenchars2.txt";
  ifstream f(filename);
  if (!f) {
    cout << "Cannot find " << filename << endl;
    cout << "Please download it from the UCI Machine Learning Repository:" << endl;
    cout << "http://archive.ics.uci.edu/ml/machine-learning-databases/uji-penchars/version2/ujipenchars2.txt" << endl;
    exit(EXIT_FAILURE);
  }
  deque<string> data;
  copy(istream_iterator<string>(f),
       istream_iterator<string>(),
       back_inserter(data));
  while (data.size()>0) {
    while(data.front()!="WORD") {
      data.pop_front();
    }
    data.pop_front();
    OnlinePicture* character = new OnlinePicture;
    character->label=find
      (characterset,characterset+nCharacters,data.front()[0])
      -characterset;
    data.pop_front();
    int train=(data.front()[1]=='r');
    data.pop_front();
    data.pop_front();
    int numstrokes=atoi(data.front().c_str());
    data.pop_front();
    for (int i=0;i<numstrokes;i++) {
      data.pop_front();
      int numpoints=atoi(data.front().c_str());
      data.pop_front();
      data.pop_front();
      OnlinePenStroke stroke;
      for (int j=0;j<numpoints;j++) {
        FloatPoint p;
        p.y=atoi(data.front().c_str());
        data.pop_front();
        p.x=atoi(data.front().c_str());
        data.pop_front();
        stroke.push_back(p);
      }
      character->ops.push_back(stroke);
    }
    normalize(character->ops);
    if (character->label<nCharacters) {
      if (train) {
        trainCharacters.push_back(character);
      } else {
        testCharacters.push_back(character);
      }
    } else delete character;
  }
}
#endif
