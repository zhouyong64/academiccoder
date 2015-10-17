using namespace std;

#ifdef PENDIGITS
void loadPendigits(string filename, vector<Picture*> &characters) {
  ifstream f(filename.c_str());
  if (!f) {
    cout << "Cannot find " << filename << endl;
    cout << "Please download it from the UCI Machine Learning Repository:" << endl;
    cout << "http://archive.ics.uci.edu/ml/datasets/Pen-Based+Recognition+of+Handwritten+Digits" << endl;
    exit(EXIT_FAILURE);
  }
  vector<string> data;
  copy(istream_iterator<string>(f),
       istream_iterator<string>(),
       back_inserter(data));
  int pen=0;
  OnlinePicture* character = new OnlinePicture;
  OnlinePenStroke stroke;
  FloatPoint p;
  for (int i=0;i< data.size();i++) {
    if (data[i]==".COMMENT") {
      if (character->ops.size()>0) {
        normalize(character->ops);
        characters.push_back(character);
        character = new OnlinePicture;
      }
      character->label=atoi(data[i+1].c_str());
    }
    if (data[i]==".PEN_UP") {
      pen=0;
      character->ops.push_back(stroke);
      stroke.clear();
    }
    if (pen==1) {
      p.y=atoi(data[i].c_str());
      i++;
      p.x=-atoi(data[i].c_str());
      stroke.push_back(p);
    }
    if (data[i]==".PEN_DOWN") pen=1;
  }

  normalize(character->ops);
  characters.push_back(character);
}


void loadData() {
  string train("Data/PenDigits/pendigits-orig.tra");
  loadPendigits(train, trainCharacters);
  string test("Data/PenDigits/pendigits-orig.tes");
  loadPendigits(test, testCharacters);
}
#endif
