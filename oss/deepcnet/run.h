ComputationalCNN ccnn(cnn);

class BatchProducer {
  boost::mutex batchDequeueMutex;
  boost::mutex batchCounterMutex;
public:
  boost::thread_group workers;
  vector<Picture*>* dataset;
  deque<SparseCnnInterface*> dq;
  int batchCounter; //# batches started to be created.
  int batchCounter2;//# batches "popped" from the deque

  SparseCnnInterface* pop() {
    while (dq.size()==0)
      boost::this_thread::sleep(boost::posix_time::milliseconds(100));
    batchDequeueMutex.lock();
    SparseCnnInterface* batch=dq.front();
    dq.pop_front();
    batchCounter2++;
    batchDequeueMutex.unlock();
    return batch;
  }
  void push(SparseCnnInterface* batch) {
    while (dq.size()>20)
      boost::this_thread::sleep(boost::posix_time::milliseconds(100));
    batchDequeueMutex.lock();
    dq.push_back(batch);
    batchDequeueMutex.unlock();
  }
  virtual bool workToDo () = 0;
  virtual void batchProducerThread() = 0;
  BatchProducer (vector<Picture*>* dataset) : dataset(dataset), batchCounter(0), batchCounter2(0) {}
  void start(int nThreads = 8) {
    for (int i=0; i<nThreads; i++){
      workers.add_thread(new boost::thread(boost::bind(&BatchProducer::batchProducerThread,this)));
    }
  }
  int getBatchCounter () {
    batchCounterMutex.lock();
    int c = batchCounter++;
    batchCounterMutex.unlock();
    return c;
  }
  void join() {
    workers.join_all();
  }
};

class RandomizedTrainingBatchProducer : public BatchProducer {
public:
  void batchProducerThread() {
    RNG rng;
    while (true) {
      int c = getBatchCounter();
      SparseCnnInterface* batch = new SparseCnnInterface(TRAINBATCH);
      for (int i=0;i<trainingBatchSize;i++) {
        Picture* pic=dataset->at(rng.index(*dataset))->distort();
        pic->codifyInputData(*batch);
        delete pic;
      }
      push(batch);
    }
  }
  bool workToDo () {
    return true;
  }
  RandomizedTrainingBatchProducer (vector<Picture*>* dataset) : BatchProducer(dataset) {
    start();
  }
};
class TestsetBatchProducer : public BatchProducer {
public:
  void batchProducerThread() {
    while (true) {
      int c = getBatchCounter();
      if (c*trainingBatchSize>=dataset->size())
        break;
      SparseCnnInterface* batch = new SparseCnnInterface(TESTBATCH);
      for (int i=c*trainingBatchSize;
           i<min((c+1)*trainingBatchSize,(int)(dataset->size()));
           i++)
        dataset->at(i)->codifyInputData(*batch);
      push(batch);
    }
  }
  bool workToDo () {
    return batchCounter2*trainingBatchSize < dataset->size() || dq.size()>0;
  }
  TestsetBatchProducer (vector<Picture*>* dataset) : BatchProducer(dataset) {
    start();
  }
};


void test(bool verbose=false) {
  TestsetBatchProducer bp(&testCharacters);
  int total=0;
  int wrong=0;
  while(bp.workToDo()) {
    SparseCnnInterface* batch=bp.pop();
    ccnn.processBatch(batch);
    wrong+=batch->nMistakes;
    total+=batch->batchSize;
    delete batch;
    if (verbose)
      cout << "Test set size: " << total << " Test error: " << wrong*100.0/total << "%" <<endl;
  }
  bp.join();
  if (!verbose)
    cout << "Test set size: " << total << " Test error: " << wrong*100.0/total << "%" <<endl;
}


void train(int nTrain=1000) {
  RandomizedTrainingBatchProducer bp(&trainCharacters);
  int mistakes=0;
  int total=0;
  for (int epoch=startEpoch+1; bp.workToDo(); epoch++) {
    SparseCnnInterface* batch=bp.pop();
    ccnn.processBatch(batch);
    mistakes+=batch->nMistakes;
    total+=trainingBatchSize;
    delete batch;
    if (epoch%nTrain==0) {
      cout << "Training batch: " << epoch << " " << "Mistakes: " << mistakes*100.0/total << "%" << endl;
      mistakes=0;
      total=0;
    }
  }
}

void train_test(int nTrain=1000, int nTest=10000) {
  RandomizedTrainingBatchProducer bp(&trainCharacters);
  int mistakes=0;
  int total=0;
  for (int epoch=startEpoch+1; bp.workToDo(); epoch++) {
    SparseCnnInterface* batch=bp.pop();
    ccnn.processBatch(batch);
    mistakes+=batch->nMistakes;
    total+=trainingBatchSize;
    delete batch;
    if (epoch%100==0)
      cout << "\rTraining batch: " << epoch << " " << "Mistakes: " << mistakes*100.0/total << "%       " << flush;
    if (epoch%nTrain==0) {
      cout << "\rTraining batch: " << epoch << " " << "Mistakes: " << mistakes*100.0/total << "%\n" ;
      mistakes=0;
      total=0;
    }
    if (epoch%nTest==0) {
      cout << "\r       ";
      test(false);
    }
  }
}


int main() {
  ACTION;
  return 0;
}
