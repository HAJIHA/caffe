#include "caffeClassifier.h"
#include "../cpp_classification_cervical_cancer/StringParse.h"
#include "ValidateModel.h"
#include "../cpp_classification_cervical_cancer/ManageDirectory.h"



void CValidateModel::InitModel(string strModelRoot, string strTrainedFile, string strImgRoot, string strTestListFile, int gpuNum)
{
    m_nBatchSize = 10;
    m_nOverSample = 10;

  string strRoot = CManageDirectory::checkNAddSlush(strModelRoot);

  m_trained_file = strRoot + strTrainedFile;
  m_mean_file = strRoot + "test.binaryproto";
  m_label_file = strRoot + "synset_words.txt";
  m_ValidateListFile = strRoot + strTestListFile;
  vector<string> vTraindFile = CStringParse::splitString(strTrainedFile, '.');
  vector<string> vTrainedFileWord = CStringParse::splitString(vTraindFile[0], '_');
  string strPredictResultFile = vTraindFile[0] + "_PrdResult.txt";

  bool bFcn = false;
  string strDeploy;
  for (int i = 0; i < vTrainedFileWord.size(); i++)
  {
    if (vTrainedFileWord[i] == "iter")
    {
      strDeploy += "deploy.prototxt";
      break;
    }
    else
    {
      strDeploy += vTrainedFileWord[i];
      strDeploy += "_";
    }
  }
  m_model_file = strRoot + strDeploy;

  m_classifier.loadModel(m_model_file, m_trained_file, m_mean_file, m_label_file, true, m_nBatchSize, gpuNum);

  m_strSavePredictResult = strRoot + strPredictResultFile;
  m_ImgRoot = strImgRoot;
  m_label = m_classifier.getLabelList();
}


void CValidateModel::testValidateSet()
{
  std::ifstream inFile;
  inFile.open(m_ValidateListFile);
  if (!inFile.is_open())
    return;

  char buf[1024];
  bool bTestSet = false;

  while (!inFile.getline(buf, sizeof(buf)).eof())
  {
    string line = string(buf);
    vector<string> vstrline = CStringParse::splitString(line, ' ');

    stValSetIntel OneSet;
    OneSet.strPath = vstrline[0];
    if (vstrline.size() == 1)
    {
      bTestSet = true;
      OneSet.strOrginCode = "NotCal";
    }
    else
    {
      int nSolLabel = atoi(vstrline[1].c_str());
      OneSet.strOrginCode = m_label[nSolLabel];
    }
    m_vValidateIntelList.push_back(OneSet);
  }

  string strHeader = "image_name,tags";
  ofstream outFile(m_strSavePredictResult, ios::out); 
  outFile << strHeader + "\n";

  const int nImgSize = m_nBatchSize / m_nOverSample;
  vector<cv::Mat> vImg;
  int nCnt = 0;
  for (int i = 0; i < m_vValidateIntelList.size(); i++)
  {
    clock_t st;
    clock_t et;
    st = clock();
	//여기서부터 휴가 복귀후 코딩
    m_vValidateIntelList[i].strPath = m_ImgRoot + m_vValidateIntelList[i].strPath;

      cv::Mat img = cv::imread(m_vValidateIntelList[i].strPath);
      if (img.empty())
      {
        inFile.close();
        //remove(m_vValidateIntelList[i].strPath.c_str());
        std::cout << "---------- Not Exist Image " << m_vValidateIntelList[i].strPath << " ----------" << std::endl;
        continue;
      }

      vImg.push_back(img);
      nCnt++;
      if (nCnt < nImgSize)
      {
        et = clock();
        double dTime = (double)(et - st) / CLOCKS_PER_SEC;
        cout << "process time one image : " + to_string(dTime) << endl;
        continue;
      }

      ///////////////////////////////////////////////////

      vector< vector<Prediction> >  vPredictions;

      vector<Prediction> prediction = m_classifier.ClassifyIntel(vImg[0]);
      vPredictions.push_back(prediction);
      vImg.clear();
      nCnt = 0;
      m_vValidateIntelList[i].strPredictCode = vPredictions[0][0].first;

      for (int labelidx = 0; labelidx < 3; labelidx++)
      {
        for (int prdidx = 0; prdidx < 3; prdidx++)
        {
          if (m_label[labelidx] == vPredictions[0][prdidx].first)
          {
            m_vValidateIntelList[i].fProb[labelidx] = vPredictions[0][prdidx].second;
            break;
          }
        }
      }

    for (int p = 0; p < nImgSize; p++)
    {
      stValSetIntel tempSet;
      tempSet = m_vValidateIntelList[i - (nImgSize - 1) + p];

      string strOneLineOut;
      string strFileName = CManageDirectory::getFileName(tempSet.strPath, '\\');
      strOneLineOut = strFileName + ",";
      strOneLineOut += to_string(tempSet.fProb[0]) + ",";
      strOneLineOut += to_string(tempSet.fProb[1]) + ",";
      strOneLineOut += to_string(tempSet.fProb[2]) + "\n";
      outFile << strOneLineOut;
    }

    et = clock();
    double dTime = (double)(et - st) / CLOCKS_PER_SEC;
    cout << "process time one image : " + to_string(dTime) << endl;
  }
}

CValidateModel::CValidateModel(string strModelRoot, string strTrainedFile, string strImgRoot, string strTestListFile, int gpuNum)
{
  InitModel(strModelRoot, strTrainedFile, strImgRoot, strTestListFile, gpuNum);
  testValidateSet();
}

CValidateModel::~CValidateModel()
{
}
