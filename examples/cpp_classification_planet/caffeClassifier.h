#pragma once
#include <caffe/caffe.hpp>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace caffe;  // NOLINT(build/namespaces)
using std::string;

/* Pair (label, confidence) representing a prediction. */
typedef std::pair<string, float> Prediction;
class CaffeClassifier
{
public:
	CaffeClassifier();
	~CaffeClassifier();
	CaffeClassifier(const string& model_file,
		const string& trained_file,
		const string& mean_file,
		const string& label_file,	const bool use_GPU,
		const int batch_size,
		const int gpuNum);

	void loadModel(const string& model_file,
		const string& trained_file,
		const string& mean_file,
		const string& label_file, const bool use_GPU,
		const int batch_size,
		const int gpuNum);

    vector<Prediction> ClassifyIntel(const cv::Mat img);


	std::vector<string> getLabelList();


private:
	void PreprocessBatch(const vector<cv::Mat> imgs, std::vector< std::vector<cv::Mat> >* input_batch);
	void PreprocessBatchNonSub(const vector<cv::Mat> imgs, std::vector< std::vector<cv::Mat> >* input_batch);
	void WrapBatchInputLayer(std::vector<std::vector<cv::Mat> > *input_batch);
	vector<cv::Mat> OverSample(const vector<cv::Mat> vImgs, int size);
	vector<cv::Mat> OverSample(const cv::Mat img, int size);
	vector<cv::Mat> OverSampleIntel(const cv::Mat img, int nOverSample);
	vector<cv::Mat> OverSampleIntel(const cv::Mat img, int nOverSample, cv::Mat mean);
	//vector<cv::Mat> OverSampleUnifDist(const cv::Mat img, int nOverSample);

	vector< float >  PredictBatch(const vector< cv::Mat > imgs);
	vector< float >  PredictBatchNonSub(const vector< cv::Mat > imgs);
    // num, channel, height, width
    vector< vector< vector< vector< float > > > > PredictFcnBatch(const vector< cv::Mat > imgs);
	void SetMean(const string& mean_file);

private:
	boost::shared_ptr<Net<float> > net_;
	cv::Size input_geometry_;
	int batch_size_;
	int num_channels_;
	cv::Mat mean_;
	std::vector<string> labels_;

	string m_modelfile;
	string m_trained_file;
	string m_mean_file;
	string m_label_file;
	int m_nUseGpuNum;
    bool m_bFcn;
};

