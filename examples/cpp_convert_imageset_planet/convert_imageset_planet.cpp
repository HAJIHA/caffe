// This program converts a set of images to a lmdb/leveldb by storing them
// as Datum proto buffers.
// Usage:
//   convert_imageset [FLAGS] ROOTFOLDER/ LISTFILE DB_NAME
//
// where ROOTFOLDER is the root folder that holds all the images, and LISTFILE
// should be a list of files as well as their labels, in the format as
//   subfolder1/file1.JPEG subfolder1/file2.tif subfolder1/file3.jpg 
//   ....

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>
#include <map>
#include <opencv2/opencv.hpp>

#include "boost/scoped_ptr.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"
#include "../cpp_classification_cervical_cancer/StringParse.h"

using namespace caffe;  // NOLINT(build/namespaces)
using namespace std;
using namespace cv;
using std::pair;
using boost::scoped_ptr;

DEFINE_bool(gray, false,
	"When this option is on, treat images as grayscale ones");
DEFINE_string(backend, "lmdb",
	"The backend {lmdb, leveldb} for storing the result");
DEFINE_int32(resize_width, 0, "Width images are resized to");
DEFINE_int32(resize_height, 0, "Height images are resized to");
DEFINE_bool(multi_label, false,
	"multi label use true : label is all zero");
DEFINE_string(label_file, "I:/imgfolder/LMDB/planet/synset_words.txt",
	"label_file path");



bool ReadImageToDatumPlanet(const vector<string> vFile, const int label,
	const int height, const int width, const int channels, Datum* datum)
{
	vector<cv::Mat> vImgMerge;
	cv::Mat merge_img;
	for (int i = 0; i < vFile.size(); i++)
	{
		vector<cv::Mat> vImgSplit;
		cv::Mat cv_img = cv::imread(vFile[i], CV_LOAD_IMAGE_UNCHANGED);
		if (height > 0 || width> 0)
			cv::resize(cv_img, cv_img, cv::Size(width, height));
		cv::split(cv_img, vImgSplit);

		for (int j = 0; j < vImgSplit.size(); j++)
		{
			cv::Mat fImg;
			int imgdepth = vImgSplit[j].depth();
			if (imgdepth == CV_8U)
			{
				vImgSplit[j].convertTo(fImg, CV_32FC1);
			}
			else
			{
				vImgSplit[j] /=  255.0;
				vImgSplit[j].convertTo(fImg, CV_32FC1);
			}
			vImgMerge.push_back(fImg);
		}
	}
	if (vImgMerge.size() != channels)
	{
		LOG(WARNING) << " Not equal channels  : first img path " << vFile[0];
		return false;
	}

	cv::merge(vImgMerge, merge_img);
	CVMatToDatum(merge_img, datum, merge_img.depth());
	datum->set_label(label);
	return true;
}

int main(int argc, char** argv) {
#ifdef USE_OPENCV
	::google::InitGoogleLogging(argv[0]);
	// Print output to stderr (while still logging)
	FLAGS_alsologtostderr = 1;

#ifndef GFLAGS_GFLAGS_H_
	namespace gflags = google;
#endif

	gflags::SetUsageMessage("Convert a set of images to the leveldb/lmdb\n"
		"format used as input for Caffe.\n"
		"Usage:\n"
		"    convert_imageset [FLAGS] ROOTFOLDER/ LISTFILE DB_NAME\n"
		"The ImageNet dataset for the training demo is at\n"
		"    http://www.image-net.org/download-images\n");
	gflags::ParseCommandLineFlags(&argc, &argv, true);

	if (argc < 4) {
		gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/convert_imageset_planet");
		return 1;
	}

	const int nFileNum = 2;
	const int nChannelNum = 7;
	const bool bMultiLabel = 2;
	const string label_file = FLAGS_label_file;
	map<string,int> mapLabel;
	ifstream label_infile(label_file);
	string line;
	int cnt = 0;
	while (getline(label_infile, line)) {
		mapLabel[line] = cnt;
		cnt++;
	}


	std::ifstream infile(argv[2]);
	std::vector<std::pair<std::string, int> > lines;
	vector<string> vStrLines;
	vector< vector<string> > vvImgFile;
	vector< vector<int> > vvLabel;
	while (std::getline(infile, line)) {
		vStrLines.push_back(line);
	}
	std::string root_folder(argv[1]);
	for (int i = 0; i < vStrLines.size(); i++)
	{
		vector<string> vParse = CStringParse::splitString(vStrLines[i], '\t');
		
		vector<string> vTemp;
		vTemp.push_back(root_folder + "/planet/train-jpg/" + vParse[0] + ".jpg");
		vTemp.push_back(root_folder + "/planet/train-tif/" + vParse[0] + ".tif");
		vvImgFile.push_back(vTemp);

		vector<int> vLabelOneHot;
		for (int vohidx = 0; vohidx < mapLabel.size(); vohidx++)
		{
			vLabelOneHot.push_back(0);
		}

		vector<string> vLabelTemp = CStringParse::splitString(vParse[1], ' ');
		for (int vlidx = 0; vlidx < vLabelTemp.size(); vlidx++)
		{
			vLabelOneHot[mapLabel[vLabelTemp[vlidx]]] = 1;
		}
		vvLabel.push_back(vLabelOneHot);
	}

	LOG(INFO) << "A total of " << vStrLines.size() << " image list.";

	int resize_height = std::max<int>(0, FLAGS_resize_height);
	int resize_width = std::max<int>(0, FLAGS_resize_width);

	// Create new DB
	scoped_ptr<db::DB> db(db::GetDB(FLAGS_backend));
	db->Open(argv[3], db::NEW);
	scoped_ptr<db::Transaction> txn(db->NewTransaction());

	// Label new DB
	scoped_ptr<db::DB> dbLabel(db::GetDB(FLAGS_backend));
	dbLabel->Open(string(argv[3]) + "_label", db::NEW);
	scoped_ptr<db::Transaction> txnLabel(dbLabel->NewTransaction());

	// Storing to db
	Datum datum;
	Datum datumLabel;
	int count = 0;
	int data_size = 0;
	bool data_size_initialized = false;
	datumLabel.set_channels(mapLabel.size());
	datumLabel.set_height(1);
	datumLabel.set_width(1);
	datumLabel.clear_data();
	datumLabel.clear_float_data();


	for (int line_id = 0; line_id < vvImgFile.size(); line_id++)
	{
		bool status;

		status = ReadImageToDatumPlanet(vvImgFile[line_id], 0,
			 resize_height, resize_width, nChannelNum,
			 &datum);

		if (status == false) continue;

		for (int mlidx = 0; mlidx<mapLabel.size(); mlidx++)
		{
			datumLabel.add_float_data(vvLabel[line_id][mlidx]);
		}
		datumLabel.set_label(0);
		// sequential
		string key_str = caffe::format_int(line_id, 8) + "_" + vvImgFile[line_id][0];
		LOG(INFO) << to_string(line_id) << "th file";
		CHECK_EQ(nChannelNum, datum.channels());
		CHECK_EQ(mapLabel.size(), datumLabel.channels());
		// Put in db
		string out;
		string outLabel;
		CHECK(datum.SerializeToString(&out));
		txn->Put(key_str, out);
		CHECK(datumLabel.SerializeToString(&outLabel));
		txnLabel->Put(key_str, outLabel);

		if (++count % 1000 == 0) {
			// Commit db
			txn->Commit();
			txn.reset(db->NewTransaction());
			txnLabel->Commit();
			txnLabel.reset(dbLabel->NewTransaction());
			LOG(INFO) << "Processed " << count << " files.";
		}
	}

	// write the last batch
	if (count % 1000 != 0) {
		txn->Commit();
		txnLabel->Commit();
		LOG(INFO) << "Processed " << count << " files.";
	}
#else
	LOG(FATAL) << "This tool requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
	return 0;
}
