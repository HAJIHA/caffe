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
DEFINE_bool(encoded, false,
	"When this option is on, the encoded image will be save in datum");
DEFINE_string(encode_type, "",
	"Optional: What type should we encode the image as ('png','jpg',...).");
DEFINE_int32(file_num, 1,
	"image file num");
DEFINE_int32(channel_num, 3,
	"image file num");
DEFINE_bool(multi_label, false,
	"multi label use true : label is all zero");

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

	const bool is_color = !FLAGS_gray;
	const bool encoded = FLAGS_encoded;
	const string encode_type = FLAGS_encode_type;
	const int nFileNum = FLAGS_file_num;
	const int nChannelNum = FLAGS_channel_num;
	const bool bMultiLabel = FLAGS_multi_label;

	std::ifstream infile(argv[2]);
	std::vector<std::pair<std::string, int> > lines;
	std::string line;
	vector<string> vStrLines;
	vector< vector<string> > vvImgFile;
	vector< int > vLabel;

	while (std::getline(infile, line)) {
		vStrLines.push_back(line);
	}
	std::string root_folder(argv[1]);
	for (int i = 0; i < vStrLines.size(); i++)
	{
		vector<string> vParse = CStringParse::splitString(vStrLines[i], ' ');
		vector<string> vTemp;
		for (int j = 0; j < nFileNum; j++)
		{
			vTemp.push_back(root_folder + vParse[j]);
		}
		vvImgFile.push_back(vTemp);

		if (bMultiLabel == true)
		{
			vLabel.push_back(0);
		}
		else
		{
			vLabel.push_back(atoi(vParse[nFileNum].c_str()));
		}
	}

	LOG(INFO) << "A total of " << vStrLines.size() << " image list.";

	if (encode_type.size() && !encoded)
		LOG(INFO) << "encode_type specified, assuming encoded=true.";

	int resize_height = std::max<int>(0, FLAGS_resize_height);
	int resize_width = std::max<int>(0, FLAGS_resize_width);

	// Create new DB
	scoped_ptr<db::DB> db(db::GetDB(FLAGS_backend));
	db->Open(argv[3], db::NEW);
	scoped_ptr<db::Transaction> txn(db->NewTransaction());

	// Storing to db
	Datum datum;
	int count = 0;
	int data_size = 0;
	bool data_size_initialized = false;

	// readimagetodatum 부분도 작성필요. 완료하지 않고 중단
	for (int line_id = 0; line_id < vvImgFile.size(); line_id++)
	{
		bool status;

		status = ReadImageToDatum(vvImgFile[line_id], vLabel[line_id],
			 resize_height, resize_width, nChannelNum,
			 &datum);

		if (status == false) continue;
		// sequential
		string key_str = caffe::format_int(line_id, 8) + "_" + vvImgFile[line_id][0];
		LOG(INFO) << to_string(line_id) << "th file";
		CHECK_EQ(nChannelNum, datum.channels());
		// Put in db
		string out;
		CHECK(datum.SerializeToString(&out));
		txn->Put(key_str, out);

		if (++count % 1000 == 0) {
			// Commit db
			txn->Commit();
			txn.reset(db->NewTransaction());
			LOG(INFO) << "Processed " << count << " files.";
		}
	}

	// write the last batch
	if (count % 1000 != 0) {
		txn->Commit();
		LOG(INFO) << "Processed " << count << " files.";
	}
#else
	LOG(FATAL) << "This tool requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
	return 0;
}
