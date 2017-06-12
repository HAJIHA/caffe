#include <stdint.h>
#include <algorithm>
#include <string>
#include <utility>
#include <vector>
#include <string>

#include "boost/scoped_ptr.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"
#include <opencv2\opencv.hpp>

using namespace caffe;  // NOLINT(build/namespaces)
using namespace cv;
using namespace std;
using std::max;
using std::pair;
using boost::scoped_ptr;

DEFINE_string(backend, "lmdb",
	"The backend {leveldb, lmdb} containing the images");
DEFINE_int32(resize_width, 256, "Width images are resized to");
DEFINE_int32(resize_height, 256, "Height images are resized to");
DEFINE_string(depth, "8U", "depth");

int main(int argc, char** argv) {
	::google::InitGoogleLogging(argv[0]);

#ifdef USE_OPENCV
#ifndef GFLAGS_GFLAGS_H_
	namespace gflags = google;
#endif

	gflags::SetUsageMessage("Compute the mean_image of a set of images given by"
		" a leveldb/lmdb\n"
		"Usage:\n"
		"    compute_image_mean [FLAGS] INPUT_DB [OUTPUT_FILE]\n");

	gflags::ParseCommandLineFlags(&argc, &argv, true);

	if (argc < 3) {
		gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/compute_image_mean");
		return 1;
	}

	int pre_width = std::max<int>(0, FLAGS_resize_height);
	int pre_height = std::max<int>(0, FLAGS_resize_width);
	string strdepth = FLAGS_depth;

	scoped_ptr<db::DB> db(db::GetDB(FLAGS_backend));
	db->Open(argv[1], db::READ);
	scoped_ptr<db::Cursor> cursor(db->NewCursor());

	BlobProto sum_blob;
	int count = 0;
	// load first datum
	Datum datum;
	datum.ParseFromString(cursor->value());

	if (DecodeDatumNative(&datum)) {
		LOG(INFO) << "Decoding Datum";
	}

	sum_blob.set_num(1);
	sum_blob.set_channels(datum.channels());
	sum_blob.set_height(pre_height);
	sum_blob.set_width(pre_width);
	const int data_size = datum.channels() * pre_height * pre_width;
	int size_in_datum = std::max<int>(data_size,
		datum.float_data_size());
	for (int i = 0; i < size_in_datum; ++i) {
		sum_blob.add_data(0.);
	}
	LOG(INFO) << "Starting Iteration";
	while (cursor->valid()) {
		Datum Predatum;
		Predatum.ParseFromString(cursor->value());
		DecodeDatumNative(&Predatum);
		cv::Mat cv_img = DatumToCVMat(Predatum);
		if (cv_img.cols < cv_img.rows)
		{
			cv::transpose(cv_img, cv_img);
			cv::flip(cv_img, cv_img, 1);
		}

		if (pre_height != cv_img.rows || pre_width != cv_img.cols)
			cv::resize(cv_img, cv_img, cv::Size(pre_width, pre_height));

		Datum datum;
		if (strdepth == "8U")
		{
			CVMatToDatum(cv_img, &datum,CV_8U);
		}
		else
		{
			CVMatToDatum(cv_img, &datum, CV_32F);
		}

		//vector<Mat> vImgSplit;
		//split(cv_img, vImgSplit);
		//for (int chidx = 0; chidx < vImgSplit.size(); chidx++)
		//{
		//	Mat dispImg;
		//	double minVal, maxVal;
		//	minMaxLoc(vImgSplit[chidx], &minVal, &maxVal); //find minimum and maximum intensities
		//	vImgSplit[chidx].convertTo(dispImg, CV_8UC1, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));
		//	imshow("ch_" + to_string(chidx), dispImg);
		//}

		//waitKey(0);

		const std::string& data = datum.data();
		size_in_datum = std::max<int>(datum.data().size(),
			datum.float_data_size());
		CHECK_EQ(size_in_datum, data_size) << "Incorrect data field size " <<
			size_in_datum;
		if (data.size() != 0) {
			CHECK_EQ(data.size(), size_in_datum);
			for (int i = 0; i < size_in_datum; ++i) {
				sum_blob.set_data(i, sum_blob.data(i) + (uint8_t)data[i]);
			}
		}
		else {
			CHECK_EQ(datum.float_data_size(), size_in_datum);
			for (int i = 0; i < size_in_datum; ++i) {
				sum_blob.set_data(i, sum_blob.data(i) +
					static_cast<float>(datum.float_data(i)));
			}
		}
		++count;
		if (count % 10000 == 0) {
			LOG(INFO) << "Processed " << count << " files.";
		}
		cursor->Next();
	}

	if (count % 10000 != 0) {
		LOG(INFO) << "Processed " << count << " files.";
	}
	for (int i = 0; i < sum_blob.data_size(); ++i) {
		sum_blob.set_data(i, sum_blob.data(i) / count);
	}
	// Write to disk
	if (argc >= 3) {
		LOG(INFO) << "Write to " << argv[2];
		WriteProtoToBinaryFile(sum_blob, argv[2]);
	}
	const int channels = sum_blob.channels();
	const int dim = sum_blob.height() * sum_blob.width();
	std::vector<float> mean_values(channels, 0.0);
	LOG(INFO) << "Number of channels: " << channels;
	for (int c = 0; c < channels; ++c) {
		for (int i = 0; i < dim; ++i) {
			mean_values[c] += sum_blob.data(dim * c + i);
		}
		LOG(INFO) << "mean_value channel [" << c << "]:" << mean_values[c] / dim;
	}
#else
	LOG(FATAL) << "This tool requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
	return 0;
}
