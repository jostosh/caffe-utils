#include <stdio.h>  // for snprintf
#include <string>
#include <vector>
#include <iostream>
#include <fstream>

#include "boost/algorithm/string.hpp"
#include "boost/filesystem.hpp"
#include "boost/regex.hpp"
#include "google/protobuf/text_format.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"
#include "caffe/vision_layers.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

using caffe::Blob;
using caffe::Caffe;
using caffe::Datum;
using caffe::Net;
using boost::shared_ptr;
using std::string;
namespace db = caffe::db;

template<typename Dtype>
int feature_extraction_pipeline(int argc, char** argv)
{
    ::google::InitGoogleLogging(argv[0]);
    const int num_required_args = 8;
    if (argc < num_required_args)
    {
        LOG(ERROR) <<
                      "\nextract_to_txt -- April 2015\n"
                      "Edited by: Jos van de Wolfshaar\n"
                      "This program loads a pretrained network and extracts the features to a \n"
                      "text file. To extract to libsvm format call the program with 'libsvm'. For\n"
                      "more info check out the README.\n"
                      "\tUsage: \n"
                      "\textract_to_txt <train|test> <path_to_weight_file> <path_to_model_prototxt> <path_to_csv> <blob_name> <n_mini_batches> <output_format>"
                      "  [CPU/GPU] [DEVICE_ID=0]\n";
        return 1;
    }
    int arg_pos = num_required_args;

    arg_pos = num_required_args;
    if (argc > arg_pos && strcmp(argv[arg_pos], "GPU") == 0)
    {
        LOG(ERROR)<< "Using GPU";
        uint device_id = 0;
        if (argc > arg_pos + 1)
        {
            device_id = atoi(argv[arg_pos + 1]);
            CHECK_GE(device_id, 0);
        }
        LOG(ERROR) << "Using Device_id=" << device_id;
        Caffe::SetDevice(device_id);
        Caffe::set_mode(Caffe::GPU);
    }
    else
    {
        LOG(ERROR) << "Using CPU";
        Caffe::set_mode(Caffe::CPU);
    }
    arg_pos = 0;

    caffe::Phase phase = (std::string(argv[++arg_pos]) == "test") ? caffe::TEST : caffe::TRAIN;

    std::string pretrained_binary_proto(argv[++arg_pos]); // path to pretrained binary file

    std::string feature_extraction_proto(argv[++arg_pos]); // path to feature extraction proto file
    shared_ptr<Net<Dtype> > feature_extraction_net(new Net<Dtype>(feature_extraction_proto, phase));
    feature_extraction_net->CopyTrainedLayersFrom(pretrained_binary_proto);   /// load pretrained weights

    std::ofstream output_file(argv[++arg_pos]);          // open output file to write features to
    LOG(INFO) << "Output file path -- " << argv[arg_pos];

    std::string blob_name;                        // get blob name
    blob_name = argv[++arg_pos];
    LOG(INFO) << "Layer name -- " << argv[arg_pos];

    // Check the given blob name
    CHECK(feature_extraction_net->has_blob(blob_name))         // check existence of blob
            << "Unknown feature blob name " << blob_name
            << " in the network " << feature_extraction_proto;

    int num_mini_batches = atoi(argv[++arg_pos]);                 // parse number of mini batches

    bool libsvm_format = (std::string(argv[++arg_pos]) == "libsvm");

    std::vector<Blob<float>*> input_vec;
    for (int batch_index = 0; batch_index < num_mini_batches; ++batch_index)
    {
        feature_extraction_net->Forward(input_vec);                                 // feed forward
        const Dtype* feature_blob_data;
        const shared_ptr<Blob<Dtype> > feature_blob = feature_extraction_net
                ->blob_by_name(blob_name);                // obtain feature blob
        int batch_size = feature_blob->num();

        const shared_ptr<Blob<Dtype> > label_blob = feature_extraction_net
                ->blob_by_name("label");                    // assumes there exists a 'label' blob
        const Dtype* label_data = label_blob->cpu_data();

        int dim_features = feature_blob->count() / batch_size;
        for (int image_index = 0; image_index < batch_size; ++image_index)
        {
            feature_blob_data = feature_blob->cpu_data() +
                    feature_blob->offset(image_index);

            output_file << label_data[image_index] << " ";  // first put the label

            for (int feature_index = 0; feature_index < dim_features; ++feature_index)
            {
                if (libsvm_format)      // libsvm format
                {
                    if (feature_blob_data[feature_index] == 0)
                        continue;
                    else
                        output_file << feature_index + 1 << ':' << feature_blob_data[feature_index] << ' ';
                } else {                // plain text format
                    if (feature_blob_data[feature_index] == 0)
                        output_file << "0 ";
                    else
                        output_file << feature_blob_data[feature_index] << ' ';
                }
            }
            output_file << '\n';
            if ((batch_index * batch_size + image_index) % 1000 == 0)
                LOG(INFO) << "Extracted features of " << batch_index * batch_size + image_index << " images";
        }
    }

    LOG(ERROR)<< "Successfully extracted the features!";
    return 0;
}


int main(int argc, char** argv)
{
    return feature_extraction_pipeline<float>(argc, argv);
}



