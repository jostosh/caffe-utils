# caffe-utils
Additional scripts for caffe

---------- extract_to_txt.cpp ------------

This script was made for extracting features from caffe to plain text format or to `libsvm` 
format. The script can be used as follows:
```
extract_to_txt \  
  <train|test> \  
  <path_to_weight_file> \  
  <path_to_model_prototxt> \  
  <path_to_output_file> \  
  <blob_name> \  
  <n_mini_batches> \  
  <output_format> \  
  [CPU/GPU] \  
  [DEVICE_ID=0]
```

The GPU arguments are optional. I recommend writing a bash script file were you put your
parameter configuration for calling the script.

**EXAMPLE:**
For extracting the features of layer `fc7` of the train set to `libsvm` format you can do:
```
extract_to_txt \
  train \
  model.caffemodel \
  train_val.prototxt \
  features.txt \
  fc7 \
  100 \
  libsvm \
  GPU \
  0
```

This will extract 100 mini batches. If your network has a batch size of 128 the call above
will extract the features of 100 * 128 images. 

**IMPORTANT NOTES:**
- The script assumes that the network has defined a `label` blob. 
- You can add this script to $CAFFE_ROOT/tools and recompile caffe after which you can
  find the script in $CAFFE_ROOT/build/tools

**LIBSVM:**
- For downloading `libsvm`, check out:
  http://www.csie.ntu.edu.tw/~cjlin/libsvm/
