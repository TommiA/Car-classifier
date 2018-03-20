
# coding: utf-8

import operator
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf

from shutil import copyfile
from PIL import Image
from utils import label_map_util

# This is needed since the notebook is stored in the object_detection folder.
#sys.path.append("..")
#from object_detection.utils import ops as utils_ops

if tf.__version__ < '1.4.0':
  raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')


#from utils import visualization_utils as vis_util

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

#Sorts image files to the matching classname folders
def sortFiles(output_dict, image_path):
    classes_over_zero_five={}
    for i in range(len(output_dict['detection_classes'])):
        classid=output_dict['detection_classes'][i]
        classname=category_index[classid]['name']
        classscore=output_dict['detection_scores'][i]
        if classscore >= 0.5:
            if classname in tag_tab:
                tag_tab[classname]=tag_tab[classname]+1               
            else:
                tag_tab[classname]=1
            if classname in classes_over_zero_five:
                if classes_over_zero_five[classname] < classscore:
                    classes_over_zero_five[classname]=classscore
            else:
                classes_over_zero_five[classname]=classscore
    highest_classname=''    
    #Just to bypass the sorting as most of the cases have only one class name over 0.5
    if len(classes_over_zero_five) > 1:
        sorted_classes = sorted(classes_over_zero_five.items(), key=operator.itemgetter(1), reverse=True)
        highest_classname=sorted_classes[0][0]
    if len(classes_over_zero_five) == 1:
        highest_classname=classes_over_zero_five.keys()[0]
    if not highest_classname=='':  
        if not os.path.exists(PATH_TO_TEST_IMAGES_DIR+"sorted/"+highest_classname):
            os.makedirs(PATH_TO_TEST_IMAGES_DIR+"sorted/"+highest_classname)    
        copyfile(image_path, PATH_TO_TEST_IMAGES_DIR+"sorted/"+highest_classname+"/"+image_path.split('/')[-1])
    else:
        if not os.path.exists(PATH_TO_TEST_IMAGES_DIR+"unsorted/"):
            os.makedirs(PATH_TO_TEST_IMAGES_DIR+"unsorted/")    
        copyfile(image_path, PATH_TO_TEST_IMAGES_DIR+"unsorted/"+image_path.split('/')[-1])

def run_inference_for_bunch(images, graph):
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}     
      for key in [
          'num_detections', 'detection_scores', 'detection_classes'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)

      for image_path in images:
          image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

          print(image_path+' -+')
          image = Image.open(open(image_path), 'r')

          try:
              image_np = load_image_into_numpy_array(image)
          except:
              print("Image load tilt, NEXT!")
              continue
          image_np_expanded = np.expand_dims(image_np, axis=0)
          
          output_dict = sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image, 0)})

          # all outputs are float32 numpy arrays, so convert types as appropriate
          output_dict['num_detections'] = int(output_dict['num_detections'][0])
          output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
          output_dict['detection_scores'] = output_dict['detection_scores'][0]
          
          #Sort the files          
          sortFiles(output_dict, image_path)
                               
# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

opener = urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
  file_name = os.path.basename(file.name)
  if 'frozen_inference_graph.pb' in file_name:
    tar_file.extract(file, os.getcwd())

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

PATH_TO_TEST_IMAGES_DIR = '/home/eletai/nettiauto_scrape/'
TEST_IMAGE_PATHS = []
for root, dirs, files in os.walk(PATH_TO_TEST_IMAGES_DIR):
    for file in files:
        if file.endswith(".jpg"):
            TEST_IMAGE_PATHS.append(os.path.join(root, file))

# In[ ]:
tag_tab={}
#Actual detection.
output_dict = run_inference_for_bunch(TEST_IMAGE_PATHS, detection_graph)
print("Images done ", len(TEST_IMAGE_PATHS))
print(tag_tab)
