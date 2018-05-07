import sys,os
import argparse
import shutil
import tensorflow as tf
from tensorflow.core.framework import graph_pb2
from tensorflow.python.platform import gfile
from tensorflow.python.framework import dtypes
from tensorflow.python.client import session
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
from tensorflow.python.summary import summary
from tensorflow.python.tools.freeze_graph import freeze_graph
from tensorflow.python.tools.optimize_for_inference_lib import optimize_for_inference

parser = argparse.ArgumentParser()
parser.add_argument('model_dir',default=None, type=str, help='estimator model dir')
parser.add_argument('--input',default="Reshape", type=str, help='input node name')
parser.add_argument('--output',default="softmax_tensor", type=str, help='output node name')
parser.add_argument('--output_file',default="optimized.pb", type=str, help='output file name')

def display_nodes(nodes):
    for i, node in enumerate(nodes):
        print('%d %s %s' % (i, node.name, node.op))
        [print(u'└─── %d ─ %s' % (i, n)) for i, n in enumerate(node.input)]
def strip_dropout(nodes):
    pass

def freeze_model(model_dir,freezed_file_name="freezed.pb",output="output"):
    print(model_dir)
    files = os.listdir(model_dir)
    #[print('%s' % (i)) for i in files]
    #model.ckpt-200.data-00000-of-00001 model.ckpt-200.meta
    count = -1
    for i in files:
        splt = i.split('.')
        if len(splt)>2 and splt[0]=="model":
            s = splt[1].find('-')
            if s != -1:
                v = int(splt[1][s+1:])
                if v > count:
                    count = v
    if count == -1:
        print("can not find model file")
        return
    #print("model num:",count)
    model_meta_file = model_dir+os.sep+"model.ckpt-"+str(count)+".meta"
    model_data_file = model_dir+os.sep+"model.ckpt-"+str(count)
    print("model meta:",model_meta_file)
    print("model data:",model_data_file)
    freeze_graph("",#input_graph,
                 "",#input_saver,
                 True,#input_binary,
                 model_data_file,#input_checkpoint,
                 output,#output_node_names,
                 "save/restore_all",#restore_op_name,
                 "save/Const:0",#filename_tensor_name,
                 freezed_file_name,#output_graph,
                 True,#clear_devices,
                 "",#initializer_nodes,
                 input_meta_graph=model_meta_file)

def optimize_remove_dropout(freezed_file,output_file,input,output):
    input_graph_def = graph_pb2.GraphDef()
    with gfile.Open(freezed_file, "rb") as f:
        input_graph_def.ParseFromString(f.read())
    output_graph_def = optimize_for_inference(
        input_graph_def,
        input.split(","),
        output.split(","),
        dtypes.float32.as_datatype_enum,#FLAGS.placeholder_type_enum,
        True#toco_compatible
        )
    #remove dropout layer
    dropout_input = None
    after_dropout_nodes=[]
    for i, node in enumerate(output_graph_def.node):
        print('%d %s %s' % (i, node.name, node.op))
        [print(u'└─── %d ─ %s' % (i, n)) for i, n in enumerate(node.input)]
        if node.name.startswith("dropout"):
            for i in node.input:
                if not i.startswith("dropout"):
                    if dropout_input is None:
                        dropout_input = i
                        print("\tdropout_input:",dropout_input)
                    elif dropout_input != i:
                        print("error dropout has more than one input")
                        return
        else:
            if dropout_input:
                for i,node_input in enumerate(node.input):
                    if node_input.startswith("dropout"):
                        print("\tnode input is dropout:",node.name)
                        node.input[i] = dropout_input
            after_dropout_nodes.append(node) 
    print("after remove dropout")
    display_nodes(after_dropout_nodes)
    # Save graph
    output_graph = graph_pb2.GraphDef()
    output_graph.node.extend(after_dropout_nodes)
    with tf.gfile.GFile(output_file, 'w') as f:
        f.write(output_graph.SerializeToString())

    tf_board_dir = output_file+".tb"
    shutil.rmtree(tf_board_dir)
    with session.Session(graph=ops.Graph()) as sess:
        importer.import_graph_def(output_graph)
        pb_visual_writer = summary.FileWriter(tf_board_dir)
        pb_visual_writer.add_graph(sess.graph)
        print("Model Imported. Visualize by running: "
          "tensorboard --logdir={}".format(tf_board_dir))

def main(argv):
    args = parser.parse_args(argv[1:])
    model_dir = args.model_dir
    input_nodes = args.input
    output_nodes = args.output
    output_file_name = args.output_file
    freezed_file_name = "freezed.pb"

    freeze_model(model_dir,freezed_file_name,output_nodes)
    optimize_remove_dropout(freezed_file_name,output_file_name,input_nodes,output_nodes)


if __name__ == '__main__':
    main(sys.argv)
