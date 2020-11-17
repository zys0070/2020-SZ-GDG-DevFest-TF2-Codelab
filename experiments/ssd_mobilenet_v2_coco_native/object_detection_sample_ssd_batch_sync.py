
from __future__ import print_function
import sys
import os
import time
from argparse import ArgumentParser, SUPPRESS
import cv2
import numpy as np
import logging as log
from openvino.inference_engine import IECore

def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-m", "--model", help="Required. Path to an .xml file with a trained model.", required=True,
                      type=str)
    args.add_argument("-i", "--input-path", type=str, required=True,
                      help="Required. Path to an image file")
    args.add_argument("-d", "--device",
                      help="Optional. Specify the target device to infer on; CPU, GPU, HDDL, MYRIAD or HETERO: is "
                           "acceptable. The sample will look for a suitable plugin for device specified. Default "
                           "value is CPU",
                      default="CPU", type=str)
    args.add_argument("--labels", help="Optional. Path to a labels mapping file", default=None, type=str)

    return parser


def main():
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()
    model_xml = args.model
    model_bin = os.path.splitext(model_xml)[0] + ".bin"

    # Plugin initialization for specified device and load extensions library if specified
    log.info("Creating Inference Engine")
    ie = IECore()

    # Read IR
    log.info("Loading network files:\n\t{}\n\t{}".format(model_xml, model_bin))
    net = ie.read_network(model=model_xml, weights=model_bin)

    assert len(net.input_info.keys()) == 1, "Sample supports only single input topologies"
    assert len(net.outputs) == 1, "Sample supports only single output topologies"

    log.info("Preparing input blobs")
    input_blob = next(iter(net.input_info))
    out_blob = next(iter(net.outputs))

    # Loading model to the plugin
    log.info("Loading model to the plugin")
    exec_net = ie.load_network(network=net, device_name=args.device)

    prob_threshold = 0.6
    if args.labels:
        with open(args.labels, 'r') as f:
            labels_map = [x.split(sep=' ', maxsplit=1)[-1].strip() for x in f]
    else:
        labels_map = None

    # Read and pre-process input images
    n, c, h, w = net.input_info[input_blob].input_data.shape
    input_images = [os.path.join(args.input_path, iname) for iname in os.listdir(args.input_path)]
    batch_num = len(input_images)//n if len(input_images) % n == 0 else (len(input_images)//n + 1)
    batch_time_list = []
    for b in range(batch_num):
        images = np.ndarray(shape=(n, c, h, w))
        images_hw = []
        images_orig = []
        pre_start = time.time()
        for i in range(n):
            print(input_images[i])
            if (b*n + i) < len(input_images):
                image = cv2.imread(input_images[b*n + i])
                images_orig.append(image)
                ih, iw = image.shape[:-1]
                images_hw.append((ih, iw))
                if (ih, iw) != (h, w):
                    log.warning("Image {} is resized from {} to {}".format(input_images[i], (ih, iw), (h, w)))
                    image = cv2.resize(image, (w, h))
                image = image.transpose((2, 0, 1))  # Change data layout from HWC to CHW
                images[i] = image
        pre_time = time.time() - pre_start
        print("Batch {} Preprocess complete, run time: {:.3f} ms".format(b, pre_time * 1000))

        # Start sync inference
        inf_start = time.time()
        log.info("Starting inference in synchronous mode")
        res = exec_net.infer(inputs={input_blob: images})
        inf_time = time.time() - inf_start
        print("Batch {} Inference complete, run time: {:.3f} ms".format(b, inf_time * 1000))

        # Processing output blob
        log.info("Processing output blob")
        res = res[out_blob]

        pos_start = time.time()
        # loop through all possible results
        for number, obj in enumerate(res[0][0]): # n * 100 iterations
            # If probability is more than specified threshold, draw and label box 
            imid = np.int(obj[0])
            if obj[2] > prob_threshold and (b*n + imid) < len(input_images):
                ih, iw = images_hw[imid]
                # get coordinates of box containing detected object
                xmin = int(obj[3] * iw)
                ymin = int(obj[4] * ih)
                xmax = int(obj[5] * iw)
                ymax = int(obj[6] * ih)

                # get type of object detected
                class_id = int(obj[1])
                
                # Draw box and label for detected object
                color = (min(class_id * 12.5, 255), 255, 255)
                det_label = labels_map[class_id] if labels_map else str(class_id)
                cv2.rectangle(images_orig[imid], (xmin, ymin), (xmax, ymax), color, 4)
                cv2.putText(images_orig[imid], det_label + ' ' + str(round(obj[2] * 100, 1)) + ' %', (xmin, ymin - 7),
                            cv2.FONT_HERSHEY_COMPLEX, 1, color, 2)

                #print("[{},{}] element, prob = {:.3}    ({},{})-({},{}) batch id : {}\n" \
                #      .format(number, det_label, obj[2], xmin, ymin, xmax, ymax, imid), end="")

        pos_time = time.time() - pos_start
        print("Batch {} Postprocess complete, run time: {:.3f} ms".format(b, pos_time * 1000))

        # total batch time
        batch_time = pre_time + inf_time + pos_time
        batch_time_list.append(batch_time)
        print("Batch {} total run time: {:.3f} ms".format(b, batch_time * 1000))

        for id, image_orig in enumerate(images_orig):
            cv2.imwrite("out_{}.bmp".format(b*n+id), image_orig)

    # End for loop
    print("All batch run time: {:.3f} ms".format(sum(batch_time_list) * 1000))

if __name__ == '__main__':
    sys.exit(main() or 0)