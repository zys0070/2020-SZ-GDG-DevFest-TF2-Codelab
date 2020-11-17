
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
    args.add_argument("-nireq", "--number_infer_requests", type=int, default=2,
                      help="Optional. Number of parallel inference requests (default is 2)")

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

    prob_threshold = 0.6
    if args.labels:
        with open(args.labels, 'r') as f:
            labels_map = [x.split(sep=' ', maxsplit=1)[-1].strip() for x in f]
    else:
        labels_map = None

    # Loading model to the plugin
    log.info("Loading model to the plugin")
    exec_net = ie.load_network(network=net, num_requests=args.number_infer_requests, device_name=args.device)

    # Read and pre-process input images
    n, c, h, w = net.input_info[input_blob].input_data.shape
    input_images = [os.path.join(args.input_path, iname) for iname in os.listdir(args.input_path)]
    batch_num = len(input_images)//n if len(input_images) % n == 0 else (len(input_images)//n + 1)

    log.info("Starting inference in async mode, {} requests in parallel...".format(args.number_infer_requests))
    current_inference = 0
    previous_inference = 1 - args.number_infer_requests
    infer_requests = exec_net.requests
    send_batch_count = 0
    parse_batch_count = 0
    images_hw = {}
    images_orig = {}
    
    try:
        infer_time_start = time.time()
        while parse_batch_count < batch_num:
            if send_batch_count < batch_num:
                # Preprocess
                images = np.ndarray(shape=(n, c, h, w))
                for i in range(n):
                    if (send_batch_count * n + i) < len(input_images):
                        image = cv2.imread(input_images[send_batch_count * n + i])
                        images_orig[send_batch_count * n + i] = image
                        ih, iw = image.shape[:-1]
                        images_hw[send_batch_count * n + i] = (ih, iw)
                        if (ih, iw) != (h, w):
                            log.warning("Image {} is resized from {} to {}".format(input_images[i], (ih, iw), (h, w)))
                            image = cv2.resize(image, (w, h))
                        image = image.transpose((2, 0, 1))  # Change data layout from HWC to CHW
                        images[i] = image
                # Increment send batch counter
                send_batch_count += 1
                # Async
                exec_net.start_async(request_id=current_inference, inputs={input_blob: images})
            
            # Retrieve the output of an earlier inference request
            if previous_inference >= 0:
                status = infer_requests[previous_inference].wait()
                if status is not 0:
                    raise Exception("Infer request not completed successfully")
                # Parse inference results                    
                res = exec_net.requests[previous_inference].output_blobs[out_blob].buffer
                for obj in res[0][0]:
                    imid = np.int(obj[0])
                    # If probability is more than specified threshold, draw and label box 
                    if obj[2] > prob_threshold and (parse_batch_count * n + imid) < len(input_images):
                        ih, iw = images_hw[parse_batch_count * n + imid]
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
                        cv2.rectangle(images_orig[parse_batch_count * n + imid], (xmin, ymin), (xmax, ymax), color, 4)
                        cv2.putText(images_orig[parse_batch_count * n + imid],
                                    det_label + ' ' + str(round(obj[2] * 100, 1)) + ' %', (xmin, ymin - 7),
                                    cv2.FONT_HERSHEY_COMPLEX, 1, color, 2)
                # Increment parse batch counter
                parse_batch_count += 1

            # Increment counter for the inference queue and roll them over if necessary 
            current_inference += 1
            if current_inference >= args.number_infer_requests:
                current_inference = 0

            previous_inference += 1
            if previous_inference >= args.number_infer_requests:
                previous_inference = 0

        for key in images_orig:
            cv2.imwrite("out_{}.bmp".format(key), images_orig[key])

        # End while loop
        total_time = time.time() - infer_time_start
        print("Total run time: {:.3f} ms".format(total_time * 1000))

    finally:
        log.info("Processing done...")
        del exec_net

if __name__ == '__main__':
    sys.exit(main() or 0)