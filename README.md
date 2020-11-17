# 2020-SZ-GDG-DevFest-TF2-Codelab

## 现场任务
* 在 TF2 中训练 MNIST 模型并使用 OpenVINO 完成本地运行
* 使用 OpenVINO 转换 TF2 OD API 中的 SSD 预训练模型并完成本地部署和优化 （给定 coco 验证集 100 张图，在正确推理的基础上逐步提升性能，可使用 batch、异步 Async API等，可尝试跨平台加速）


## 环境准备
* OS: Linux 18.04+ / Windows 10 / masOS
* TensorFlow 2.x
* [OpenVINO 2021.1](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit.html)

## 目录结构
* data: coco 测试数据
* docs: 主题分享 PPT，TF2 OD 模型转换说明
* experiments: TF2 训练 MNIST 模型以及使用 OpenVINO 本地运行参考
* models: 包含 TF2 OD 预训练模型 ssd_mibilenet_v2_coco 的不同精度 FP32 | FP16 | INT8 的中间表示（IR）
* openvino: 最新版本 OpenVINO 2021.1 的本地修改文件，用于支持 TF2 OD SSD 模型转换

## OpenVINO Codelab 其它参考
### 使用 OpenVINO Model Optimizer 转换 TF2 Object Detection SSD 预训练模型
* 从 [TF2 Detection Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md) 下载 SSD MobileNet v2 320x320
```
tar -xf ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz
```
* 添加本地修改文件
```
cd <OPENVINO_INSTALL_DIR>/deployment_tools/model_optimizer/extensions/front/tf
cp ObjectDetectionAPI.py ObjectDetectionAPI.py.bak
cp <THIS_REPO_DIR>/openvino/deployment_tools/model_optimizer/extensions/front/tf/ObjectDetectionAPI.py .
cp <THIS_REPO_DIR>/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support_tf2.json .
```
* 模型转换
```
mo_tf.py --saved_model_dir <path to ssd_mobilenet_v2_320x320_coco17_tpu-8/saved_model> \
         --input_shape [1,300,300,3] \
         --transformations_config <OPENVINO_INSTALL_DIR>/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support_tf2.json \
         --tensorflow_object_detection_api_pipeline_config pipeline.config \
         --reverse_input_channels
```

### 使用 Async API 加速推理
除 <OPENVINO_INSTALL_DIR>/deployment_tools/inference_engine/samples/python/classification_sample_async 示例中基于 callback 机制实现 Async API 外，此处给出另一种基于 wait completion 的实现供参考。
```
# ie = IECore()
# input_blob, out_blob are name of input and output node
# exec_net = ie.load_network()
# number_infer_requests: maximum number of inference requests
current_inference = 0
previous_inference = 1 - number_infer_requests
infer_requests = exec_net.requests
try:
    while (inference not done)
        # Preprocess and fill images
        # Start inference and return immediately
        exec_net.start_async(request_id=current_inference, inputs={input_blob: images})
        if previous_inference >= 0:
            # ms_timeout = 0      - Return status immediately whether inference is complete or not
            # ms_timeout = -1     - Wait until inference is complete, then return (default)
            # ms_timeout = <time> - Wait time in milliseconds or until inference complete
            status = infer_requests[previous_inference].wait()
            if status is not 0:
                raise Exception("Infer request not completed successfully")
            # Parse and Postprocess inference results                    
            res = exec_net.requests[previous_inference].output_blobs[out_blob].buffer
            ...
        # Increment counter for the inference queue and roll them over if necessary 
        current_inference += 1
        if current_inference >= args.number_infer_requests:
            current_inference = 0
        previous_inference += 1
        if previous_inference >= args.number_infer_requests:
            previous_inference = 0
finally:
    del exec_net
```

### 使用 OpenVINO Post-training Optimization Tool 进行 INT8 模型量化
* 从 [COCO](https://cocodataset.org/) 官方下载 2017 val images 和 val annotations
* 使用 POT 工具中的 DefaultQuantization 算法对模型进行 INT8 量化
```
cd <OPENVINO_INSTALL_DIR>/deployment_tools/deployment_tools/tools/post_training_optimization_toolkit
mkdir ssd_mobilenet_v2_coco-tf2
cd ssd_mobilenet_v2_coco-tf2/
cp <path to converted SSD model .bin & .xml> .
cp <path to downloaded coco val2017 & instances_val2017.json> .
cp <OPENVINO_INSTALL_DIR>/deployment_tools/open_model_zoo/tools/accuracy_checker/configs/ssd_mobilenet_v2_coco.yml .

# 打开 ssd_mobilenet_v2_coco.yml 并添加 model, weights, datasets 信息
  models:
    - name:  ssd_mobilenet_v2_coco
      launchers:
        - framework: dlsdk
          device: CPU
          model: ssd_mobilenet_v2_coco.xml
          weights: ssd_mobilenet_v2_coco.bin
          adapter: ssd
      datasets:
        - name: ms_coco_detection_91_classes
          data_source: val2017
          annotation_conversion:
            converter: mscoco_detection
            annotation_file: instances_val2017.json
            has_background: True
            sort_annotations: True
            use_full_label_map: True
          preprocessing:
            - type: resize
              size: 300
          postprocessing:
            - type: resize_prediction_boxes
          metrics:
            - type: coco_precision
# 保存并关闭 ssd_mobilenet_v2_coco.yml

accuracy_check -c ssd_mobilenet_v2_coco.yml

touch ssd_mobilenet_v2_coco_pot.json

# 打开 ssd_mobilenet_v2_coco_pot.json 并添加如下配置信息
  {
      "model": {
          "model_name": "ssd_mobilenet_v2_coco",
          "model": "ssd_mobilenet_v2_coco.xml",
          "weights": "ssd_mobilenet_v2_coco.bin"
      },
      "engine": {
          "config": "ssd_mobilenet_v2_coco.yml"
      },
      "compression": {
          "target_device": "CPU",
          "algorithms": [
              {
                  "name": "DefaultQuantization",
                  "params": {
                      "preset": "performance",
                      "stat_subset_size": 300
                  }
              }
          ]
      }
  }
# 保存并关闭 ssd_mobilenet_v2_coco_pot.json

pot -c ssd_mobilenet_v2_coco_pot.json -e
```
以上 accuracy checker 和 pot 运行结束后，生成的 INT8 模型保存在 results 目录下。
