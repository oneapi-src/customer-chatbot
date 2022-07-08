## Benchmarking Concurrency

To benchmark concurrency, you will first need to clone the torchserve repo at: 
https://github.com/pytorch/serve and follow the installation instructions at https://github.com/pytorch/serve/tree/master/benchmarks#benchmarking-with-apache-bench.

Install the following packages using pip
```shell
pip install torch-model-archiver captum
```

### Preparing the model for benchmarking

#### **1. Convert the model to TorchScript**

To use the trained models in torch-serve, they first need to be converted to a TorchScript model.  To do this, use the `convert_jit.py` script

```shell
usage: convert_jit.py [-h] -s SAVED_MODEL_DIR -o OUTPUT_MODEL [--is_inc_int8]

optional arguments:
  -h, --help            show this help message and exit
  -s SAVED_MODEL_DIR, --saved_model_dir SAVED_MODEL_DIR
                        directory of saved model to benchmark.
  -o OUTPUT_MODEL, --output_model OUTPUT_MODEL
                        saved torchscript (.pt) model
  --is_inc_int8         saved model dir is a quantized int8 model. defaults to False.
```

If the model is not quantized using INC and assuming the saved model is saved in the `../saved_models/intel` directory, we can use

```shell
python convert_jit.py -s ../saved_models/intel -o ../saved_models/intel/convai_jit.pt
```
which will convert the saved model into a torchscript model called `convai_jit.pt`.

If the model is quantized using INC, we need to specify the flag `--is_inc_int8` and then use,

```shell
python convert_jit.py -s ../saved_models/intel_int8 -o ../saved_models/intel_int8/convai_jit.pt --is_inc_int8
```

#### **2. Package the torchscript model using torch-model-archiver**

After creating a TorchScript model, the trained model needs to be packaged to a `.mar` file using torch-model-archiver.  Assuming the serialized model is saved as `convai_jit.pt` in the current directory, a sample command to do this is:

```shell
torch-model-archiver --model-name convai --version 1.0 --serialized-file convai_jit.pt --handler custom_handler.py
```

This will create a file called `convai.mar` which can be used to deploy to torchserve.  

### Benchmarking using the torchserve-benchmarking script

To benchmark this model using the [TorchServe benchmarking tools](https://github.com/pytorch/serve/tree/master/benchmarks#benchmarking-with-apache-bench),

1. Copy the `config.json` file and the `config.properties` file into the cloned `serve/benchmarks` directory.  
   
   ```shell
   cp config.json serve/benchmarks/config.json && cp config.properties serve/benchmarks/config.properties
   ```

2. Modify the `config.json` and `config.properties` to point to the relevant files and the desired experimental parameters
3. Run the benchmark using
   
```shell
python benchmark-ab.py --config config.json
```

We included a simple `input_data.json` file to provide a test input for running the benchmarks.

#### config.json

The available fields for the `config.json` file, as an example, are: 

```python
{'url': "file:///PATH_TO_MAR",
'gpus': '',
'exec_env': 'local',
'batch_size': 1,
'batch_delay': 200,
'workers': 1,
'concurrency': 10,
'requests': 100,
'input': 'PATH_TO_INPUT',
'content_type': 'application/json',
'image': '',
'docker_runtime': '',
'backend_profiling': False,
'config_properties': 'PATH_TO_CONFIG_PROPERTIES',
'inference_model_url': 'predictions/benchmark',
```

### config.properties

The `config.properties` file adjusts the parameters for the torchserve server.

The two most important fields are to either enable or disable Intel Pytorch
Extensions using

`vi``shell
ipex_enable=true
cpu_launcher_enable=true
cpu_launcher_args=
```

The cpu_launcher is optional for `ipex`, the available configuration arguments
can be found at https://intel.github.io/intel-extension-for-pytorch/1.11/tutorials/performance_tuning/torchserve.html#torchserve-with-launcher.


### Running the benchmark

To run the benchmark, go to the `serve/benchmark` directory and run the command

```shell
python benchmark-ab.py --config config.json
```

The reports should be stored in the temporary directory `/tmp/benchmark`.  Measurements for latency and throughput can be found in the file `/tmp/benchmark/ab_report.csv`.