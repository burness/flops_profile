# Flops Profiler In Pytorch

> Measures the parameters, latency, and floating-point operations of your model.

  - [Overview](#overview)
  - [Flops Measurement](#flops-measurement)
  - [Multi-GPU, Multi-node, Data Parallelism, and Model Parallelism](#multi-gpu-multi-node-data-parallelism-and-model-parallelism)
  - [Usage](#usage)
## Overview

Effective use of hardware resources is critical to good performance, but performance inefficiency in existing implementations for large-scale model training and inference are often hard to spot and attributed to specific module components.  Flops Profiler helps users easily measure both the model training/inference speed (latency, throughput) and efficiency (floating-point operations per second, i.e., FLOPS) of a model and its submodules, with an eye towards eliminating inefficiencies in existing implementations.

Below is an example output for BERT-Large(NVIDIA) on an A100 GPU with batch size `80`:

```shell
--------------------------  Flops Profiler --------------------------
Profile Summary at step 10:
Notations:
data parallel size (dp_size), model parallel size(mp_size),
number of parameters (params), number of multiply-accumulate operations(MACs),
number of floating-point operations (flops), floating-point operations per second (FLOPS),
fwd latency (forward propagation latency), bwd latency (backward propagation latency),
step (weights update latency), iter latency (sum of fwd, bwd and step latency)

world size:                                                   1
data parallel size:                                           1
model parallel size:                                          1
batch size per GPU:                                           80
params per gpu:                                               336.23 M
params of model = params per GPU * mp_size:                   336.23 M
fwd MACs per GPU:                                             3139.93 G
fwd flops per GPU:                                            6279.86 G
fwd flops of model = fwd flops per GPU * mp_size:             6279.86 G
fwd latency:                                                  76.67 ms
bwd latency:                                                  108.02 ms
fwd FLOPS per GPU = fwd flops per GPU / fwd latency:          81.9 TFLOPS
bwd FLOPS per GPU = 2 * fwd flops per GPU / bwd latency:      116.27 TFLOPS
fwd+bwd FLOPS per GPU = 3 * fwd flops per GPU / (fwd+bwd latency):   102.0 TFLOPS
step latency:                                                 34.09 us
iter latency:                                                 184.73 ms
samples/second:                                               433.07

----------------------------- Aggregated Profile per GPU -----------------------------
Top modules in terms of params, MACs or fwd latency at different model depths:
depth 0:
    params      - {'BertForPreTrainingPreLN': '336.23 M'}
    MACs        - {'BertForPreTrainingPreLN': '3139.93 GMACs'}
    fwd latency - {'BertForPreTrainingPreLN': '76.39 ms'}
depth 1:
    params      - {'BertModel': '335.15 M', 'BertPreTrainingHeads': '32.34 M'}
    MACs        - {'BertModel': '3092.96 GMACs', 'BertPreTrainingHeads': '46.97 GMACs'}
    fwd latency - {'BertModel': '34.29 ms', 'BertPreTrainingHeads': '3.23 ms'}
depth 2:
    params      - {'BertEncoder': '302.31 M', 'BertLMPredictionHead': '32.34 M'}
    MACs        - {'BertEncoder': '3092.88 GMACs', 'BertLMPredictionHead': '46.97 GMACs'}
    fwd latency - {'BertEncoder': '33.45 ms', 'BertLMPredictionHead': '2.61 ms'}
depth 3:
    params      - {'ModuleList': '302.31 M', 'Embedding': '31.79 M', 'Linear': '31.26 M'}
    MACs        - {'ModuleList': '3092.88 GMACs', 'Linear': '36.23 GMACs'}
    fwd latency - {'ModuleList': '33.11 ms', 'BertPredictionHeadTransform': '1.83 ms''}
depth 4:
    params      - {'BertLayer': '302.31 M', 'LinearActivation': '1.05 M''}
    MACs        - {'BertLayer': '3092.88 GMACs', 'LinearActivation': '10.74 GMACs'}
    fwd latency - {'BertLayer': '33.11 ms', 'LinearActivation': '1.43 ms'}
depth 5:
    params      - {'BertAttention': '100.76 M', 'BertIntermediate': '100.76 M'}
    MACs        - {'BertAttention': '1031.3 GMACs', 'BertIntermediate': '1030.79 GMACs'}
    fwd latency - {'BertAttention': '19.83 ms', 'BertOutput': '4.38 ms'}
depth 6:
    params      - {'LinearActivation': '100.76 M', 'Linear': '100.69 M'}
    MACs        - {'LinearActivation': '1030.79 GMACs', 'Linear': '1030.79 GMACs'}
    fwd latency - {'BertSelfAttention': '16.29 ms', 'LinearActivation': '3.48 ms'}

------------------------------ Detailed Profile per GPU ------------------------------
Each module profile is listed after its name in the following order:
params, percentage of total params, MACs, percentage of total MACs, fwd latency, percentage of total fwd latency, fwd FLOPS

BertForPreTrainingPreLN(
  336.23 M, 100.00% Params, 3139.93 GMACs, 100.00% MACs, 76.39 ms, 100.00% latency, 82.21 TFLOPS,
  (bert): BertModel(
    335.15 M, 99.68% Params, 3092.96 GMACs, 98.50% MACs, 34.29 ms, 44.89% latency, 180.4 TFLOPS,
    (embeddings): BertEmbeddings(...)
    (encoder): BertEncoder(
      302.31 M, 89.91% Params, 3092.88 GMACs, 98.50% MACs, 33.45 ms, 43.79% latency, 184.93 TFLOPS,
      (FinalLayerNorm): FusedLayerNorm(...)
      (layer): ModuleList(
        302.31 M, 89.91% Params, 3092.88 GMACs, 98.50% MACs, 33.11 ms, 43.35% latency, 186.8 TFLOPS,
        (0): BertLayer(
          12.6 M, 3.75% Params, 128.87 GMACs, 4.10% MACs, 1.29 ms, 1.69% latency, 199.49 TFLOPS,
          (attention): BertAttention(
            4.2 M, 1.25% Params, 42.97 GMACs, 1.37% MACs, 833.75 us, 1.09% latency, 103.08 TFLOPS,
            (self): BertSelfAttention(
              3.15 M, 0.94% Params, 32.23 GMACs, 1.03% MACs, 699.04 us, 0.92% latency, 92.22 TFLOPS,
              (query): Linear(1.05 M, 0.31% Params, 10.74 GMACs, 0.34% MACs, 182.39 us, 0.24% latency, 117.74 TFLOPS,...)
              (key): Linear(1.05 M, 0.31% Params, 10.74 GMACs, 0.34% MACs, 57.22 us, 0.07% latency, 375.3 TFLOPS,...)
              (value): Linear(1.05 M, 0.31% Params, 10.74 GMACs, 0.34% MACs, 53.17 us, 0.07% latency, 403.91 TFLOPS,...)
              (dropout): Dropout(...)
              (softmax): Softmax(...)
            )
            (output): BertSelfOutput(
              1.05 M, 0.31% Params, 10.74 GMACs, 0.34% MACs, 114.68 us, 0.15% latency, 187.26 TFLOPS,
              (dense): Linear(1.05 M, 0.31% Params, 10.74 GMACs, 0.34% MACs, 64.13 us, 0.08% latency, 334.84 TFLOPS, ...)
              (dropout): Dropout(...)
            )
          )
          (PreAttentionLayerNorm): FusedLayerNorm(...)
          (PostAttentionLayerNorm): FusedLayerNorm(...)
          (intermediate): BertIntermediate(
            4.2 M, 1.25% Params, 42.95 GMACs, 1.37% MACs, 186.68 us, 0.24% latency, 460.14 TFLOPS,
            (dense_act): LinearActivation(4.2 M, 1.25% Params, 42.95 GMACs, 1.37% MACs, 175.0 us, 0.23% latency, 490.86 TFLOPS,...)
          )
          (output): BertOutput(
            4.2 M, 1.25% Params, 42.95 GMACs, 1.37% MACs, 116.83 us, 0.15% latency, 735.28 TFLOPS,
            (dense): Linear(4.2 M, 1.25% Params, 42.95 GMACs, 1.37% MACs, 65.57 us, 0.09% latency, 1310.14 TFLOPS,...)
            (dropout): Dropout(...)
          )
        )
        ...
        (23): BertLayer(...)
      )
    )
    (pooler): BertPooler(...)
  )
  (cls): BertPreTrainingHeads(...)
)
------------------------------------------------------------------------------

```

In the summary profile, the  Flops Profiler outputs the number of parameters, floating-point operations (flops), FLOPS, latency, and throughput in samples/second of the model. This profile shows how much performance gap (compared to the peak hardware performance) the current model execution has and helps users tune the training or inference setup (e.g., hyperparameters, data parallelism, model parallelism, system configurations, etc.) for better performance.

The  Flops Profiler also measures significant modules at different model depths (aggregated profile) and module-specific profile in the model architecture (detailed profile). Using these profiles  users can understand how each layer or submodule contributes to the overall model complexity/performance. Then users can adjust or refactor the model design to achieve better performance. For example, using the profiler,  users can quantitatively tell if stacking smaller layers is lighter or more performant than having bigger ones. The aggregated and detailed profiles also allow users to quickly identify bottleneck modules. In the BERT-Large example above, using the  Flops Profiler, we find that BertLayer is the most significant layer and contains quite a few dropout, softmax, and layer norm along with linear modules. These modules are not heavy in flops and would trigger many GPU kernel invocations and create excessive read/write requests to memory. The pattern shown in the detailed profile suggests this is a perfect match for kernel fusion, and we developed fused transformer-kernels to reduce data movement (See Bert). After applying our optimizations, we see a 25% improvement in FLOPS per GPU and overall training samples/second in the  Flops Profiler output.

The  Flops Profiler can be used with the  runtime without any user code change or be used independently from  as a standalone package. When using  for model training, the profiler can be enabled in the  configuration file. As a standalone package, the profiler API can be used in both training and inference code. The  profiler is still under active development and includes just initial features.  Stay connected for more exciting features to be added soon.

## Flops Measurement

Similar to existing flops calculation tools or methods, the  Flops Profiler measures the flops of the forward pass of a module and the flops of the backward pass is estimated as `2` times of that of the forward pass.
Different from the PyTorch profiler which calculates the flops of PyTorch operators, the  Flops Profiler measures the flops within modules in a model and provides more insights to the users about the model execution.
The flops estimation is partly inspired by [ptflops](https://github.com/sovrasov/flops-counter.pytorch) with the major difference being that the  Flops Profiler not only supports flops computation directly at module level, but can also capture ```torch.nn.functional``` invoked in a module to estimate the flops.

## Usage

The  Flops Profiler can be used with the  runtime or as a standalone package. When using  for model training, the profiler can be configured in the  configuration file without user code change. To use the flops profiler outside of the  runtime, one can simply install  and import the flops_profiler package to use the APIs directly. Examples of each usage are given below.

  - [Usage With the  Runtime](#usage-with-the--runtime)
    - [Example: Megatron-LM](#example-megatron-lm)
  - [Usage Outside the  Runtime](#usage-outside-the--runtime)
    - [In Model Inference](#in-model-inference)
      - [Example: AlexNet](#example-alexnet)
      - [Example: Bert](#example-bert)
    - [In Model Training Workflow](#in-model-training-workflow)
      - [Example Training Workflow](#example-training-workflow)
### Usage With the  Runtime

When using  for model training, the profiler can be configured in the  configuration file. No explicit API calls are needed to use the profiler. The profiler can be enabled by adding the following field to the `_config` json file. Refer to [flops profiler](https://www..ai/docs/config-json/#flops-profiler) for details.

```json
{
  "flops_profiler": {
    "enabled": true,
    "profile_step": 1,
    "module_depth": -1,
    "top_modules": 1,
    "detailed": true,
    "output_file": null
    }
}
```

###  Usage Outside the  Runtime

The profiler can be used as a standalone package outside of the  runtime.
One can simply install  and import the `flops_profiler` package to use the APIs directly.
Refer to [installation of ](https://www..ai/getting-started/#installation) for installing .

#### In Model Inference

To profile a trained model in inference, use the `get_model_profile` function.
Examples are given below.

##### Example: AlexNet

The following example shows how to profile AlexNet using the  flops profiler.

```python
import torchvision.models as models
import torch
from .profiling.flops_profiler import get_model_profile

model = models.alexnet()
batch_size = 256
flops, macs, params = get_model_profile(model=model, # model
                                input_shape=(batch_size, 3, 224, 224), # input shape to the model. If specified, the model takes a tensor with this shape as the only positional argument.
                                args=None, # list of positional arguments to the model.
                                kwargs=None, # dictionary of keyword arguments to the model.
                                print_profile=True, # prints the model graph with the measured profile attached to each module
                                detailed=True, # print the detailed profile
                                module_depth=-1, # depth into the nested modules, with -1 being the inner most modules
                                top_modules=1, # the number of top modules to print aggregated profile
                                warm_up=10, # the number of warm-ups before measuring the time of each module
                                as_string=True, # print raw numbers (e.g. 1000) or as human-readable strings (e.g. 1k)
                                output_file=None, # path to the output file. If None, the profiler prints to stdout.
                                ignore_modules=None) # the list of modules to ignore in the profiling
```

##### Example: Bert

```python
from functools import partial
import torch
from transformers import BertForSequenceClassification, BertTokenizer
from .profiling.flops_profiler import get_model_profile
from .accelerator import get_accelerator


def bert_input_constructor(batch_size, seq_len, tokenizer):
    fake_seq = ""
    for _ in range(seq_len - 2):  # ignore the two special tokens [CLS] and [SEP]
      fake_seq += tokenizer.pad_token
    inputs = tokenizer([fake_seq] * batch_size,
                       padding=True,
                       truncation=True,
                       return_tensors="pt")
    labels = torch.tensor([1] * batch_size)
    inputs = dict(inputs)
    inputs.update({"labels": labels})
    return inputs


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
batch_size = 4
seq_len = 128
enable_profile = True
if enable_profile:
  flops, macs, params = get_model_profile(
      model,
      kwargs=bert_input_constructor(batch_size, seq_len, tokenizer),
      print_profile=True,
      detailed=True,
  )
else:
  inputs = bert_input_constructor((batch_size, seq_len), tokenizer)
  outputs = model(inputs)
```

#### In Model Training Workflow

To profile model forward in a training workflow, use the `FlopsProfiler`class.
The `FlopsProfiler`class provides the following methods:
  * `start_profile()` - starts profiling
  * `get_total_flops(as_string=False)` - returns the total number of floating-point operations in the model
  * `get_total_macs(as_string=False)` - returns the total number of MACs in the model
  * `get_total_params(as_string=False)` - returns the total number of parameters in the model
  * `print_model_profile(profile_step=1, module_depth=-1, top_modules=1, detailed=True, output_file=None)` - prints the model profile
  * `stop_profile()` - stops profiling. This stops the flops counting in the model.
  * `end_profile()` - cleans up. This cleans up the profile attributes added to the model during the profiling. This should be invoked at the end of the profiling and AFTER `get_total_flops`, `get_total_macs`, `get_total_params` or `print_model_profile`.

##### Example Training Workflow

Below is an example of this usage in a typical training workflow.

```python
from .profiling.flops_profiler import FlopsProfiler

model = Model()
prof = FlopsProfiler(model)

profile_step = 5
print_profile= True

for step, batch in enumerate(data_loader):
  # start profiling at training step "profile_step"
  if step == profile_step:
    prof.start_profile()

  # forward() method
  loss = model(batch)

  # end profiling and print output
  if step == profile_step: # if using multi nodes, check global_rank == 0 as well
    prof.stop_profile()
    flops = prof.get_total_flops()
    macs = prof.get_total_macs()
    params = prof.get_total_params()
    if print_profile:
        prof.print_model_profile(profile_step=profile_step)
    prof.end_profile()

  # runs backpropagation
  loss.backward()

  # weight update
  optimizer.step()

```
