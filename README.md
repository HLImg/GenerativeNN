# Generative Neural Network

[![](https://qiniu.lianghao.work/CodeBase-v0.1-blue)](https://github.com/HLImg/PictoRestore)

## [Deep Generative Models (CS236 Fall 2023)](https://deepgenerativemodels.github.io/)


We can thich of any kind of observed data, say $\mathcal D$, as a finite samples from an **underlying distribution**, say $p_{data}$. The goal of any generative models is then to approximate this data distribution given access to the dataset $\mathcal D$.

### Learning
![image-20231117170535094](https://qiniu.lianghao.work/image-20231117170535094.png)

We will be primarily interested in parametric approximations to the data distribution, which summarize all the information about the dataset $\mathcal D$ in a finte set of parameters. In contrast with non-parametric models, **parametric models scale more efficiently with large datasets but are limited in the family of distributions they can represent**.


![](https://qiniu.lianghao.work/image-20231117170642832.png)

We might be given access to a dataset of dog images $\mathcal D$ and our goal is to learn the parameters of a generative model $\theta$ within a model family $\mathcal M$ such that the models distribution $p_{\theta}$ is close to the data distribution over dogs $p_{data}$. We can specify our goal as the following optimization problem:
$$
\mathop{\min}\limits_{\theta \in \mathcal M} d(p_{data}, p_{\theta})
$$

where $p_{data}$ is accessed via the dataset $\mathcal D$ and $d(\cdot)$ is a notion of distance between probability distributions.

The real world is highly structured and automatically discovering the underlying structure is key to learning generative models. We hope to learn some basic artifacts. Instead of incorporating this prior knowledge explictily, we will hope the model learns the underlying structure directly from data. **Indeed successful learning of generative models will involve instantiating the optimization problem in Eq.(1) in a suitable way**.

### Inference
For a discriminative model such as logistic regression, the fundamental inference task is to predict a label for any given datapoint. Generative models, on the other hand, learn a joint distribution over the entire data.

We can identify three fundamental inference queries for evaluating a generative model.:
1. Density estimation: Given a datapoint $\mathcal x$, what is the probability assigned by the model, i.e., $p_{\theta}(\mathbf x)$?
2. Sampling: How can we generate novel data from the model distribution, i.e., $\mathbf x_{new} \in p_{\theta}(\mathbf x)$？
3. Unsupervied representation learning: How can we learn meaningful feature representations for a datapoint $\mathbf x$ ?




- [ ] [Autoregressive models](./docs/notes/1_autoregressive.md)
- [ ] [Variational Autoencoders](./docs/notes/2_vae.md)
- [ ] [Normalizing](./docs/notes/3_normlizing.md)
- [ ] [Generative Adversarial Networks](./docs/notes/4_gan.md)



## Traning details
### single-machine-multi-gpus

配置文件如下
```yaml
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
downcast_bf16: 'no'
gpu_ids: 0, 1, 2, 3, 4, 5, 6, 7, 8
#machine_rank: 0
main_training_function: main
mixed_precision: 'no'
#num_machines: 1
num_processes: 8
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false

```
启动训练命令如下
```shell
accelerate launch --config_file=resource/acc_config/single_node.yaml --machine_rank=0 --num_machines=1 main.py  --yaml options/nafnet/train_nafnet_wf_32.yaml --train
```
### multi-machines-multi-gpus
相同的配置文件，以2台机器为例（显卡数默认为8）
```shell
# machine-id : 0
accelerate launch --config_file=config.yaml --machine_rank=0 --num_machines=2  main.py  --yaml options/nafnet.yaml
# machine-id : 1
accelerate launch --config_file=config.yaml --machine_rank=1 --num_machines=2  main.py  --yaml options/nafnet.yaml
```
如果多机训练时，在prepare(model)处休眠，可以执行下面代码
```shell
export NCCL_SOCKET_IFNAME=eth0 # 根据自己的网卡设置
export NCCL_IB_DISABLE=1
```
如果多机启动时出现，*nccl error*，在训练之前，执行下面代码
```shell
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TC=106
export NCCL_IB_HCA=$(pushd /sys/class/infiniband/ > /dev/null; for i in mlx*_*; do cat $i/ports/1/gid_attrs/types/* 2>/dev/null | grep v >/dev/null && echo $i ; done; popd > /dev/null)
```
