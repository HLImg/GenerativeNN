# Generative Neural Network

[![](https://qiniu.lianghao.work/CodeBase-v0.1-blue)](https://github.com/HLImg/PictoRestore)



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
accelerate launch --config_file=resource/acc_config/single_node.yaml --machine_rank=0 --num_machines=1 main.py  --yaml options/nafnet/train_nafnet_wf_32.yaml
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
