# README

## Introduction
This repository contains a **re-implementation** of our deep learning training infrastructure, described in the paper "AntMan: Dynamic Scaling on GPU Clusters for Deep Learning" ([OSDI'20](https://www.usenix.org/conference/osdi20/presentation/xiao)).

**Note**

0. The original implementation of our paper is based on [FUXI](https://dl.acm.org/doi/10.14778/2733004.2733012), which is tightly coupled with the internal infrastructure of Alibaba. The goal of this project is to provide a cloud-native solution to demonstrate the feature of the paper and benenfit the community.
1. This is a **WIP** project. Please grant us several days to fix the missing components with code cleaning and show the end-to-end demo with some benchmarks. We are working hard to achieve that. More detailed documents are on the way.
2. The implementation of our kubernetes scheduler is only tested in the ACK cluster of alibaba cloud, based on Kubernetes V1.18. The deployement script we provide may not be able to apply in other kubernetes infrastructures directly.

## Modules
The development of this repository is based on some open-source repositories.

### k8s-related
1. [KubeDL](https://github.com/alibaba/kubedl): an all-in-one operator, responsible to reconcile tfjobs
2. [Scheduler Plugins](https://github.com/kubernetes-sigs/scheduler-plugins/tree/release-1.18): a k8s cluster scheduler, responsible to schedule DL GPU pods for both resource-guarantee/opportunistic jobs
3. [k8s-device-plugin](https://github.com/NVIDIA/k8s-device-plugin): report GPU resources to k8s

### TensorFlow
The dynamic scaling mechianism is initially implemented in PAI-TF, a highly-optimized TensorFlow verison used in Alibaba. 
We port the core implementation to the open-source TensorFlow v1.15.
1. [TensorFlow](https://github.com/tensorflow/tensorflow/tree/r1.15)
