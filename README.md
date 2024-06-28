# Weakly supervised CT scan instance segmentation - Raidium ENS Data Challenge

## Overview 
Implementation of an instance segmentation algorithm for CT scan. It is ccomposed of a hybrid CNN-Transformer architecture (TransUnet) and a permutation invariant loss  variational formulation of the Mumford-Shah
algorithm.

## References 

@misc{chen2021transunet,
      title={TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation}, 
      author={Jieneng Chen and Yongyi Lu and Qihang Yu and Xiangde Luo and Ehsan Adeli and Yan Wang and Le Lu and Alan L. Yuille and Yuyin Zhou},
      year={2021},
      eprint={2102.04306},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}


@misc{yuan2020deep,
      title={Deep Variational Instance Segmentation}, 
      author={Jialin Yuan and Chao Chen and Li Fuxin},
      year={2020},
      eprint={2007.11576},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
