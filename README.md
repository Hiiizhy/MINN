# MINN

This repository contains Pytorch implementation of [MINN](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10095063):

> Multi-aspect Interest Neighbor-augmented Network for Next-basket Recommendation.
> Zhiying Deng, Jianjun Li, Zhiqiang Guo, Guohui Li.
> The 48th IEEE International Conference on Acoustics, Speech, & Signal Processing (ICASSP 2023).

MINN decouples the multi-aspect interests of users and mines semantic neighbors under different interests to enhance the interest representation of users for the next-basket recommendation.

## Environments

- torch 1.10.1+cuda 11.2
- python 3.6.13
- numpy 1.19.5
- scipy 1.5.4
- scikit-learn 0.23.2

## Running the code

```python
$ cd src
$ python main.py --dataset TaFeng --lr 0.001 --l2 0.0001 --asp 2 --h 5 --nbNUM 4 --batch_size 100 --dim 32 --udim 1 --isTrain 0 --epochs 20
```

## Reference

```
@inproceedings{deng2023multi,
  title={Multi-Aspect Interest Neighbor-Augmented Network for Next-Basket Recommendation},
  author={Deng, Zhiying and Li, Jianjun and Guo, Zhiqiang and Li, Guohui},
  booktitle={ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1--5},
  year={2023},
  organization={IEEE}
}
```

