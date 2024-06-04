# Distributional Refinement Network (DRN): Distributional Forecasting via Deep Learning

## Table of Contents  
- [Overview](#overview) 
- [Related Repository](#related-repository)
- [License](#license) 
- [Authors](#authors)
- [Citations](#citation)
- [Contact](#Contact)

## Overview

A key task in actuarial modelling involves modelling the distributional properties of losses. Classic (distributional) regression approaches like Generalized Linear Models (GLMs; Nelder and Wedderburn, 1972) are commonly used, but challenges remain in developing models that can:
1. Allow covariates to flexibly impact different aspects of the conditional distribution,
2. Integrate developments in machine learning and AI to maximise the predictive power while considering (1), and,
3. Maintain a level of interpretability in the model to enhance trust in the model and its outputs, which is often compromised in efforts pursuing (1) and (2).

We tackle this problem by proposing a Distributional Refinement Network (DRN), which combines an inherently interpretable baseline model (such as GLMs) with a flexible neural network--a modified Deep Distribution Regression (DDR; Li et al., 2021) method.
Inspired by the Combined Actuarial Neural Network (CANN; Schelldorfer and W{\''u}thrich, 2019), our approach flexibly refines the entire baseline distribution. 
As a result, the DRN captures varying effects of features across all quantiles, improving predictive performance while maintaining adequate interpretability.
 
This repository yields the results demonstrated in our [DRN paper](https://arxiv.org/abs/2406.00998) (Avanzi et al. 2024).

## Related Repository

The full range of key features, installation procedure, examples can be found in the [package repository](https://github.com/EricTianDong/drn). 

## Related Repository

See [License.md](https://github.com/agi-lab/DRN?tab=GPL-3.0-1-ov-file).

## Authors

- Eric Dong (author, maintainer),
- Patrick Laub (author).

## Citation

```sh
@misc{avanzi2024distributional,
      title={Distributional Refinement Network: Distributional Forecasting via Deep Learning}, 
      author={Benjamin Avanzi and Eric Dong and Patrick J. Laub and Bernard Wong},
      year={2024},
      eprint={2406.00998},
      archivePrefix={arXiv},
      primaryClass={stat.ML}
}
```

## Contact

For any questions or further information, please contact tiandong1999@gmail.com.

