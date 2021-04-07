# **Bayesian multiscale deep generative model for the solution of high-dimensional inverse problems**

[Bayesian multiscale deep generative model for the solution of high-dimensional inverse problems](https://doi.org/10.1016/j.jcp.2018.04.018)

Yingzhi Xia, [Nicholas Zabaras](https://www.zabaras.com)

PyTorch Implementation of Bayesian inverse problems.  The parameterization using the proposed multiscale deep generative model can exploite the multiscale nature of the parameter of interest. Combining the multiscale generative model with Markov Chain Monte Carlo (MCMC), inference across scales is achieved enabling us to efficiently obtain posterior parameter samples at various scales. 

The developed method is demonstrated with two types of permeability estimation for flow in heterogeneous media. One is a Gaussian random field (GRF) with uncertain length scales, and the other is channelized permeability with the two regions defined by different GRFs.

# Insight

- Based on the vanilla VAE, we extend and derive the MDGM, which can generate spatial parameters at various scales with an appropriately designed latent space.

- The proposed multiscale inference method performs efficiently inference across scales based on the MDGM.

- A flexible scheme allows efficient estimation of rough/global parameter features with coarse-scale inference and parameter refinement with fine-scale inference

- The proposed method is demonstrated in Gaussian and non-Gaussian inversion tasks.

  

## Dependencies

- python 3
- PyTorch 0.4
- Fenics
- h5py
- matplotlib
- seaborn


## Installation

- Install PyTorch and other dependencies

- Clone this repo:

```
git clone https://https://github.com/zabaras/MH-MDGM.git
```


## Dataset

The datasets used have been uploaded to Google Drive and can be downloaded corresponding links

Gaussian: Please change the direction and download all files from the link:

```bash
cd MDGM/Gaussian/data
```

Link: https://drive.google.com/drive/folders/1CkktP9v-k74ALU2CWHmB9xS_Vcm-cSI5

Channel: Please change the direction and download all files from the link:

```bash
cd MDGM/Channel/data
```

Link: https://drive.google.com/drive/folders/1ovrdWre3yMEmbDY5tPCECxzlSCLpc5Bq



## Multiscale deep generative model (MDGM)

To train MDGM, you can change the directions and run corresponding  example (Gaussian/ Channel) for different scales: 

The list of the generative model realized in the paper (refer the paper for details):

| Gaussian        | Channel                        |
| --------------- | ------------------------------ |
| 16 / 64 / 16_64 | 16/ 64/ 16_32/ 16_64/ 16_32_64 |

For example, to train Channel for the 16 and 16_32 experiment:

```bash
cd MDGM
python Channel/VAE_train_16.py
python Channel/VAE_train_16_32.py
```

## Inverse model

To estimate the parameter in the Bayesian inverse problem, you can change the directions and run corresponding  example (Gaussian/ Channel) for inverse modeling using MCMC (refer the paper for details): 

For example, to estimate  the Channel random field for the 16 and 16_32 experiment:

```bash
cd Inversion
python Channel/inversion_16.py
python Channel/inversion_16_32.py
```

## Citation

If you find this repo useful for your research, please consider to cite:

```latex
@article{xia2021bayesian,
  title={Bayesian multiscale deep generative model for the solution of high-dimensional inverse problems},
  author={Xia, Yingzhi and Zabaras, Nicholas},
  journal={arXiv preprint arXiv:2102.03169},
  year={2021}
}
```

## Questions

For any questions or comments regarding this paper, please contact Yingzhi Xia via [xiayzh@shanghaitech.edu.cn](mailto:xiayzh@shanghaitech.edu.cn).
