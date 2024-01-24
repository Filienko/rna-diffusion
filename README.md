# Gene Expression Generation with Diffusion Models

## Requirements

Install the required python librairies:

`pip install -r requirements.txt`

## Datasets
The first dataset is The Cancer Genome Atlas (TCGA): 
- [About TCGA](https://www.cancer.gov/about-nci/organization/ccg/research/structural-genomics/tcga)
- [R package](https://bioconductor.org/packages/release/bioc/html/RTCGA.html) to retrieve data

To preprocess the TCGA data, go to the `data` folder.

The second dataset is GTEx... TODO

## Models
TODO

Scripts of the different models can be found at the path the `./src/generation`.

## Metrics
To assess our generated expression data quality, we evaluated the data in a supervised and unsupervised manner.
Scripts of these metrics can be found in the `metrics` folder.

### Supervised performance indicators
- Reverse validation: the performance (accuracy) of a classifier trained only on generated data

### Unsupervised performance indicators
- Correlation score [(Vinas et al., 2022)](https://academic.oup.com/bioinformatics/article/38/3/730/6104825)
- Precision and Recall [(Kynkäänniemi et al., 2019)](https://arxiv.org/pdf/1904.06991.pdf)
- Frechet Distance (FD) [(Heusel et al., 2018)](https://arxiv.org/pdf/1706.08500.pdf)
- Adversarial accuracy (AA) [(Yale et al., 2020)](https://www.sciencedirect.com/science/article/abs/pii/S0925231220305117)

## Results

Here is a **UMAP representation** of real (blue) and fake (red) TCGA data generated by our best DDIM. Colors highlight different tissue types.
<p align="center">
  <img src="figures/umap_tcga_ddim.pdf" width="750" />
</p>