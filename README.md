# opticsbench_user_defined_kernels
Complementary framework for AROW ICCVW2023 to create user-defined aberations to measure robustness on image datasets or validation images based on Zernike optics descriptions (c) Patrick Müller 2020-2023 - licensed under GNU General Public license v3.

If you want to generate our pre-defined corruptions from OpticsBench please use our OpticsBench Github repository (https://github.com/PatMue/classification_robustness/tree/main/opticsbench) instead by simply typing: 
> python benchmark.py --generate_datasets --database imagenet-1k_val  

notes:
pip install . 



If you find this useful, please cite: 

```
  @article{mueller2023_opticsbench,
      author   = {Patrick Müller, Alexander Braun and Margret Keuper},
      title    = {Classification robustness to common optical aberrations},
      journal  = {Proceedings of the International Conference on Computer Vision Workshops (ICCVW)},
      year     = {2023}
   }
```
