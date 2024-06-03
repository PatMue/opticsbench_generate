# opticsbench_user_defined_kernels
Complementary framework for AROW ICCVW2023 to create user-defined aberations to measure robustness on image datasets or validation images based on Zernike optics descriptions (c) Patrick Müller 2020-2024 - licensed under GNU General Public license v3.


#### OpticsBench kernels:
If you want to generate our **pre-defined corruptions from OpticsBench** please use our OpticsBench Github repository instead [classification_robustness/opticsbench](https://github.com/PatMue/classification_robustness/tree/main/opticsbench)

The OpticsBench corruptions can be generated from [classification_robustness/opticsbench](https://github.com/PatMue/classification_robustness/tree/main/opticsbench) by simply following [classification_robustness/dataset-generation](https://github.com/PatMue/classification_robustness/tree/main?tab=readme-ov-file#dataset-generation-generate-opticsbench-image-corruptions):
```
python benchmark.py --generate_datasets --database imagenet-1k_val  
```

#### User-defined kernels:
notes:
pip install . 

To create kernels from single Zernike Polynomials see the demo:
```
python opticsbench_generate/psf_simple.py
```



If you find this useful, please cite: 

```
  @article{mueller2023_opticsbench,
      author   = {Patrick Müller, Alexander Braun and Margret Keuper},
      title    = {Classification robustness to common optical aberrations},
      journal  = {Proceedings of the International Conference on Computer Vision Workshops (ICCVW)},
      year     = {2023}
   }
```
