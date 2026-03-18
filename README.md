# opticsbench_user_defined_kernels
Complementary framework for AROW ICCVW2023 to create user-defined aberations to measure robustness on image datasets or validation images based on Zernike optics descriptions - licensed under GNU General Public license v3.
<img width="1619" height="677" alt="corruptions" src="https://github.com/user-attachments/assets/72002156-8ff8-437f-9abd-0823aed85c33" />


#### OpticsBench kernels:
If you want to generate our **pre-defined corruptions from OpticsBench** please use our OpticsBench Github repository instead [classification_robustness/opticsbench](https://github.com/PatMue/classification_robustness/tree/main/opticsbench)

The OpticsBench corruptions can be generated from [classification_robustness/opticsbench](https://github.com/PatMue/classification_robustness/tree/main/opticsbench) by simply following [classification_robustness/dataset-generation](https://github.com/PatMue/classification_robustness/tree/main?tab=readme-ov-file#dataset-generation-generate-opticsbench-image-corruptions):
```
python generate_datasets.py --testdata_path <path_to_images_val_folder>  
```

#### User-defined kernels:
notes:
pip install . 

To create kernels from single Zernike Polynomials see the demo:
```
python opticsbenchgen/psf_generator.py
```

You can then use the generated psf_stack to generate your own corruption and test for it.


If you find this useful, please cite: 

```
  @article{mueller2023_opticsbench,
      author   = {Patrick Müller, Alexander Braun and Margret Keuper},
      title    = {Classification robustness to common optical aberrations},
      journal  = {Proceedings of the International Conference on Computer Vision Workshops (ICCVW)},
      year     = {2023}
   }
```
