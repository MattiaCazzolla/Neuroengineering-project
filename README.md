# Neuroengineering-project
Vocal Tract Segmentation with U-net based framework from MRI images with overimposed Gaussian noise

Project from the course Neuroengineering @ Politecnico di Milano 
- ####  Mattia Cazzolla  ([@MattiaCazzolla](https://github.com/MattiaCazzolla)) mattia.cazzolla@mail.polimi.it
- ####  Alix de Langlais ([@Adelanglais](https://github.com/Adelanglais)) alixanne.delanglais@mail.polimi.it
- ####  Paolo Marzolo ([@pollomarzo](https://github.com/pollomarzo)) paolo.marzolo@mail.polimi.it
- ####  Olmo Notarianni (michelangeloolmo.nogara@mail.polimi.it)
- ####  Sara Rescalli (sara.rescalli@mail.polimi.it)
Final grade: 32/30

# Dataset
The dataset provided was generated using the frames of Dynamic Supine MRI (dsMRI) videos recorded for different patients under specific speech protocols. <br>
All the images had additive Gaussian noise overimposed.


<p align="center">
<img src="/imgs/lll.jpeg" alt="" width="700"/>
</p>

The dataset contained a total of 820 images from 4 patients (respectively 280, 240, 150, 150).

# Preprocessing
The preprocessing pipeline implemented aims at:
- Removing the Gaussian noise with a Total Variation Denoising technique ([link](https://www.sciencedirect.com/science/article/abs/pii/016727899290242F?via%3Dihub))
- Enhancing the high frequency component


<p align="center">
<img src="/imgs/preprocessing.jpeg" alt="" width="700"/>
</p>

# Model
The U-net architecture implemented consists of a variation from the IMU-NET described in this [paper](https://www.sciencedirect.com/science/article/abs/pii/S0169260721000201?via%3Dihub)

<p align="center">
<img src="/imgs/unet.jpg" alt="" width="600"/>
</p>

# Evaluation
The images were split into different datasets as follows:
- Patient 1 and 2 $\rightarrow$ Training Set
- Patient 3 $\rightarrow$ Validation Set
- Patient 4 $\rightarrow$ Test Set

The results on the test set are reported in the following table

<div align="center">

| Class | DICE (mean $\pm$ std) | 
|:-----------:|:----------------------:|
| Background | 0.991  $\pm$ 0.001 |
| Upper Lip | 0.901  $\pm$ 0.033 |
| Lower Lip | 0.898  $\pm$ 0.018 |
| Hard Palate | 0.819  $\pm$ 0.045 |
| Soft Palate | 0.797  $\pm$ 0.059 |
| Tongue | 0.931  $\pm$ 0.012 |
| Head | 0.968  $\pm$ 0.007 |

</div>

The progress in learning can be observed by the segmentation at each epoch of the training

<p align="center">
<img src="/imgs/training.gif" alt="" width="250"/>
</p>

# Video
The project required us to produce a 3 minutes video explaining our approach. 

https://user-images.githubusercontent.com/88252848/219075915-f0dc738e-6254-40e7-ad40-32489ccfecf9.mp4

<br> 

Voiced by: [@Adelanglais](https://github.com/Adelanglais), [@pollomarzo](https://github.com/pollomarzo) <br>
Animated by: [@Adelanglais](https://github.com/Adelanglais), [@MattiaCazzolla](https://github.com/MattiaCazzolla)
# Licence
This project is licensed under the [MIT](LICENSE) License.
