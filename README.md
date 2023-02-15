# Neuroengineering-project
Vocal Tract Segmentation with U-net based framework from MRI images with overimposed Gaussian Noise

Project from the course Neuroengineering @ Politecnico di Milano 
- ####  Mattia Cazzolla  ([@MattiaCazzolla](https://github.com/MattiaCazzolla)) mattia.cazzolla@mail.polimi.it
- ####  Alix De Langlais (alixanne.delanglais@mail.polimi.it)
- ####  Paolo Marzolo (paolo.marzolo@mail.polimi.it)
- ####  Olmo Notarianni (michelangeloolmo.nogara@mail.polimi.it)
- ####  Sara Rescalli (sara.rescalli@mail.polimi.it)

# Dataset
The dataset provoded was generated using the frames of Dynamic Supine MRI (dsMRI) videos recorded for different patients under specific speech protocols.
All the images had additive Gaussian Noise overimposed.


<p align="center">
<img src="/imgs/lll.jpeg" alt="" width="700"/>
</p>

The dataset contains a total of 820 images from 4 patients (respectively 280, 240, 150, 150).

# Preprocessing
The preprocessing pipeling implemented tries to remove the Gaussian Noise with a Total variation denoising [technique](https://www.sciencedirect.com/science/article/abs/pii/016727899290242F?via%3Dihub) and to enhance the the high frequency component (borders).


<p align="center">
<img src="/imgs/preprocessing.jpeg" alt="" width="700"/>
</p>

# Model
The U-net architecture implemented consist of a variation from the IMU-NET described in this [paper](https://www.sciencedirect.com/science/article/abs/pii/S0169260721000201?via%3Dihub)

<p align="center">
<img src="/imgs/unet.jpg" alt="" width="700"/>
</p>

# Evaluation
The images from the first two patients were used as Training Set, the ones from the third patient as Validation Set and the remaining, from the last patient, as Test Set. 

<p align="center">
<img src="/imgs/training.gif" alt="" width="250"/>
</p>

The results on the test set are reported in the following table

<div align="center">

| Class | DICE (mean $\pm$ std) | 
|:-----------:|:----------------------:|
| Background | 0.991  $\pm$ 0.001 |
| Upper lip | 0.901  $\pm$ 0.033 |
| Hard palate | 0.819  $\pm$ 0.045 |
| Soft palate | 0.797  $\pm$ 0.059 |
| Tongue | 0.931  $\pm$ 0.012 |
| Lower lip | 0.898  $\pm$ 0.018 |
| Head | 0.968  $\pm$ 0.007 |

  
</div>

# Video
The project required us to produce a video of 3 minutes explaing our approach. 

https://user-images.githubusercontent.com/88252848/219075915-f0dc738e-6254-40e7-ad40-32489ccfecf9.mp4

# Licence
This project is licensed under the [MIT](LICENSE) License.
