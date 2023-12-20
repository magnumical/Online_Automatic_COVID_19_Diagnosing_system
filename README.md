# WebBased COVID_19 Diagnosing System (SaaS)
Machine Learning approach to diagnose COVID-19 via X-Ray images (image classification)

<img src="https://github.com/magnumical/Online_Automatic_COVID_19_Diagnosing_system/blob/master/logom.png" data-canonical-src="https://imreza.ir" width="300" height="100" />

## [Preprint is available here](https://www.researchsquare.com/article/rs-64472/v1)


This repository contains an extraordinary project! Machine learning is helping medical doctors to obtain maximum accuracy! 

### Please star, fork and cite if you use! 

### How to use?
1. First you need to train your model. use folder _Train_. Run _maincode_ to build your model.<br>
<b>NOTICE</b>: for Inception V3, you need to change input image sizes. <br>
<b>Dataset </b> folder should have two subfolders. _NORMAL_ and _COVID_. <br>
<b>Models</b> will be savel in a folder with this name. 
2. Second step is related to Google Cloud Platform. <br>
_main.py_ is handler function which calls our ML model on server and use its power to evaluate images! <br>
<b>NOTICE</b> that you should put your prefered _.h5_ model in this folder and modify the code.
3. Now it's time to run our site! you can use my theme! <br>
_script.js_ contains all JS we need to transfer images and results!
<br> <b>Notice</b> that the models in this repository are trained by images collected by my own!

### Results for VGG16:
<img src="https://github.com/magnumical/Online_Automatic_COVID_19_Diagnosing_system/blob/master/img/confusinVGG16.png" data-canonical-src="https://imreza.ir" width="350" height="250" />
<img src="https://github.com/magnumical/Online_Automatic_COVID_19_Diagnosing_system/blob/master/img/historyvgg16.png" data-canonical-src="https://imreza.ir" width="450" height="250" />

### Cite like this!
`Reza Amini Gougeh. An Online Automatic Corona Diagnose System Based on Chest X-ray Images, 22 September 2020, PREPRINT (Version 1) available at Research Square [+https://doi.org/10.21203/rs.3.rs-64472/v1+]`


### Some valuable refrences:
| Name | Description |
| --- | --- |
| [Michaël Defferrard](https://github.com/mantasbandonis/covid19-classification)| Created the first versions of ONLINE Diagnosing system! |
| [Adrian Rosebrock](https://www.pyimagesearch.com/2020/03/16/detecting-covid-19-in-x-ray-images-with-keras-tensorflow-and-deep-learning/) | Awsome in image processing!|
| [Cohen et. al](https://github.com/ieee8023/covid-chestxray-dataset) | Collected the image dataset|

