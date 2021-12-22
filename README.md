# Final Project level3-cv-05
---
## Getting Started
### Installation
Create conda enviroment
```bash
conda create -n mathnote python=3.7.11 -y
conda activate mathnote
```

Install pytorch&mmcv
```bash
conda install pytorch=1.6.0 cudatoolkit=10.1 torchvision -c pytorch -y
pip install mmcv-full==1.3.17 -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.6.0/index.html
```

Install requirements
```bash
pip install -r requirements.txt
```

---
### Authors
* 강수빈 ([@suuuuuuuubin](https://github.com/suuuuuuuubin)) : GAN, Back-end
* 김인재 ([@K-nowing](https://github.com/K-nowing)) : Project Manager, Object Detection, Augmentation
* 원상혁 ([@wonsgong](https://github.com/wonsgong)) : Segmentation, Data Pre-processing
* 이경민 ([@lkm2835](https://github.com/lkm2835)) : Segmentation, Generating Synthetic Data, Code Review
* 최민서 ([@minseo0214](https://github.com/minseo0214)) : Front-end, Data Crawling
