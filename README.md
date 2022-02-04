## **E**valuation of human **P**ose recovery methods **O**n a **C**ollection of **H**eterogeneous **E**cosystems

```
epoche
│   README.md
│   config.ini.example    
│   
└───toolkit
│   │   ...
│
└───methods
    └───BLAZEPOSE
    │   ...
    │   
    └───GCN
    │   ...
    │
    └───HRNET
    │   ...
    │
    └───SBL
    │   ...
    │
    └───STGCN
    │   ...
    │
    └───VPOSE
    │   ...
    │
    └───YOLOX
        ...
    
```

### Dataset Preparation
	- Copy the configuration file config.ini.example to config.ini.
#### Human 3.6M
	- Use the scripts available at [https://github.com/anibali/h36m-fetch](https://github.com/anibali/h36m-fetch/) to download and preprocess Human3.6M data.
	- Edit config.ini and fill in your PATHTOH36MFOLDER

#### MPI-INF-3DHP
	- Download the original MPI-INF-3DHP dataset test set from [here](https://vcai.mpi-inf.mpg.de/3dhp-dataset/).
	- Edit config.ini and fill in your PATHTO3DHPFOLDER to point to the resulting mpi_inf_3dhp_test_set folder

### Methods' Conversion
	- download pretrained pool of models using `toolkit/handledownloads.py`
	- use `toolkit/handleconversions.py` to create ONNX and blob verions of the considered methods

### Evaluation


