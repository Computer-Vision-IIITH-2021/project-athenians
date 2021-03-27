# project-athenians
# Style Transfer

Implementation of paper [Universal Style Transfer via Feature Transforms, Li, Y., Fang, C., Yang, J., Wang, Z., Lu, X., & Yang, M. H. (2017)](https://arxiv.org/pdf/1705.08086.pdf)

## Repository structure
```
├── docs
├── results
├── src
	├── images
	├── models
	├── main.py
	├──other python files
├── environment.yml
```
## Instructions to run the code

* clone repository
* install requirements
```bash
conda env create -f environment.yml
```
* download content and style images and name the content image same as corresponding style image.
* cd src
* run main.py
```bash
python main.py --contentPath <path to content images> --stylePath <path to style images> --outf <path for saving result> --cuda
```
example: 
```bash
python main.py --cuda
```

