#Unsupervised DeepView for Uncertainty Visualization

This is an implementation of a novel framework allowing for unsupervised uncertainty visualization depicting a global decision boundary. Inspired by [Deepview](https://github.com/LucaHermes/DeepView), it no longer requires labels for reliable visualization of misclassifications or adversarials. It solves the lack of global explainability methodologies for unsupervised learning areas that depict uncertain areas in data sets, whereas [Deepview](https://github.com/LucaHermes/DeepView) explicitly required labeled data to spot any uncertainties in the visualization method. The 'deepview' repository is just an evaluation of our unsupervised method using the supervised implementation by the authors of the Deepview paper. 
## Key Features
* model-agnostic, only requires prediction uncertanties and the raw data
* global visualization method

## Requirements
All required python libraries are stored in the ```requirements.txt``` file.
Tested on Python 3.6.
```bash
pip install -r requirements.txt
```
### Output of our visualization method
See DeepView_unsup.png
