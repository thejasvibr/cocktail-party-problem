# BioCPPNet: Automatic Bioacoustic Source Separation with Deep Neural Networks

Fork of Bermant 2021 (Sci. Rep.) - my own notes and points which need clarification during installation and recreating the steps. 

### System description: Win 10 Dell Laptop,intel i7 CPU, 16GB ram,no graphics card

## Setup
1. Clone the repo
```command
git clone https://github.com/earthspecies/cocktail-party-problem.git
```

*works on Win10*

2. Install the requirements
```command
pip install -r requirements.txt
```

*works on Win10* 

## Pipeline

1. Download the bioacoustic datasets into a directory `BioacousticData/`

a) *What is a 'dataset' - the raw unzipped folder? The default Git repo has a 'Data' folder, or are we supposed to make a separate BioacousticData folder?*
b) Managed to download the Macaque dataset from https://datadryad.org/stash/dataset/doi:10.5061/dryad.7f4p9 - and the individual coo data is a couple of folders into the raw unzipped folder.

2. Generate the config.json file containing the configurations for constructing datasets, building and training models, and evaluating model performance. *config file seems to be generated without error*


   ```command
   python ConfigGenerator.py --animal Animal --file config.json
   ```

   where `Animal` is the particular animal of interest (in our case, `Macaque`,  `Dolphin`, or `Bat`).


3. Generate the datasets for training and evaluating the classifier and separator models

   ```command
   python DataGenerator.py --animal Animal --data_directory Data --config config.json --os Ubuntu --objective Classification --regime Closed
   ```
 ### Error 1
*here the training/eval datasets are created using the 'Data' folder as input. This is where the first error message pops in*

```
(sourcesep) PS C:\Users\..\cocktail-party-problem> python DataGenerator.py --animal Macaque --data_directory Data --config config.json --os Ubuntu --objective Classification --regime Closed
Traceback (most recent call last):
  File "C:\Users\..\cocktail-party-problem\DataHelpers.py", line 5, in <module>
    from fastai.vision.all import untar_data, get_files
  File "C:\Users\theja\anaconda3\envs\sourcesep\lib\site-packages\fastai\vision\all.py", line 1, in <module>
    from . import models
  File "C:\Users\theja\anaconda3\envs\sourcesep\lib\site-packages\fastai\vision\models\__init__.py", line 1, in <module>
    from . import xresnet
  File "C:\Users\theja\anaconda3\envs\sourcesep\lib\site-packages\fastai\vision\models\xresnet.py", line 13, in <module>
    from torchvision.models.utils import load_state_dict_from_url
ModuleNotFoundError: No module named 'torchvision.models.utils'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "DataGenerator.py", line 14, in <module>
    from DataHelpers import *
  File "C:\Users\..\cocktail-party-problem\DataHelpers.py", line 7, in <module>
    from fastai2.vision.all import untar_data, get_files
ModuleNotFoundError: No module named 'fastai2'
```
*This error is apparently because torchvision changed some things and fastai hadn't kept up with these changes (see [Issue 3507](https://github.com/fastai/fastai/issues/3507)*

*Potential solutions:*
*1)uninstall default pip installed torch and torchvision, conda install the cpu version. (```pip uninstall torchvision``` and then ``` conda install pytorch torchvision torchaudio cpuonly -c pytorch```)*
*2) Also uninstall the pip installed fastai and conda install it afresh. (first ```pip uninstall fastai``` and then ```conda install -c fastchan fastai anaconda```)*

### Error 2
*Now, after the above steps, we sort out the ```torchvision.models.utils``` error, and proceed to get Error 2 - which is because the URL in the Macaque part of the ```DataHelpers.py``` is broken.*

```
Traceback (most recent call last):
  File "DataGenerator.py", line 198, in <module>
    X, Y = generate_labeled_waveforms(animal,
  File "DataGenerator.py", line 35, in generate_labeled_waveforms
    X, Y = loader.run(balance=kwargs['balance'])
  File "C:\Users\..\cocktail-party-problem\DataHelpers.py", line 91, in run
    data_df = self.fixed_dataframe()
  File "C:\Users\..\cocktail-party-problem\DataHelpers.py", line 52, in fixed_dataframe
    dataframe = self.construct_dataframe()
  File "C:\Users\..\cocktail-party-problem\DataHelpers.py", line 24, in construct_dataframe
    path = untar_data(self.url)
  File "C:\Users\..\anaconda3\envs\sourcesep\lib\site-packages\fastai\data\external.py", line 124, in untar_data
    return d.get(url, force=force_download, extract_key=c_key)
  File "C:\Users\..\anaconda3\envs\sourcesep\lib\site-packages\fastdownload\core.py", line 121, in get
    self.download(url, force=force)
  File "C:\Users\..\anaconda3\envs\sourcesep\lib\site-packages\fastdownload\core.py", line 96, in download
    return download_and_check(url, urldest(url, self.arch_path()), self.module, force)
  File "C:\Users\..\anaconda3\envs\sourcesep\lib\site-packages\fastdownload\core.py", line 65, in download_and_check
    res = download_url(url, fpath)
  File "C:\Users\..\anaconda3\envs\sourcesep\lib\site-packages\fastdownload\core.py", line 23, in download_url
    return urlsave(url, dest, reporthook=progress if show_progress else None, timeout=timeout)
  File "C:\Users\..\anaconda3\envs\sourcesep\lib\site-packages\fastcore\net.py", line 178, in urlsave
    nm,msg = urlretrieve(url, dest, reporthook, timeout=timeout)
  File "C:\Users\..\anaconda3\envs\sourcesep\lib\site-packages\fastcore\net.py", line 143, in urlretrieve
    with contextlib.closing(urlopen(url, data, timeout=timeout)) as fp:
  File "C:\Users\..\anaconda3\envs\sourcesep\lib\site-packages\fastcore\net.py", line 105, in urlopen
    return _opener.open(urlwrap(url, data=data, headers=headers), timeout=timeout)
  File "C:\Users\..\anaconda3\envs\sourcesep\lib\urllib\request.py", line 531, in open
    response = meth(req, response)
  File "C:\Users\..\anaconda3\envs\sourcesep\lib\urllib\request.py", line 640, in http_response
    response = self.parent.error(
  File "C:\Users\..\anaconda3\envs\sourcesep\lib\urllib\request.py", line 569, in error
    return self._call_chain(*args)
  File "C:\Users\..\anaconda3\envs\sourcesep\lib\urllib\request.py", line 502, in _call_chain
    result = func(*args)
  File "C:\Users\..\anaconda3\envs\sourcesep\lib\urllib\request.py", line 649, in http_error_default
    raise HTTPError(req.full_url, code, msg, hdrs, fp)
urllib.error.HTTPError: HTTP Error 404: Not Found
```

```
python DataGenerator.py --animal Animal --data_directory Data --config config.json --os Ubuntu --objective Separation --regime Closed
```


We can also consider the open speaker regime in which the evaluation subset contains calls generated by individuals not included in the training distribution
	
```
python DataGenerator.py --animal Animal --data_directory Data --config config.json --os Ubuntu --objective Separation --regime Open
```
4. Train the classifier model, which is used to evaluate the performance of the separator model on a downstream task

   ```command
	python Classifier.py --animal Animal --data Data --config config.json
   ```

5. Train the separator model including a classifier to evaluate performance as well as the classifier's testing accuracy to account for the probabilistic nature of classifying biacoustic signals and the stochasticity of the classifier model

   ```command
	python Separator.py --animal Animal --data Data --config config.json --classifier_name classifier_name --classifier_peak_acc classifier_peak_acc --regime Closed
   ```

6. Evaluate the performance in the appropriate regime, for example the `Open` regime

   ```command
	python Evaluate.py --animal Animal --data Data --config config.json --separator_name separator_name --classifier_name classifier_name --classifier_peak_acc classifier_peak_acc --regime Open
   ```

## Acknowledgements
We thank Laela Sayigh, Frants Jensen, Michelle Fournet, Andrés Babino, and James Crutchfield for reviewing the manuscript and Steve Vassallo, Stefan Thomas, Evan Sharp, Munjal Shah, Shiva Ranjaram, Meghan Railey, Alex Payne, Chris Larsen, Mike Kreiger, Nicole Brodeur, and Scott Belsky for their invaluable support.
