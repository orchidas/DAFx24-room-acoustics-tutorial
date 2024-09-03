## Room acoustics modelling for immersive audio applications
### Orchisama Das, International Conference on Digital Audio Effects, 2024
#### University of Surrey, Guildford, UK

This repository contains materials related to the tutorial presented at DAFx 2024. The slides are [here](resources/DAFx24_tutorial_slides.pdf).

### Installation
- Clone and create virtual environment, `python3 -m venv .venv`
- Activate virtual environment with `source .venv/bin/activate`
- Installation can be done with `pip install .`
	* Requires `pip > 21.3`
	* Upgrade pip with `pip install --upgrade pip`
- Notebooks are in the `src` folder.
- Additional datasets are in the `resources` folder

### Other information
- This repo contains a Python implementation of the Feedback Delay Network. For a more comprehensive Matlab toolbox, see [FDNToolbox](https://github.com/SebastianJiroSchlecht/fdnToolbox)
- Some additional open-source repositories that are in Matlab, in case people want to play around with them. These include:
	* [FDNToolbox](https://github.com/SebastianJiroSchlecht/fdnToolbox)
	* [SDN implementation](https://github.com/enzodesena/sdn-matlab)
	* [SDM Toolbox](https://github.com/facebookresearch/BinauralSDM)
	* [HO-SRIR Toolbox](https://github.com/leomccormack/HO-SIRR)
- Image method implemented in [pyroomacoustics](https://github.com/LCAV/pyroomacoustics) 
- For binaural reproduction:
	* HRTF sets can be downloaded from the [SONICOM](https://imperialcollegelondon.app.box.com/s/c3ddjr3z4r8n1t4sus6h4pw0npxx8po7) dataset
	* SOFA file handling with [sofar](https://pypi.org/project/sofar/) library

