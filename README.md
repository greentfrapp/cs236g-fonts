# cs236g-fonts
Final project for CS236G Winter 2021

## Dataset

The dataset used in this project will be based on the Google Fonts dataset.

First, follow the instructions in the Google Fonts [repo](https://github.com/google/fonts) and download the fonts (as TTF files) from [https://github.com/google/fonts/archive/master.zip](https://github.com/google/fonts/archive/master.zip).

The fonts will be downloaded as TTF files. To train our models, we will require SVG and JPG files. More specifically, the pipeline follows the following formats:

1. TTF
2. UFO
3. SVG
4. JPG

You can run the following command to convert the files:

```
$ python3 preprocess_fonts.py
```
