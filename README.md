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

## Train Raster Generator

The generator used here is based on the MultiContent-GAN (MCGAN) proposed by Azadi et al.

After preprocessing the dataset, train the generator with the following command:

```
$ python3 train.py
```

This trains the GAN for 300 epochs.

## Train SVG Autoencoder

The next component in the pipeline is an autoencoder augmented with the Differentiable Rasterizer for Vector Graphics proposed by Li et al. This component takes raster glyphs as inputs and generates vector glyphs.

Train the SVG autoencoder with the following command:

```
$ python3 train_ae.py
```

This trains the autoencoder for 200 epochs.

## Sample from pipeline

To generate samples from the pipeline combining the raster generator and the SVG autoencoder, run the following command:

```
$ python3 pipeline.py --gan-dir <GAN_MODEL_PATH> --ae-dir <AE_MODEL_PATH>
```

This interpolates between `LibreBaskerville-Regular` and `Asap-Bold` fonts. To change the start and end fonts, specify the `--start-font` and `--end-font` flags.

## References

[https://bair.berkeley.edu/blog/2018/03/13/mcgan/](https://bair.berkeley.edu/blog/2018/03/13/mcgan/)

[https://people.csail.mit.edu/tzumao/diffvg/](https://people.csail.mit.edu/tzumao/diffvg/)


