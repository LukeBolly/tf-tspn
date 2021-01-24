A Tensorflow 2.3 implementation of the paper:

Conditional Set Generation with Transformers (AR Kosiorek, H Kim, DJ Rezende)
[Arxiv Link](https://arxiv.org/abs/2006.16841)

<img src="animation.gif" alt="Image not found" width="400"/>

To train the TSPN Autoencoder, run `mnist_train.py`

To load weights from a saved step, use the `-s` argument. You can pass a specific saved step, or -1 to load the latest.

To train the Size Predictor MLP after training TSPN, use the `-p` flag in combination with `-s`

A Tensorflow port of [FSPool](https://github.com/Cyanogenoid/fspool) is also available in the models [folder](https://github.com/LukeBolly/tf-tspn/blob/master/models/fspool.py).

Requires:
* Content root of the project must be set as the top level folder
* Python 3.7
* tensorflow 2.3
* tensorflow-datasets 3.2.1
* matplotlib 3.3.1
* imagemagick if exporting using `mnist_gif.py`
