# jpeg-dcgan
JPEG圧縮過程を利用したDCGAN

|英語|日本語|
|:-:|:-:|
|Add|追加|
|Fix|修正|
|Improve|改善|
|Update|更新|
|Remove|削除|
|Rename|改名|
|Move|移動|
|Modify|変更|

## リポジトリ
### [uber-research/jpeg2dct](https://github.com/uber-research/jpeg2dct)

### [TonyMooori/dct_2dim.py](https://gist.github.com/TonyMooori/661a2da7cbb389f0a99c)

## ライブラリ
### [kornia](https://kornia.github.io/)
GPU上で動作するData Augmentationライブラリ
[Documentation](https://kornia.readthedocs.io/en/latest/index.html)

## 参考文献
### [Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677)
学習率をバッチサイズに比例させるLinear Scalingを行う。

`Goyal, P., Dollár, P., Girshick, R., Noordhuis, P., Wesolowski, L., Kyrola, A., Tulloch, A., Jia, Y., & He, K. (2017). Accurate, large minibatch sgd: Training imagenet in 1 hour. arXiv preprint arXiv:1706.02677.`

```
@misc{goyal2018accurate,
      title={Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour}, 
      author={Priya Goyal and Piotr Dollár and Ross Girshick and Pieter Noordhuis and Lukasz Wesolowski and Aapo Kyrola and Andrew Tulloch and Yangqing Jia and Kaiming He},
      year={2018},
      eprint={1706.02677},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

### [cGANs with Projection Discriminator](https://arxiv.org/abs/1802.05637)
Discriminatorの最終層の特徴とクラスを埋め込んだベクトルの内積を取る。

`Miyato, T., & Koyama, M. (2018). cGANs with projection discriminator. arXiv preprint arXiv:1802.05637.`

```
@misc{miyato2018cgans,
      title={cGANs with Projection Discriminator}, 
      author={Takeru Miyato and Masanori Koyama},
      year={2018},
      eprint={1802.05637},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

### [GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium](https://arxiv.org/abs/1706.08500)
GeneratorとDiscriminatorでそれぞれ異なる学習率を適用する。

`Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., & Hochreiter, S. (2017). Gans trained by a two time-scale update rule converge to a local nash equilibrium. In Advances in neural information processing systems (pp. 6626-6637).`

```
@misc{heusel2018gans,
      title={GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium}, 
      author={Martin Heusel and Hubert Ramsauer and Thomas Unterthiner and Bernhard Nessler and Sepp Hochreiter},
      year={2018},
      eprint={1706.08500},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

### [Self-Attention Generative Adversarial Networks](https://arxiv.org/abs/1805.08318)
GeneratorとDiscriminatorの途中にSelf-Attentionを適用する。

`Zhang, H., Goodfellow, I., Metaxas, D., & Odena, A. (2019, May). Self-attention generative adversarial networks. In International Conference on Machine Learning (pp. 7354-7363). PMLR.`

```
@misc{zhang2019selfattention,
      title={Self-Attention Generative Adversarial Networks}, 
      author={Han Zhang and Ian Goodfellow and Dimitris Metaxas and Augustus Odena},
      year={2019},
      eprint={1805.08318},
      archivePrefix={arXiv},
      primaryClass={stat.ML}
}
```

### 参考記事とか
1. [soumith/ganhacks - GitHub](https://github.com/soumith/ganhacks)
2. [Tips for Training Stable Generative Adversarial Networks](https://machinelearningmastery.com/how-to-train-stable-generative-adversarial-networks/)
3. [Keep Calm and train a GAN. Pitfalls and Tips on training Generative Adversarial Networks](https://medium.com/@utk.is.here/keep-calm-and-train-a-gan-pitfalls-and-tips-on-training-generative-adversarial-networks-edd529764aa9)
4. [GAN — Ways to improve GAN performance](https://towardsdatascience.com/gan-ways-to-improve-gan-performance-acf37f9f59b)
5. [GAN(Generative Adversarial Networks)を学習させる際の14のテクニック](https://qiita.com/underfitting/items/a0cbb035568dea33b2d7)
6. [DCGAN(Deep Convolutional GAN)｜DeepLearning論文の原文を読む #12](https://lib-arts.hatenablog.com/entry/paper12_DCGAN)
7. [DCGAN](https://medium.com/@liyin2015/dcgan-79af14a1c247)
8. [GANで学習がうまくいかないときに見るべき資料](https://gangango.com/2018/11/16/post-322/)
9. [個人的GANのTipsまとめ](https://qiita.com/pacifinapacific/items/6811b711eee1a5ebbb03)
10. [PyTorchでDCGANやってみた](https://blog.shikoan.com/pytorch-dcgan/)
11. [CIFAR10を混ぜたままChainerのDCGANに突っ込んだら名状しがたい何かが生成された話](https://ensekitt.hatenablog.com/entry/2017/11/07/123000)

#### 総合
1. [GANの発展の歴史を振り返る！GANの包括的なサーベイ論文の紹介(アルゴリズム編)](https://ai-scholar.tech/articles/treatise/gansurvey-ai-371)
2. [GANの発展の歴史を振り返る！GANの包括的なサーベイ論文の紹介(応用編)](https://ai-scholar.tech/articles/treatise/gan-survey-ai-375)
