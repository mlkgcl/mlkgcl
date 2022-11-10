## ML-KGCL

The Recbole library has implemented many well-known recommendation models. It has good support for the developer to develop their own recommendation models.  We implemented our ML-KGCL based on the Recbole library. More details can be obtained from the [README_RecBole.md](.\README_RecBole.md).

### Running Environment：

RecBole requires Python version 3.7 or later. The runtime environment in which Recbole should be installed before running:

##### Install from conda：

~~~bash
conda install -c aibox recbole
~~~

##### Install from pip：

```bash
pip install recbole
```

### Running Example：

You can see the recommended results by running the trained model in the [case_study.py](./case_study.py) file. Because the model file is too large, please download the saved models and unzip them in "./saved" folder from [link](https://figshare.com/s/899e39b2bee3d4a042c4) before running. And switch the model by changing the "model_file" configuration.

You can see how groups of items with different exposures contribute to Recall in the [long-tail-validation.py](./case_study/long-tail-validation.py) file.

### Dataset

Because the datasets are too large, so we do not include them in the source code. All datasets are available on [RecBole](https://github.com/RUCAIBox/RecSysDatasets). Please download datasets before running. A more detailed description can be found in [README_RecBole.md](.\README_RecBole.md).

### Model Training：

##### Quick Start：

~~~bash
nohup python run_mlkgcl.py > /dev/null 2>&1 &
~~~

##### Specific Implementation
The specific implementation of the model can be found in the [mlkgcl.py](./recbole/model/knowledge_aware_recommender/mlkgcl.py) file.

 ##### Configuration File：

You can modify the configuration file [mlkgcl.yaml](./mlkgcl.yaml) and [run_mlkgcl.py](./run_mlkgcl.py) to set different hyperparameters and switch datasets.

### RecBole Cite：
> @article{recbole,
>     title={RecBole: Towards a Unified, Comprehensive and Efficient Framework for Recommendation Algorithms},
>     author={Wayne Xin Zhao and Shanlei Mu and Yupeng Hou and Zihan Lin and Kaiyuan Li and Yushuo Chen and Yujie Lu and Hui Wang and Changxin Tian and Xingyu Pan and Yingqian Min and Zhichao Feng and Xinyan Fan and Xu Chen and Pengfei Wang and Wendi Ji and Yaliang Li and Xiaoling Wang and Ji-Rong Wen},
>     year={2020},
>     journal={arXiv preprint arXiv:2011.01731}
> }
