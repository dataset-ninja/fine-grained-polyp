Dataset **Fine Grained Polyp** can be downloaded in [Supervisely format](https://developer.supervisely.com/api-references/supervisely-annotation-json-format):

 [Download](https://www.dropbox.com/scl/fi/mt4wwpd1nqt8ishbt5t3p/fine-grained-polyp-DatasetNinja.tar?rlkey=c30h6k1oijg1xjpcu5r0glhr9&dl=1)

As an alternative, it can be downloaded with *dataset-tools* package:
``` bash
pip install --upgrade dataset-tools
```

... using following python code:
``` python
import dataset_tools as dtools

dtools.download(dataset='Fine Grained Polyp', dst_dir='~/dataset-ninja/')
```
Make sure not to overlook the [python code example](https://developer.supervisely.com/getting-started/python-sdk-tutorials/iterate-over-a-local-project) available on the Supervisely Developer Portal. It will give you a clear idea of how to effortlessly work with the downloaded dataset.

The data in original format can be [downloaded here](https://drive.google.com/file/d/1rmMLHohni3Vq_fZ-Ddt42vj6C6SSlkvW).