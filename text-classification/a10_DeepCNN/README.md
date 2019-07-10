## Deep Pyramid Convolutional Neural Networks for Text Categorization

> This is the implementation of [DPCNN](http://www.aclweb.org/anthology/P17-1052) in tensorflow.

![DPCNN](/img/dpcnn.png)

The key operation of this paper is 
- fixed feature map:250
- 2 stride downsampling which can compress effective information of long distance.
![pyramid](/img/pyramid.png)

The format of data :
- .csv file
- it has two columns,one column is content,the other column is label.
- you can modify value of parameter **--file_name** to use your train,val,test dataset. 
- example in data/
- you should put your dataset in data/

```
python run.py --train --model_name DPCNN --write_vocab True --experiment_name test
```
When you run the code first time,you should set write_vocab True to write vocab for the data.


