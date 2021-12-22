# 50.007 Project

### GROUP
Gargi Pandkar (1004680), Mihir Chhiber (1004359)

### INSTRUCTIONS TO RUN 
To run our code, execute like the following for each part:

```
python part1.py ES
python part1.py RU
```
The training data is from **inputData/ES/train**. By default, the test data is from **inputData/ES/dev.in**. Similarly for RU data.

To read other test data, specify the filepath like: 

```
python part1.py <FILEPATH>
```

For part 4, you can also execute the following:

```
python part4.py inputData/test.in
```
### EVALUATION

To get performance scores, execute the following:

```
python evalResult.py <GOLDFILE> <PREDICTIONFILE>
```

To evaluate all parts 1-4 on **dev.in**, you may also exceute:
```
./evalAll.sh
```

### DIRECTORY STRUCTURE
```
inputData
    - ES
        - dev.in
        - dev.out
        - test.in
        - train
   - RU
        - dev.in
        - dev.out
        - test.in
        - train
ES
    - dev.p1.out
    - dev.p2.out
    - dev.p3.out
    - dev.p4.out
    - test.p4.out
RU
    - dev.p1.out
    - dev.p2.out
    - dev.p3.out
    - dev.p4.out
    - test.p4.out
```