# Assignment Internet Security with AI


We have attempted to use the material given to us in class. The material can be be found [here](https://archive.ics.uci.edu/ml/datasets/Kitsune+Network+Attack+Dataset#).

We chose to work with the SSL Renegotiation sample. With this sample, we received 3 files from the download, a dataset of 6.2GB containing 115 features and 2.2 million rows.

We also received a file with labels of 236 mb. containing each rows label, either its benign or malicious, denoted as either a 0 for benign or 1 for malicious.

the third file was a pcap aka a capture from [wireshark](https://www.wireshark.org/) containing meta information about each packet.

At the moment we don't know how to fit the dataset to make it work with an algorithm. 

We have tried to import it as is, and use it along with the [logisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) algorithm, from scikit learn. It processed the dataset and fit the data without complaint:

```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('SSL Renegotiation_dataset-002.csv') 
labels = pd.read_csv('SSL Renegotiation_labels.csv')
y = labels.iloc[:,1:2]



X_train, X_test, y_train, y_test = train_test_split(dataset, y, test_size = 0.2, random_state = 0)

regressor = LogisticRegression()

regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

```

However due to its large dataset size, we were unable to use score method, but when comparing the predicted labels with the expected labels, we didn't find a single hit, bear in mind we didn't look all 2.2 million entries through.

We expect that we are getting such imprecise predictions due to the way the dataset is written, to give you an idea:

![](/images/dataset-snippet.png)

Upon reading the source material a little deeper, it was clear that there were originally 23 features, which could be extracted with the help of the included featureExtractor.py, the features we were reading were timestamped windows of approximately: 100ms, 500ms, 1.5sec, 10 sec and 1 min into the past this resulting in the 115 features we were seeing in the dataset.

By importing the code from the original projects [github](https://github.com/ymirsky/Kitsune-py) page, we should be able to extract the features we would need to run our classifier.

We have included the modifications we made to the sourcecode, which we got from Kitsune.

In brief, we modified the featureExtractor, and fed it with the given path to the file we wished to extract data from.
Furthermore we made small altercations to the code, as it didn't run nor complied out of the box.
If you wish to replicate this, you have to make sure featureExtractor.py is given the correct path.

We have therefore not been able to run our classifier and we don't have any data to show.

We have tried to find other datasets online, but its hard to find any datasets that make sense to us with our current knowledge.

***
## LogisticRegression

We will instead make a case for why LogisticRegression would be the right algorithm to use.


According to the labels there are only 2 outcomes for each row; Besign and malicious - 0 and 1 in other words we are looking at a binary value.
What we are trying to make the classifier do, is to make it recognize weather a packet is malicious or benign, we should according to documentation have had 23 features to determine this.

Logistic Regression works by measuring several features, and then determine, by the defining a classifier, which class the data belongs to, either 1 or 0. Or in our case; benign or malicious. The curve of the logistic function looks like an 'S', by measuring data on the x-axis, the data can be plotted on the y axis, which as mentioned only has two possible outcomes, 1 or 0.


We aren't pleased that we weren't able to have any real data to show for our hard work in trying to make it work, but we hope to start a dialogue in class regarding how this could be avoided and hopefully see other students succeed where we failed and learn from them.

@Authors

Mikkel, Nikolai, Nikolaj
