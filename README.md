## Data_Augmentation
A large amount of data is essential for training deep learning models. However, it is quite difficult to obtain a large amount of high-quality data to apply to the desired task. Therefore, in this repo, a series of methodologies that can compensate for the lack of data are presented separately by task.

<br>
<br>

## Translation Data Augmentaion

**Back Translation**

Machine translation is an NLP field that is already well serviced. In particular, large companies such as google and naver are showing very good performance. Obtain and organize the translation of the original text through Web Scraping.

<br>

**Generate Translation**

This Methodology generate new translation through a pre-trained model trained on a free usable dataset. The availability of the generated data is judged through the Discriminator, which determines the naturalness of the translation.

<br>

<br>
<br>

## Dialogue Data Augmentation
Conversation consists of utterances and responses to those utterances. Generating only utterances is not suitable as data for training. Therefore, data generation proceeds in two ways so that data can consist of pairs of utterances and answers. And it uses Discriminator to determine whether it is suitable as a data pair.

<br>

**Generate Utterance**

<br>

**Generate Reply**


<br>
<br>


## Summarization Data Augmentaion

The Document Summarization task requires data that consists of a long text and a summary of it. To this end, data augmentation of Summarization consists of a total of two steps. First, it finds available long text data and web scrapes it. Then, a summary of these texts is generated through a pre-trained model.

<br>

**Scraping Documents**

<br>

**Generate Summary**

<br>
<br>

## TBD
### OCR
In addition to data accessible on the web, various texts exist in our real life. And most of the text is written on paper. Among these, text that is suitable for the user's intention and is free from copyright can be extracted through OCR. Of course, a series of steps are additionally required to transfer the hard copy to the computer, but it is useful when data is desperately needed.


