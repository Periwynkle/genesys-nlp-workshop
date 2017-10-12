# How to Analyze Large Volumes of Text

This day-long methods workshop will teach basic natural language processing (NLP) techniques for large volumes of online text (from websites, blogs, forums, social media, etc.) to researchers in the social sciences and humanities. Online text is qualitatively different from offline text, and many traditional corpus methods do not directly translate to online material. Moreover, the concept of ‘big data’ is closely connected with the internet, so issues of scale make quantitative approaches more necessary. How can we find patterns in online text? What are the opportunities, and what are the main challenges and constraints? These patterns can relate to sentiment, topic, or simple frequencies, all of which the workshop will cover (using the Python programming language). The workshop will also introduce participants to CLARIN resources and tools that are relevant to a) the analysis of large volumes of digital discourse and b) East Asian languages.


## Preparation

**Note**: Participants should bring a laptop with all of the software downloaded in order to fully participate in the workshops. Installation instructions are detailed below.

### Python and Jupyter Notebook Installation Instructions
To attend and fully benefit from the training day, we expect participants to have basic knowledge of the programming language Python and its ecosystem. To accommodate this requirement, we recommend that participants read the Introduction to Python Programming for the Humanities by Karsdorp et al. (2017), which can be downloaded from https://github.com/fbkarsdorp/python-intro.

The workshop requires an installation of Python 3.6 or higher versions on Linux, macOS, or Microsoft Windows. We highly recommend that participants install Python through the Anaconda distribution (https://www.continuum.io/), which bundles together a range of open-source Python packages and libraries used in data analysis and scientific computing—it includes Jupyter Notebook, the web application that we will be using in the workshops to run our code. Alternatively, you can install Python using a binary installer from the Python Software Foundation (https://www.python.org/), or through an operating system’s package manager (e.g., apt on Debian Linux and homebrew on macOS).

As noted above, it is essential that participants download not only Python 3.6 or higher, but also Jupyter Notebook. A very useful Jupyter Notebook Quick Start Guide can be found here: https://jupyter-notebook-beginner-guide.readthedocs.io/en/latest/install.html. As noted above, Jupyter Notebook is included in the Anaconda distribution, so does not have to be downloaded separately. Participants who are unfamiliar with Jupyter should watch this 30-minute YouTube tutorial prior to the workshop (not strictly required, but highly recommended): https://www.youtube.com/watch?v=HW29067qVWk&t=4s

### SentiStrength Overview and Installation Instructions

SentiStrength is a commercial sentiment analysis Java program that is available for free for academic research. It reads social web texts and then assigns them two scores: one each for positive and negative sentiment strength.

    -1 (no negative sentiment) to -5 (contains extremely negative sentiment)
    1 (no positive sentiment) to 5 (contains extremely positive sentiment)

SentiStrength has a dictionary of over 3,000 words that are pre-classified for sentiment strength (e.g., love=+3; hate=-4) and applies these scores to words when found in a text. It uses rules to cope with sentiment expressed or modified in other ways, such as negation (e.g., not happy), boosting (very nice!), emoticons (e.g., :)), and sentiment spelling (e.g., Yaaaaaay!!!).

SentiStrength can be tried out online at: http://sentistrength.wlv.ac.uk. For Java users, there is a bit of extra information for the Java version on the SentiStrength website. There is no JavaDoc but there is a brief manual online.

The **SentiStrength software** and data files for the workshop are in the following folder: http://sentistrength.wlv.ac.uk/jkpop/
1.	Make a new folder on your computer to save the software and data.
2.	With a web browser, download the SentiStrength program SentiStrength.jar to the new folder on your computer.
3.	Download the SentiStrength linguistic data files SentStrength_Data.zip to the new folder on your computer and unzip it. Make sure that the linguistic files are inside a separate folder.
4.	Download the YouTube data files TWICE-BTS-EXO-BLACKPINK_english comments, one file per video.zip and extract one of them from the zipfile.
5.	Download the Python program ClassifyCommentSentiment.py to the new folder on your computer. Open this program and edit the lines below to point to the location of SentiStrength, its data files, and the YouTube comment file on your computer.
    1.	SentiStrengthLocation = ‘D:/Downloads/SentiStrength.jar’ #This must point to the location of SentiStrength on your computer
    2.	SentiStrengthUnzippedTextFilesLocation = ‘D:/SentiStrength_Data/’ #This must point to the location of the unzipped SentiStrength data files on your computer
    3.	FileToClassify = ‘E:/data/YouTube/BTS/BLACKPINK_eng-_NVwS4mcVYg_commentsOnly.txt’


## Workshop Trainers

**Yin Yin Lu** is a doctoral candidate at the Oxford Internet Institute (OII) and Balliol College, Oxford. Fascinated by the intersection between language and technology, she is exploring the resonance and rhetoric of Brexit tweets: what makes some messages more popular than others? Prior to joining the OII, Yin obtained a Masters in English Language from the University of Oxford and a Bachelor of Arts in English and Linguistics from Columbia University. Between these degrees, she worked at Pearson Education and a digital media agency in Manhattan. At Oxford, she is the founder and convenor of the TORCH #SocialHumanities research network and a member of the Global Leadership Initiative.

**Mike Thelwall** is a Professor of Information Science and leads the Statistical Cybermetrics Research Group at the University of Wolverhampton. He has developed and evaluated free software and methods for systematically gathering and analysing web and social web data. He is the developer of the SentiStrength software, which detects sentiment strength in social web text. Mike has co-authored hundreds of refereed journal articles and five books, including Social web research with Mozdeh. A full list of his publications can be found here.

**Folgert Karsdorp** is a post-doctoral researcher at the Meertens Institute of the Royal Netherlands Academy of Arts and Sciences. His research is interdisciplinary, adopting computational methods to study the field of humanities, in particular folkloristics. His research interests lie in the development of computational text analysis methods in the context of ethnology, anthropology, literary theory, and cultural evolution.

**Martin Wynne** is a digital research specialist in the Bodleian Libraries at the University of Oxford. Martin runs the Oxford Text Archive, a repository of digital literary and linguistic datasets, which has been in operation since 1976. Martin has worked in research, teaching, and support in corpus linguistics and digital humanities since the early 1990s. Martin is one of the founders of CLARIN, and has served as a member of the Executive Committee and the Board of Directors. He is National Coordinator for CLARIN in the UK.

**Chico Camargo** is a research assistant at the Oxford Internet Institute working on network science and traffic modelling. His doctoral research at the University of Oxford’s Department of Physics uses complex systems and data science to study evolution. Chico is fascinated not only by how genes and living things change, but also by how culture, language, and society changes. He is a core member of the TORCH #SocialHumanities network.

## Licence

The materials of this workshop are available under a [Creative Commons BY-CC-NC-SA 4.0 License](https://creativecommons.org/licenses/by-nc-sa/4.0/). See LICENSE for text.

Copyright 2017 Yin Yin Lu, Mike Thelwall, Folgert Karsdorp, Martin Wynne, Chico Camargo
