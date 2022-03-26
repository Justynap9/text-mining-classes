from matplotlib import pyplot as plt
from read_file import read_csv
from cleaning import cleaning, stop_words, stemming
from wordcloud import WordCloud


text = read_csv("files/True.csv")
text = cleaning(text)
text = stop_words(text)
list_w = stemming(text)


def word_count(wordlist: list[str]) -> dict:
    counts = dict()
    for word in wordlist:
        if word in counts:
            counts[word] += 1
        else:
            counts[word] = 1
    return counts


word_cloud = WordCloud()
word_cloud.generate_from_frequencies(word_count(list_w))
plt.imshow(word_cloud, interpolation='bilinear')
plt.axis("off")
plt.show()
