import re

# ex. 1

text = "Dzisiaj mamy 4 stopnie na plusie, 1 marca 2022 roku"
result = re.sub(r'[0-9]+', '', text)
print(result)

text1 = "<div><h2>Header</h2> <p>article<b>strong text</b> <a href="">link</a></p></div>"
result1 = re.sub('<.*?>', '', text1)
print(result1)

text2 = "Lorem ipsum dolor sit amet, consectetur; adipiscing elit.\
Sed eget mattis sem. Mauris egestas erat quam, ut faucibus eros congue et. In\
blandit, mi eu porta; lobortis, tortor nisl facilisis leo, at tristique augue risus\
eu risus."
result2 = re.sub(r'[^\w\s]', '', text2)
print(result2)

# ex. 2

text3 = "Lorem ipsum dolor\
sit amet, consectetur adipiscing elit. Sed #texting eget mattis sem. Mauris #frasista\
egestas erat #tweetext quam, ut faucibus eros #frasier congue et. In blandit, mi eu porta\
lobortis, tortor nisl facilisis leo, at tristique #frasistas augue risus eu risus."
hashtags = re.findall('#(\w+)', text3)
print(hashtags)

# ex. 3
text4 = "Lorem ipsum dolor sit amet, consectetur adipiscing elit.\
Sed :) ;) eget mattis sem. Mauris :> egestas erat ;( quam,\
ut faucibus eros :-) congue et. :< ;<  In blandit, mi eu porta\
lobortis, tortor nisl facilisis leo, at tristique ;-) augue risus eu risus."
emoticons = re.findall('[:;]\W+', text4)
print(emoticons)


