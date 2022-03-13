import text_cleaning
import stemming

text = " <html> <body> Beautiful interior, excellent" \
       " service and very friendly staff." \
       " I spent a lovely evening here. :) " \
       " We had the Cote de Boeuf to share which was yummy." \
       " Then the chocolate 9 23 and raspberry slice was very nice." \
       " I would recommend this place and I would love to try their" \
       " breakfast and Sunday lunch in the future. ;)"

text = text_cleaning.cleaning(text)
text = text_cleaning.stop_words(text)
print(text)
print(stemming.stemming(text))
