#HamMuch2 copy, taking things out

library(gutenbergr)
full_text <- gutenberg_download(1122)

cleaned_text0 <- full_text[-c(1:279, 1290:1297, 2189:2196, 3313:3320, 3534:3541, 3627:3634, 3930:3937, 3996:3403, 4238:4245, 5145:5152),]

cleaned_text1 <- cleaned_text0

names <- c("Ber", "Fran", "Mar", "Hor", "King", "Queen", "Cor", "Volt", "Laer", "Pol", "Ham", "Oph", "Ghost", "Rey", "Ros", "Guil", "For", "Capt", "Sailor", "Mess", "Clown", "Osr")
shakes_stop <- c("thee", "thou", "thy", "tis", "Tis", "hath", "hast", "Enter", "twill", "art", "thyself", "ere", "whence", "Exeunt", "twixt", "Exit", "thine", "canst", "o’er", "is’t", "on’t", "wherefore", "wither", "wilt", "shalt", "shouldst", "wouldst", "nay", "yea", "Ay", "ay", "twere", "thence", "ye", "twas", "prithee", "doth", "th", "hither", "Act", "ACT", "Scene","II", "III", "IV", "V", "VI", "VII", "1")

library(tm)
cleaned_text1$text <- unlist(lapply(cleaned_text1$text, FUN=removeWords, words=names))
cleaned_text1$text <- unlist(lapply(cleaned_text1$text, FUN=removeWords, words=shakes_stop))

tidy_book <- cleaned_text1 %>%
  mutate(line = row_number()) %>%
  unnest_tokens(word, text)
tidy_book

tidy_book %>%
  count(word, sort = TRUE)

get_stopwords()

get_stopwords(source = "smart")

tidy_book %>%
  anti_join(get_stopwords(source = "smart")) %>%
  count(word, sort = TRUE) %>%
  top_n(20) %>%
  ggplot(aes(fct_reorder(word, n), n)) +
  geom_col() +
  coord_flip()

get_sentiments("afinn")
get_sentiments("bing")
get_sentiments("loughran")
#get_sentiments("nrc")

tidy_book %>%
  inner_join(get_sentiments("bing")) %>%
  count(sentiment, sort = TRUE)

tidy_book %>%
  inner_join(get_sentiments("bing")) %>%            
  count(sentiment, word, sort = TRUE)

tidy_book %>%
  inner_join(get_sentiments("bing")) %>%
  count(sentiment, word, sort = TRUE) %>%
  group_by(sentiment) %>%
  top_n(10) %>%
  ungroup %>%
  ggplot(aes(fct_reorder(word, n),
             n, 
             fill = sentiment)) +
  geom_col() +
  coord_flip() +
  facet_wrap(~ sentiment, scales = "free")

#Nice

tidy_ngram <- cleaned_text1 %>%
  unnest_tokens(bigram, text, token = "ngrams", n = 2)
tidy_ngram

tidy_ngram %>%
  count(bigram, sort = TRUE)

tidy_ngram %>%
  separate(bigram, c("word1", "word2"), sep = " ") %>%
  filter(!word1 %in% stop_words$word,
         !word2 %in% stop_words$word) %>%
  count(word1, word2, sort = TRUE)

