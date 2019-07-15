#Modified blog post code
#HamMuch1 copy, taking things out

#Hi Professor. So, yeah, I can get a lot of this to work with just "Hamlet", but that isn't what I want. There are also things that I can't get to work even with the simple, given Hamlet.

library(tidyverse)
library(gutenbergr)

#This Hamlet is not the right one. The one I've been using is "The Tragedy of Hamlet, Prince of Denmark" or gutenberg_download(1122). For some reason, either won't work. Thus, I also cannot get the cleaned data to work.
#titles <- c(
#  "Hamlet",
#  "Much Ado about Nothing"
#)
#books <- gutenberg_works(title %in% titles) %>%
#  gutenberg_download(meta_fields = "title") %>%
#  mutate(document = row_number())
#books

library(gutenbergr)
full_text <- gutenberg_download(1122)

cleaned_text0 <- full_text[-c(1:279, 1290:1297, 2189:2196, 3313:3320, 3534:3541, 3627:3634, 3930:3937, 3996:3403, 4238:4245, 5145:5152),]

cleaned_text1 <- cleaned_text0

names <- c("Ber", "Fran", "Mar", "Hor", "King", "Queen", "Cor", "Volt", "Laer", "Pol", "Ham", "Oph", "Ghost", "Rey", "Ros", "Guil", "For", "Capt", "Sailor", "Mess", "Clown", "Osr")
shakes_stop <- c("thee", "thou", "thy", "tis", "Tis", "hath", "hast", "Enter", "twill", "art", "thyself", "ere", "whence", "Exeunt", "twixt", "Exit", "thine", "canst", "oâer", "isât", "onât", "wherefore", "wither", "wilt", "shalt", "shouldst", "wouldst", "nay", "yea", "Ay", "ay", "twere", "thence", "ye", "twas", "prithee", "doth", "th", "hither", "Act", "ACT", "Scene","II", "III", "IV", "V", "VI", "VII", "1")

library(tm)
cleaned_text1$text <- unlist(lapply(cleaned_text1$text, FUN=removeWords, words=names))
cleaned_text1$text <- unlist(lapply(cleaned_text1$text, FUN=removeWords, words=shakes_stop))
cleaned_text1$title <- "Hamlet"

Pride <- gutenberg_download(1342)
Pride$title <- "Pride and Prejudice"

books <- rbind(cleaned_text1, Pride)
books$document <- 1:nrow(books)

# It seems to work if we read in the titles using their id numbers
# we need to add meta_fields="title" so it understands when we reference title later on.-AJS
#books <-gutenberg_download(c(1122, 1342), meta_fields = "title") %>%
#  mutate(document = row_number())
#books

library(tidytext)

tidy_books <- books %>%
  unnest_tokens(word, text) %>%
  group_by(word) %>%
  filter(n() > 10) %>%
  ungroup()

tidy_books

tidy_books %>%
  count(title, word, sort = TRUE) %>%
  anti_join(get_stopwords()) %>%
  group_by(title) %>%
  top_n(20) %>%
  ungroup() %>%
  ggplot(aes(reorder_within(word, n, title), n,
             fill = title
  )) +
  geom_col(alpha = 0.8, show.legend = FALSE) +
  scale_x_reordered() +
  coord_flip() +
  facet_wrap(~title, scales = "free") +
  scale_y_continuous(expand = c(0, 0)) +
  labs(
    x = NULL, y = "Word count",
    title = "Most frequent words after removing stop words",
    subtitle = "Words like 'said' occupy similar ranks but other words are quite different"
  )  

library(rsample)



books_split <- books %>%
  select(document) %>%
  initial_split()
train_data <- training(books_split)
test_data <- testing(books_split)    

sparse_words <- tidy_books %>%
  count(document, word) %>%
  inner_join(train_data) %>%
  cast_sparse(document, word, n)

class(sparse_words)

dim(sparse_words)

word_rownames <- as.integer(rownames(sparse_words))

books_joined <- data_frame(document = word_rownames) %>%
  left_join(books %>%
              select(document, title))

library(glmnet)
library(doParallel)
registerDoParallel(cores = 3)

#I changed the title to match the full title in the version we're using.-AJS
is_ham <- books_joined$title == "Hamlet"
model <- cv.glmnet(sparse_words, is_ham,
                   family = "binomial",
                   parallel = TRUE, keep = TRUE
)

#Professor, I do not recommend running either of these plots. They take forever on my computer and I have no idea what they mean anyway.
#thanks for the warning-AJS
#plot(model)

#plot(model$glmnet.fit)

library(broom)

coefs <- model$glmnet.fit %>%
  tidy() %>%
  filter(lambda == model$lambda.1se)

#This I simply cannot get to work. I always get some "stopifnot" error.

#I removed an extra 2 in line 104 (it read fct_reorder2(term, estimate))-AJS
coefs %>%
  group_by(estimate > 0) %>%
  top_n(10, abs(estimate)) %>%
  ungroup() %>%
  ggplot(aes(fct_reorder(term, estimate), estimate, fill = estimate > 0)) +
  geom_col(alpha = 0.8, show.legend = FALSE) +
  coord_flip() +
  labs(
    x = NULL,
    title = "Coefficients that increase/decrease probability the most",
    subtitle = "A document mentioning Martians is unlikely to be written by Jane Austen"
  )

intercept <- coefs %>%
  filter(term == "(Intercept)") %>%
  pull(estimate)

classifications <- tidy_books %>%
  inner_join(test_data) %>%
  inner_join(coefs, by = c("word" = "term")) %>%
  group_by(document) %>%
  summarize(score = sum(estimate)) %>%
  mutate(probability = plogis(intercept + score))

classifications

library(yardstick)

comment_classes <- classifications %>%
  left_join(books %>%
              select(title, document), by = "document") %>%
  mutate(title = as.factor(title))

comment_classes %>%
  roc_curve(title, probability) %>%
  ggplot(aes(x = 1 - specificity, y = sensitivity)) +
  geom_line(
    color = "midnightblue",
    size = 1.5
  ) +
  geom_abline(
    lty = 2, alpha = 0.5,
    color = "gray50",
    size = 1.2
  ) +
  labs(
    title = "ROC curve for text classification using regularized regression",
    subtitle = "Predicting whether text was written by Jane Austen or H.G. Wells"
  )

comment_classes %>%
  roc_auc(title, probability)

comment_classes %>%
  mutate(
    prediction = case_when(
      probability > 0.5 ~ "Pride and Prejudice",
      TRUE ~ "The Tragedy of Hamlet, Prince of Denmark"
    ),
    prediction = as.factor(prediction)
  ) %>%
  conf_mat(title, prediction)

comment_classes %>%
  filter(
    probability > .8,
    title == "The Tragedy of Hamlet, Prince of Denmark"
  ) %>%
  sample_n(10) %>%
  inner_join(books %>%
               select(document, text)) %>%
  select(probability, text)

comment_classes %>%
  filter(
    probability < .3,
    title == "Pride and Prejudice"
  ) %>%
  sample_n(10) %>%
  inner_join(books %>%
               select(document, text)) %>%
  select(probability, text)

