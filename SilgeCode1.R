#From the blog post
#Code by Julia Silge

library(tidyverse) #Downloads tidyverse package
library(gutenbergr) #Downloads gutenbergr package

titles <- c(
  "The War of the Worlds",
  "Pride and Prejudice"
) #Assigns "titles" to be the complete books of Pride and Prejudice and The War of the Worlds
books <- gutenberg_works(title %in% titles) %>% 
  gutenberg_download(meta_fields = "title") %>%
  mutate(document = row_number()) #Not so sure here.
books #Look at what we just named "books"

library(tidytext) #Download tidytext package

tidy_books <- books %>% #We're taking books and we're going to...
  unnest_tokens(word, text) %>% #Split the columns into tokens, and these tokens are going to be individual words
  group_by(word) %>% #Make groups out of the words
  filter(n() > 10) %>% #Just show us words that occur more than ten times
  ungroup() #Return grouped data back to an ungrouped form
#We just put books into tidy format, which means each variable has a column, each observation has a row, and each type of observational unit is a table
tidy_books #Look at what we just named "tidy_books"

tidy_books %>% #We're taking tidy_books and we're going to...
  count(title, word, sort = TRUE) %>% #Count the words in the data
  anti_join(get_stopwords()) %>% #Compares the two datasets but omits similarities
  group_by(title) %>% #Two datasets are grouped by the title of each
  top_n(20) %>% #Just give us the top 20
  ungroup() %>% #Return grouped data back to an ungrouped form
  ggplot(aes(reorder_within(word, n, title), n,
             fill = title
  )) + #Make a plot and deal with the aesthetic
  geom_col(alpha = 0.8, show.legend = FALSE) + #something to do with colour
  scale_x_reordered() + #works with x axis
  coord_flip() + #rotates graph
  facet_wrap(~title, scales = "free") + #wraps a 1d sequence of panels into 2d
  scale_y_continuous(expand = c(0, 0)) + #works with y axis
  labs(
    x = NULL, y = "Word count",
    title = "Most frequent words after removing stop words",
    subtitle = "Words like 'said' occupy similar ranks but other words are quite different"
  ) #works with titles

library(rsample) #Download rsample package

books_split <- books %>% #We're taking books and we're going to...
  select(document) %>% #take the data and...
  initial_split() #split it up and...
train_data <- training(books_split) #this half is what we're using to train the model and...
test_data <- testing(books_split) #this half is what we'll test the model on

sparse_words <- tidy_books %>% #going from tidy data to a sparse matrix
  count(document, word) %>% #count the words in the data
  inner_join(train_data) %>% #using the training data
  cast_sparse(document, word, n) #make it sparse

class(sparse_words) 

dim(sparse_words) #set the dimensions of sparse_words

word_rownames <- as.integer(rownames(sparse_words))

books_joined <- data_frame(document = word_rownames) %>%
  left_join(books %>%
              select(document, title))

library(glmnet) #download glmnet package
library(doParallel) #download doParallel package
registerDoParallel(cores = 8) #register doParallel, and use 8 cores to do it

is_jane <- books_joined$title == "Pride and Prejudice" #is_jane is when the sentence is from Pride and Prejudice
model <- cv.glmnet(sparse_words, is_jane,
                   family = "binomial", #either it is or it isn't
                   parallel = TRUE, keep = TRUE
)

plot(model) #a plot of the model, but I have no idea what kind of plot

plot(model$glmnet.fit) #again, I have no idea what kind of plot this is

library(broom) #download broom package

coefs <- model$glmnet.fit %>% 
  tidy() %>%
  filter(lambda == model$lambda.1se)

coefs %>%
  group_by(estimate > 0) %>% #coefficients are each grouped by their estimates, given they are greater than zero
  top_n(10, abs(estimate)) %>% #show top ten #only show the top ten, and in absolute value of the estimate
  ungroup() %>% #Return grouped data back to an ungrouped form
  ggplot(aes(fct_reorder2(term, estimate), estimate, fill = estimate > 0)) + #Make a plot and deal with the aesthetic
  geom_col(alpha = 0.8, show.legend = FALSE) + #something about colour
  coord_flip() + #rotates graph
  labs(
    x = NULL,
    title = "Coefficients that increase/decrease probability the most",
    subtitle = "A document mentioning Martians is unlikely to be written by Jane Austen"
  ) #works with titles

intercept <- coefs %>%
  filter(term == "(Intercept)") %>% #filter through by that which we have named intercept
  pull(estimate) #pull out the single variable of estimates

classifications <- tidy_books %>%
  inner_join(test_data) %>%
  inner_join(coefs, by = c("word" = "term")) %>% #Join together the table of test data and the table of coefficients
  group_by(document) %>% #group them by the document
  summarize(score = sum(estimate)) %>% #summarise the sum of the estimates
  mutate(probability = plogis(intercept + score)) #add the variable of probability into all of this

classifications #show all that

library(yardstick) #download yardstick package

comment_classes <- classifications %>%
  left_join(books %>%
              select(title, document), by = "document") %>%  #join together classifications and books
  mutate(title = as.factor(title)) #also add the variable of the title

comment_classes %>%
  roc_curve(title, probability) %>% #Make an ROC curve
  ggplot(aes(x = 1 - specificity, y = sensitivity)) + #aesthetic prep
  geom_line(
    color = "midnightblue",
    size = 1.5
  ) + #design detail like colour
  geom_abline(
    lty = 2, alpha = 0.5,
    color = "gray50",
    size = 1.2
  ) + #parametres for graph
  labs(
    title = "ROC curve for text classification using regularized regression",
    subtitle = "Predicting whether text was written by Jane Austen or H.G. Wells"
  ) #title of graph

comment_classes %>%
  roc_auc(title, probability) #give area under the curve of this roc line as it relates to probability

comment_classes %>%
  mutate( #add in...
    prediction = case_when( #prediction for...
      probability > 0.5 ~ "Pride and Prejudice", #when over half of the examples are Pride and Prejudice
      TRUE ~ "The War of the Worlds" #but true means that it is War of the Worlds?
    ),
    prediction = as.factor(prediction)
  ) %>%
  conf_mat(title, prediction) #calculates cross-tabulation of observed and predicted classes

comment_classes %>%
  filter(
    probability > .8, #probability of it being WW is more than 80%
    title == "The War of the Worlds"
  ) %>%
  sample_n(10) %>% #show ten
  inner_join(books %>%
               select(document, text)) %>%
  select(probability, text) #join together the table of books with the comment classes

comment_classes %>%
  filter(
    probability < .3,
    title == "Pride and Prejudice"
  ) %>%
  sample_n(10) %>%
  inner_join(books %>%
               select(document, text)) %>%
  select(probability, text) 
# same, but different probability
