library(tidyverse)
library(tidytext)
library(SnowballC)
library(data.table)
library(janitor)
library(lubridate)
library(quanteda)
library(slider)

complaints <- fread("Raw Data/complaints-2024-04-05_11_12.csv")

complaints <- complaints %>%
  clean_names() %>%
  mutate(date_received = mdy(date_received)) %>%
  filter(date_received < ymd("2020-01-01"))

# word counts
complaints %>%
  unnest_tokens(word, consumer_complaint_narrative) %>%
  anti_join(get_stopwords(), by = "word") %>%
  mutate(stem = wordStem(word)) %>%
  count(complaint_id, stem) %>%
  cast_dfm(complaint_id, stem, n)

#tf-idf - relative frequency of words
complaints %>%
  unnest_tokens(word, consumer_complaint_narrative) %>%
  anti_join(get_stopwords(), by = "word") %>%
  mutate(stem = wordStem(word)) %>%
  count(complaint_id, stem) %>%
  bind_tf_idf(stem, complaint_id, n) %>%
  cast_dfm(complaint_id, stem, tf_idf)

# filtering out rarely used words

tidy_complaints <- complaints %>%
  select(complaint_id, consumer_complaint_narrative) %>%
  unnest_tokens(word, consumer_complaint_narrative) %>%
  add_count(word) %>%
  filter(n >= 50) %>%
  select(-n)

nested_words <- tidy_complaints %>%
  nest(words = c(word))

nested_words

## Word Embeddings
# 
# The function identifies skipgram windows in order to calculate
# the skipgram probabilities,
# how often we find each word near each other word. 
# We do this by defining a fixed-size moving window that centers around each word. 
# Do we see word1 and word2 together within this window? 
# We can calculate probabilities based on when we do or do not.

# One of the arguments to this function is the window_size, 
# which determines the size of the sliding window that moves
# through the text, 
# counting up words that we find within the window. 
# The best choice for this window size depends on your 
# analytical question because it determines what kind of
# semantic meaning the embeddings capture.
# A smaller window size, like three or four, 
# focuses on how the word is used and learns 
# what other words are functionally similar.
# A larger window size, like 10, 
# captures more information about the domain or topic of each word, 
# not constrained by how functionally similar the words are (Levy and Goldberg 2014). 
# A smaller window size is also faster to compute.

slide_windows <- function(tbl, window_size) {
  skipgrams <- slider::slide(
    tbl, 
    ~.x, 
    .after = window_size - 1, 
    .step = 1, 
    .complete = TRUE
  )
  
  safe_mutate <- safely(mutate)
  
  out <- map2(skipgrams,
              1:length(skipgrams),
              ~ safe_mutate(.x, window_id = .y))
  
  out %>%
    transpose() %>%
    pluck("result") %>%
    compact() %>%
    bind_rows()
}

## Intensive computation
library(widyr)
library(furrr)

plan(multisession)  ## for parallel processing

system.time(tidy_pmi <- nested_words %>%
  mutate(words = future_map(words, slide_windows, 4L)) %>%
  unnest(words) %>%
  unite(window_id, complaint_id, window_id) %>%
  pairwise_pmi(word, window_id))

tidy_pmi


## Alternativa

#create context window with length 8
tidy_skipgrams <- elected_no_retweets %>%
  unnest_tokens(ngram, text, token = "ngrams", n = 8) %>%
  mutate(ngramID = row_number()) %>% 
  tidyr::unite(skipgramID, postID, ngramID) %>%
  unnest_tokens(word, ngram)

tidy_complaints <- complaints %>%
  select(complaint_id, consumer_complaint_narrative) %>%
  unnest_tokens(ngram, consumer_complaint_narrative,  token = "ngrams", n = 4) %>%
  mutate(ngramID = row_number()) %>%
  tidyr::unite(skipgramID, complaint_id, ngramID) %>%
  unnest_tokens(word, ngram)
  add_count(word) %>%
  filter(n >= 50) %>%
  select(-n)

#calculate unigram probabilities (used to normalize skipgram probabilities later)
unigram_probs <- elected_no_retweets %>%
  unnest_tokens(word, text) %>%
  count(word, sort = TRUE) %>%
  mutate(p = n / sum(n))

#calculate probabilities
skipgram_probs <- tidy_skipgrams %>%
  pairwise_count(word, skipgramID, diag = TRUE, sort = TRUE) %>%
  mutate(p = n / sum(n))

#normalize probabilities
normalized_prob <- skipgram_probs %>%
  filter(n > 20) %>%
  rename(word1 = item1, word2 = item2) %>%
  left_join(unigram_probs %>%
              select(word1 = word, p1 = p),
            by = "word1") %>%
  left_join(unigram_probs %>%
              select(word2 = word, p2 = p),
            by = "word2") %>%
  mutate(p_together = p / p1 / p2)

normalized_prob[2005:2010,]

normalized_prob %>% 
  filter(word1 == "trump") %>%
  arrange(-p_together)


# SVD

pmi_matrix <- normalized_prob %>%
  mutate(pmi = log10(p_together)) %>%
  cast_sparse(word1, word2, pmi)

library(irlba)

#remove missing data
pmi_matrix@x[is.na(pmi_matrix@x)] <- 0
#run SVD
pmi_svd <- irlba(pmi_matrix, 256, maxit = 500)
#next we output the word vectors:
word_vectors <- pmi_svd$u
rownames(word_vectors) <- rownames(pmi_matrix)