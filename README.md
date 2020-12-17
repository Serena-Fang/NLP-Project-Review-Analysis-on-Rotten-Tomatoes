# Review-Analysis-on-Rotten-Tomatoes
Use NLP to perform word vectorization, topic modeling and sentiment analysis on movie reviews.

**Review Analysis on Rotten Tomatoes**
Sophy Huang, Julian Zhao, Zhou Fang

In recent years, watching a movie has always been the top entertainment for people on weekends and holidays. According to the Numbers statistics, US cinemas ticket sales reached 1,235,503,823 (“Movie Market” n.d.) in 2019, which means that a person watched 4 movies in a year on average in the United States. However, as people are busy nowadays and the movie tickets are not that cheap, people tend to turn to movie review websites before they decide on a movie so that they won’t waste time and money on an unsatisfying one. Studies have shown that scores on movie review websites became increasingly tightly linked to movie ticket sales (Wilkinson, 2017). As some moviegoers look up movie reviews before watching a movie, other people also utilize the review websites to share experiences and engage in further discussion after watching a movie. Contemporary recognition of the significance of movie review websites stems mainly from these two trends. In recent years, movie review websites such as Roger Ebert, IMDB, Guardian, and Rotten Tomatoes have gained great attention and popularity.

In our project, we focused on movie reviews on Rotten Tomatoes, which is an American review-aggregation website for film and television (“Rotten Tomatoes” n.d.). Nowadays, it has become one of the most popular movie review websites in the US. Rotten Tomatoes is a website for quick rating (“8 Best” n.d.). People mark their review “fresh” if it’s generally favourable or “rotten” otherwise. Moreover, Rotten Tomatoes has been a trusted movie review source as it uses critics’ data for overall scoring of a film. The website categorizes critics’ and top critics’ reviews. To be specific, to be accepted as a “critic” on Rotten Tomatoes, his or her original reviews must garner a specific number of “likes” from users. Those classified as “top critics” generally write for major newspapers. (“Movie & TV Critics” n.d.) If 60% or more of critics (including top critics) like a movie, the movie earns an overall Fresh score with a red tomato. A movie earns a Rotten score with a green splat if under 60% of critics rate the movie favorably. Since Rotten Tomatoes has been widely used and has clear categories for sentiment and critics, we’re interested in performing review analysis on this website.

In our project, we used different models and methods to analyze movie reviews from the Rotten Tomatoes website. Our research question is mainly composed of two parts:

**1) Do top critics’ reviews differ from normal critics’? If so, how do they differ?**
The original dataset contains three categories of reviewers: amateurs, critics, and top critics. In our project, we focused on the dataset of reviews from critics and top critics. In the modern era of the Internet, it has become significantly easier to submit a body of text and potentially have the entire world see it. However, though this brings a plethora of beautiful possibilities, it has also arguably cheapened the realm of professional publication. Rotten Tomatoes review website selects the critics' review category for people to get more meaningful information with less time. Classifying features of both top critics’ and critics’ reviews could help the Rotten Tomatoes operation team check critics’ performance, consider the necessity of separating top critics from critics, and inspire them an optimized selection of top critics and critics in the future. Moreover, the analysis results may give insights for amateurs reviewers who want to be professional critics, and for critics who want to become top critics.

**2) Do “rotten” reviews and “fresh” reviews differ? If so, how do they differ?**
By classifying movie reviews into positive and negative categories, we could see what the most frequent words people are using to describe a good/bad movie, and what the most frequent reasons are for people to like/dislike a movie. We could summarize features of those compliments and complaints and give suggestions to future movie makers so that they could learn from past movies’ merits and avoid drawbacks when producing future works.

**II. Relevant studies**
From what we’ve discovered, no study has touched on the novel problem of distinguishing between top critics’ comments and non-top critics’ comments. Nevertheless, many studies focus on the binary classification problem of positive and negative comments using sentiment analysis, vectorization, and topic modeling.

https://towardsdatascience.com/sentiment-analysis-with-python-part-1-5ce197074184
In this project, the author implements vectorization on a IMDB movie review dataset and builds a machine learning model for a binary classification problem (positive comment vs. negative comment). The author applies simple regular expression as preprocessing, utilizes a binary CountVectorizer for the transformation, and uses Logistic Regression as the training model. As a result, the model achieves an accuracy of 0.88 for classifying positive and negative comments. The author then looks at the five most distinguished words for both positive and negative comments by sorting the weights of the Logistic Regression. These approaches are similar to our attempts of differentiating subjects of interest by word choice, but what is different in our project is our goal. We aim to use classification methods to see how much we can distinguish two groups, while they aim to build a classification model with high accuracy.

https://towardsdatascience.com/sentiment-analysis-a-how-to-guide-with-movie-reviews- 9ae335e6bcb2
The project called “Sentiment Analysis — A how-to guide with movie reviews” by Shiao-li Green on Towardsdatascience website uses the IMDB movie dataset which has 25,000 labeled reviews for training and 25,000 reviews for testing. The project aims to train a sentiment analysis model and predict the sentiment of testing reviews. After preprocessing data, the author tries several methods—Bag of Words, Word2Vec, and Tf-idf—to transform word representations to numerical versions so that a model can interpret. Then, the author attempts two different models for this dataset: Naïve Bayes and Random Forest. The result shows that Tf-Idf does marginally better, and Naïve Bayes performs slightly better than the Random Forest. The combination of Naïve Bayes and Tf-Idf reaches an accuracy of 0.8478 of sentiment prediction. This project looks at sentiments of movie reviews in a different direction than ours but provides us with insights into how different methods aid in analyzing sentiments. Our project steps into the conversation by performing sentiment analysis to distinguish attitudes from groups of top critic vs. critic reviews and fresh vs. rotten reviews.

**III. Data**
The dataset for our project is provided through the Kaggle platform by user Stefano Leone, who scraped the movie reviews from the publicly available website Rotten Tomatoes (https://www.rottentomatoes.com). The data has been directly scraped from the website, but the technique is not disclosed by Stefano Leone. The data is collected and updated as of 2020-10-31. Rotten Tomatoes was launched on August 12, 1998. So the data we use in our project would be a collection of movie reviews from 1998 to 2020.

Instances in the dataset include rotten_tomatoes_link (i.e., link from which the review data was scraped), critic_name (i.e., name of users who rated the movie), top_critic (i.e., a boolean value of whether the critic is a top critic or not), publisher_name (i.e., name of the publisher for which the critic works), review_type (i.e., fresh or rotten), review_score, review_date, and review_content. There are a total of 17,712 instances, and each instance consists of raw data (directly scraped from the website).
One side note is that every movie review page consists of 2 review sections—critic reviews and audience reviews. However, because many movies do not have featured audience reviews but most movies have critic reviews, and critic reviews come from professionals, we will only be focusing on critic reviews for this project, which corresponds to this dataset. For the review_type variable, it is labeled according to the Tomatometer score provided by critics: it shows as “rotten” when at least 60% of reviews are positive and “rotten” when at least 60% of reviews are negative. We will only be using top_critic, review_type, and review_content in this project.


**IV. Methods**
We did all of our analysis with Python. We wrote and shared our code via Jupyter Notebook. We explored differences between fresh and rotten reviews, critics and top critics reviews mainly from the following three dimensions (with specific tools in parentheses):
1) **Word Choice (CountVectorizer + Logistic Regression)**
2) **Topics/Focuses (Topic Modelling)**
3) **Sentiments (Sentiment Analysis)**

Here we’ll show the methods we used for analyzing critics’ and top critics’ reviews as a demonstration of our approaches. Same methods were applied to fresh and rotten reviews.

**Preprocessing**
First, we preprocessed our dataset by performing text normalization. We did this by
1) Getting rid of null content (NA’s)
2) Turning all text into lowercase
3) Tokenizing text (split text into a list of words)
4) Removing stopwords (a set of very common words like the, a, and, etc.)
5) Removing special characters (so that our word lists only contains letters a-z and
numbers 0-9)

We added a column called “processed_review” to store processed textual data so that our data was ready for analysis. Then we selected only the “processed_review” column and the column indicating our groups of interest (in this case, top_critic). We also performed subsetting to further separate the selected dataset (rt_critic) into two datasets: one contained only reviews written by top critics (rt_top), and the other contained only those written by critics (rt_nottop). These datasets were helpful in later approaches of topic modeling and sentiment analysis.

**1) Word Choices (CountVectorizer + Logistic Regression)**
We used both classification and frequent word to explore word choice differences between different categories (top critic vs. critic reviews, fresh vs. rotten reviews). We randomly selected 50,000 reviews (25,000 each) from the original dataset to balance our dataset used for classification and converted reviews written by top critics to 1’s and those written by critics to 0’s.

**Classification**
After tokenization, we vectorized the text by setting up CountVectorizer from sklearn’s library and fitting it on our dataset to transform the text into a frequency count in the form of vectors. To be specific, we looked at all the unique words in our dataset and then counted how many times these words appeared in each of the pieces of texts, resulting in a vector representation of each piece of text.

Then, we ran the classifier. We used train_test_split (function from the sklearn package) to divide our dataset into training set and testing set. Since our focus is to determine if it is plausible to differentiate comments using countVectorizer (instead of building the best model to classify them), there is no need to build a complicated system to estimate the exact accuracy of our model. In comparison, implementing train_test_split to evaluate our accuracy is time efficient. After splitting the training and testing dataset, we used a logistic regression model to classify critics’ reviews and top critics’ reviews.

We also calculated featured weights for 5 words that had the highest coefficients (i.e., words that most likely indicated 1’s, in this case, top critics) and 5 words that had the lowest coefficients (i.e., words that most likely indicated 0’s, in this case, critics).

**Most Frequent Words**
When calculating most frequent words, we looked at the subsets. For example, we counted the Top 20 Most Frequent words in top critics’ reviews by first creating a new dataframe with all the terms and their frequencies. Since the dataframe was ordered with ascending frequency, the tail 20 words were the Top 20 most frequent words. We analyzed them in forms of graphs and tables, and we will present the visualizations in the analysis/discussion section.

**2) Topics/Focuses (topic modeling)**
In order to perform topic analysis, we used Little MALLET Wrapper in Python. MALLET (originally written in Java) is a software package for topic modeling and other natural language processing techniques. We created a sub folder and changed directories to store out outputs. We set the number of topics to 5. Then we imported the data and used the little mallet wrapper to train the topic model for top critics’ reviews and critics’ reviews. Then we printed the 5 topics with 20 keywords each. Detailed explanations of outputs will be given in the Analysis/Discussion section.

**3) Sentiments (Sentiment Analysis)**
The last part of our methods was to analyze sentiments. We performed sentiment analysis (vader’s SentimentIntensityAnalyzer) on processed review text within the top critic/critic dataset, and then put positive, negative, neutral and compound scores of each review into the dataframe. The compound score refers to the overall sentiment score of a review (1 means most extreme positive, -1 means most extreme negative). In the three subcategories of sentiment (positive, negative, and neutral), the score ranges from 0-1.

Then we grouped the results by top critic (top_critic==True) and critic (top_critic==False) and printed their corresponding summaries of compound/negative/neutral/positive score. The example below is a summary of the compound score for reviews written by top critics and critics, which includes useful statistics such as number of records (count), average score (mean), standard deviation (std), minimum and maximum (min/max). Another method we implemented was independent samples t-test using ttest_ind from scipy stats, which compared the means of scores for two groups (in this case, top critics and critics). This function returns statistic, or t-statistic, and p-value.
