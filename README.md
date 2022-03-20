# Recommendation System Overview

> What is recommender system?
>
> Application that provided to users personalized recommendations about content that they may be interested in.
>
> For some side topic, check [here](https://github.com/LLan-ovoy/Memory-Baguette/blob/main/Recommendation_System/SIDE_TOPICS.md).

### Two Types of recommendation Methods

* > Content-based: analysis of the attributes of items and use simliarity between items to recommend items similar to what a user likes

  * creat content features for every items, but creating features is not easy

* > Collaborative filtering: use similarity between users and items simultaneously to provide recommendations, based on past user behavior (ratings, transactions)

  * similarities + former user behavior

  * need extra data - [feedbacks](https://github.com/LLan-ovoy/Memory-Baguette/blob/main/Recommendation_System/SIDE_TOPICS.md#side-topic-1-user-feedback)

    > explicit feedback: ratings by users
    >
    > implicit feedback: infer user preference by monitering user actions like purchase, clicks

#### Elements

* user item, context(device/location/time), interface(phone/PC..)

* User, item, utility matrix(sparse)

  * > utility matrix: degree of preference that a user has for an item, sparse(~1%)

* Architecture

### Content based Recommendations

* > Content based recommendation?
  >
  > Analysis of the attributes of the items, build user profiles based on the content features of the items rated by users, recommend new item based on items features and user profile
  >
  > Collaborative filtering: can be only based on user/item(KNN based), or both(MF)

- Ingredients for content based recommendation:
  - item representation: based on content
  - user representsation: based on item user liked
  - similarity metric/ machine learning model
  - ratings

#### Item profile and item-based User profile

- item profile

  - music item profile: 

    - audio content analysis
      - Features: beat..
      - high level descriptors from low-level features via Deep learning: genre, semantic
    -  text analysis: lyric, user generated context/review

  - movies item profile: actors, director, year, genre, publisher, location, image, audio

  - Document item profile: 

    - > TF-IDF: 
      > $$
      > tf(t,d) = \frac{count(t)}{|d|}\\
      > \\
      > \begin{align*}
      > df(t,N) &= \frac{|\{d_i:t \in d_i,i = 1,...,N\}|}{N}\\
      >  & \approx \frac{|\{d_i:t \in d_i,i = 1,...,N \}|+ 1}{N + 1}
      > \end{align*}
      > \\
      > \text{this is to avoid zero in side log in tfidf function}
      > $$
      >
      > $$
      > \begin{align*}
      > tfidf(t,d,N) &= \frac{tf(t,d)}{df(t,N)}\\
      >  & \approx  tf(t,d) \times log(\frac{1}{df(t,N)})
      > \end{align*}
      > $$

      - remove stopwords, compute scores, keep N words with high score as a vector for presentation 

    - Word embeddings/ sentence embedding

    - Topic modeling: the probability of a document in a topic

- User profile based on items: "user likes actor A,  director B, genre C..."

  - domographics
  - declared interests: select when creating a account
  - location/IP
  - usage features: last time visit, frequency
  - search history
  - item set: user showed interest, click, shared, liked

  

### Collabrative filtering

* memory based methods - KNN

  * > memory based methods: memorize the utility matrix, a type of KNN, tend to be slow

  * > KNN: computer distance between points, find the k nearest neighbors and vote/average

  * to predict whether alice is gonna like Joker

    * user based: find user similar to Alice and watched and rated joker

    * item based: find item similar to Joker that Alice wached and rated

    * > find movies that Alice rated that are similar to 'Joker', predict average rating on these movies for Alice's possile rating on 'Joker

  * [measure](https://github.com/LLan-ovoy/Memory-Baguette/blob/main/Recommendation_System/SIDE_TOPICS.md#side-topic-2-similarity-measurement)

    * Pearson correlation
    * Cosine similarity
    * Jaccard similarity

* model based methods - Matrix Factorization

  * > model based methods: fit a model, tend to have better performance

  * Matrix Factorization Loss function:

    > $$
    > \frac{1}{N} \sum_{(i,j): r_{ij}=1} (y_{ij}-u_iv_j)^2
    > $$

  * Gradient descent with momentum

    > $$
    > w \leftarrow random \ inital \\
    > v \leftarrow 0 \\
    > β= 0.9 \\ 
    > \text{for i = 1: max-iter do} \\
    > v \leftarrow \beta v+(1-\beta) \triangledown E(w) \\
    > w \leftarrow w - \eta v\\
    > \text{end}
    > $$
    >
    > 

  * Regularization, avoid overfitting, how to pick lambda

#### Steps to deploy

* First always, compute profile vectors for users and items

##### Memory based

> K-Nearest Neighbor(KNN)
>
> * steps:
>   * pick similarity measurement and compute similarity between users and items
>   * Recommend to user items with high similarity form maybe item or other user
>
> * Scale:
>   * When too many user and items, it is too expensive to compute similarity
>     * <span style="color:#D0505D">*locality-sensitive-hashing can be used to place item profiles in buckets?*</span>
>     * <span style="color:#D0505D">given a user, easy to find buckets with high similarity to user?</span>

##### Model based

> Matrix Factorization(MF)
>
> * Train a model using the feature vectors
>   * actually likes, ratings, clicks
>   * regression to predicted ratings, classification to predict prob of click
>   * can include features that are not content based

