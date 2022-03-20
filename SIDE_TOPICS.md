# Recommendation System Side Topics

### SIDE TOPIC 1: User Feedback

- Feedback

  - explicit: Ask for rating - movies, online stores

    - User are unwilling to provide responses

    - > Implicit feedback:  biased by only from people who willing to provide ratings

  - implicit: infer user preference

    - purchase history, browsing history, search patterns

    - Pros: easy to get

    - Cons:

      - No nagative feedback, positive-only data

        - > Missing as negative: fill empty with zeros 
          >
          > * matrix is fully defined no need $(i,j): r_{ij}=1$, but can be expensice
          >
          > * may want to weight 0s less than 1s: weak negatives
          >
          > Negative sampling: sample empty spaces to get zeros
          >
          > * uniform random sampling
          > * user-oriented sampling: if a user has viewed more items, then those he/she hasn'e viewed can be sampled as negative with higher probability
          > * Item-orinted sampling: a more popular item has higher probability to be sampled as negative

      - No level of preference: how much time spent or how many times

      

### SIDE TOPIC 2: similarity measurement

> meansure of how similar, used in KNN

- Jaccard similarity
  $$
  \frac{|A \cap B | \leftarrow \text{intersection}}{|A\cup B| \leftarrow \text{union- - - - -}}
  $$

  * > when to use Jaccard similarity?
    >
    > * measure similarity between two sets
    >
    > * binary features, will lose information if used in non-binary features
    >   * <span style="color:#D0505D">*how to take binary features to sets?*</span>
    >   * <span style = 'color:#d0505d'>*how to go from x,y to A,B?*</span>
    > * model lack of ratings, since it only count numbers

- Cosine similarity
  $$
  \begin{align*}
  sim(x,y) = cos(\theta) &= \frac{x\cdot y}{|x|\times|y|} \\
  & = \frac{\sum_{i=1}^n x_iy_i}{\sqrt{\sum_{i=1}^{n}x_i^2}{\sqrt{\sum_{i=1}^{n}y_i^2}}}
  \end{align*}
  $$

  * > when to use cosine similarity?
    >
    > * often used in similarity between documents
    > * lack of rating is treated as 0, more similar to disliking than liking
    > * of used in positve space, $x_i, y_i \ge 0$

- Pearson coorelation coefficient
  $$
  r(x,y) = \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i-\bar{y})}{\sqrt{\sum_{i=1}^{n}(x_i - \bar{x})^2}{\sqrt{\sum_{i=1}^{n}(y_i - \bar{y})^2}}}
  $$

  * > when to use pearson coorelation?
    >
    > Measure of linear correaltion between two variables
    >
    > 1 is total positive correlation, 0 is no correlation, -1 is total negative correlation



### SIDE TOPIC 3: Cold start problem

- new user with no implicite or explicite feedback
  - popular thing
  - ask for more info - tag yourself
  - social network assumption: people have similar tastes to their friends
- new item with no rating
  - content feature
