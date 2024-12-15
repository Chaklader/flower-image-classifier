# Neural Networks
# ===============

# Perceptron Algorithm

A perceptron is one of the simplest forms of artificial neural networks - imagine it as a basic decision-maker that draws a line to separate two groups of data points. Just like how you might draw a line to separate apples from oranges based on their features, a perceptron tries to find the best dividing line between two classes of data.

## Technical Details

### Basic Formula
The prediction (ŷ) is given by:
ŷ = step(w₁x₁ + w₂x₂ + b)

Where:
- (w₁, w₂) are weights
- (x₁, x₂) are input features
- b is the bias term
- step() is the step function

### Update Rules
For a point with coordinates (p, q) and label y:

1. If correctly classified:
   - No changes needed to weights or bias

2. If classified positive but actually negative:
   - w₁ = w₁ - αp
   - w₂ = w₂ - αq
   - b = b - α

3. If classified negative but actually positive:
   - w₁ = w₁ + αp
   - w₂ = w₂ + αq
   - b = b + α

Where α is the learning rate.

### Key Points:
- Algorithm iteratively adjusts the decision boundary
- Learning continues until all points are correctly classified or maximum iterations reached
- Convergence guaranteed if data is linearly separable
- Visualized as a line in 2D space or hyperplane in higher dimensions

The final solution shows the optimal decision boundary (solid line), while the dotted lines show the algorithm's learning progression.



# Log-Loss Error Function

Think of log-loss as a way to measure how wrong your predictions are in classification problems. Instead of just counting incorrect predictions, it punishes you more heavily when you're very confident about a wrong prediction (like being 99% sure someone is a fraud when they're not) and less when you're uncertain. It's like a teacher who takes off more points when you write a wrong answer confidently versus when you show some doubt.

## Technical Details

### Basic Formula
Log Loss = -1/N ∑(y_i * log(p_i) + (1-y_i) * log(1-p_i))

Where:
- N is the number of observations
- y_i is the actual value (0 or 1)
- p_i is the predicted probability
- log is the natural logarithm

### Properties
1. Always positive
2. Lower values are better
3. Perfect predictions → Log Loss = 0
4. Approaches infinity as predictions get worse

### Key Points:
- Used primarily in binary classification
- Common in logistic regression
- Differentiable (good for gradient descent)
- Penalizes confident incorrect predictions heavily
- Rewards accurate probability estimates

### Practical Applications:
- Machine learning model evaluation
- Model comparison
- Risk assessment
- Probability calibration
- Classification tasks

Remember: Log-loss is particularly useful when you need probabilistic predictions rather than just class labels.


# Predictions

The prediction is essentially the answer we get from the algorithm:

- A discrete answer will be of the form "no" or "yes" (or 0 or 1).
- A continuous answer will be a number, normally between 0 and 1.

When our prediction gives us a continuous number between 0 and 1, this is equivalent to the probability that the given data point should belong to the given classification (e.g., a data point might get a prediction of 0.5, with this being equivalent to a 50% probability that it is correctly classified).

With our linear example, the probability is a function of the distance from the line. The further a data point is from the line, the larger the probability that it is classified correctly.

# Sigmoid Functions

The way we move from discrete predictions to continuous is to simply change our activation function from a step function to a sigmoid function. A sigmoid function is an s-shaped function that gives us:

- Output values close to 1 when the input is a large positive number
- Output values close to 0 when the input is a large negative number
- Output values close to 0.5 when the input is close to zero

The formula for our sigmoid function is:

σ(x) = 1/(1 + e^(-x))

Before our model consisted of a line with a positive region and a negative region. But now, by applying the sigmoid function, our model consists of an entire probability space where, for each point, we can get a probability that the classification is correct.




# Multi-Class Classification and Softmax

Imagine sorting different fruits into baskets - you don't just decide between apples and oranges, but also bananas, pears, and more. That's multi-class classification, and softmax is like your brain calculating the probability of which basket each fruit belongs in. Instead of a simple yes/no decision, it gives you percentage chances for each possible category.

## Technical Details

### Softmax Function
For a vector z of K classes:
```
softmax(z_i) = e^(z_i) / Σ(e^(z_j))
```
Where:
- z_i is the input for class i
- e is Euler's number
- Σ runs over all classes j

### Properties
1. Outputs sum to 1 (100%)
2. All outputs are between 0 and 1
3. Larger inputs lead to larger probabilities
4. Preserves relative order of inputs

### Key Features
- Converts raw scores to probabilities
- Handles any number of classes
- Each output represents probability for that class
- Commonly used in neural networks
- Differentiable (good for backpropagation)

### Common Applications
- Image classification
- Natural language processing
- Speech recognition
- Document categorization
- Medical diagnosis

### Loss Function
Usually paired with cross-entropy loss:
```
Loss = -Σ(y_i * log(p_i))
```
Where:
- y_i is true label (one-hot encoded)
- p_i is predicted probability

Remember: Softmax is essentially a "soft" version of the maximum function, giving proportional probabilities rather than a single winner.


# One-Hot Encoding

Imagine you're organizing a wardrobe with different types of clothing items. Instead of labeling a piece as "number 1" or "number 2", you create separate yes/no columns for each type - one for shirts, one for pants, one for dresses, etc. That's one-hot encoding - converting categorical data into a format where each category gets its own binary column.

## Technical Details

### Basic Format
For categories [A, B, C]:
```
A → [1, 0, 0]
B → [0, 1, 0]
C → [0, 0, 1]
```

### Example
Original data: "Color" column with [Red, Blue, Green]
```
Red   → [1, 0, 0]
Blue  → [0, 1, 0]
Green → [0, 0, 1]
```

### Properties
- Only one '1' per encoding (hence "one-hot")
- Rest are '0's
- Number of columns equals number of categories
- No ordinal relationship implied
- Sparse representation

### Common Applications
- Machine learning models
- Neural networks
- Text processing
- Categorical feature encoding
- Natural language processing

### Advantages
- No ordinal relationship assumed
- Equal distance between categories
- Works well with most algorithms
- Clear binary representation

### Disadvantages
- High dimensionality for many categories
- Sparse matrices
- Memory intensive
- Can be computationally expensive

Remember: One-hot encoding is essential when working with algorithms that expect numerical input but your data is categorical.


# Maximum Likelihood

Imagine you're trying to figure out if a coin is fair by flipping it many times. Maximum likelihood is like asking: "What probability of getting heads would make my actual results most likely to occur?" It's a method that finds the parameters that make your observed data most probable.

The key idea is that we want to calculate P(all), which is the product of all the independent probabilities of each point. This helps indicate how well the model performs in classifying all the points. To get the best model, we will want to maximize this probability.

## Technical Details

### Basic Formula
Log Likelihood = Σ log(P(x|θ))
Where:
- P(x|θ) is probability of data x given parameters θ
- Σ sums over all observations
- Log is used for computational convenience

### Key Concepts
1. Likelihood Function
   - Measures probability of observed data
   - Function of parameters, not data
   - Usually maximized using calculus

2. Log-Likelihood
   - Converts products to sums
   - Preserves same maximum
   - Computationally more stable

### Common Applications
- Parameter estimation
- Statistical inference
- Machine learning models
- Distribution fitting
- Regression analysis

### Steps
1. Write likelihood function
2. Take log of function
3. Find derivative
4. Set derivative to zero
5. Solve for parameters

### Advantages
- Statistically well-founded
- Often has closed-form solution
- Provides consistent estimates
- Works with many distributions

Remember: Maximum likelihood gives us the parameters that make the observed data most probable, but doesn't guarantee they're the "true" parameters.




# Cross-Entropy

The logarithm function has a very nice identity that says that the logarithm of a product is the sum of the logarithms of the factors:

log(ab) = log(a) + log(b)

So if we take the logarithm of our product, we get a sum of the logarithms of the factors.

We'll actually be taking the natural logarithm, ln, which is base e instead of base 10. In practice, everything works the same as what we showed here with log because everything gets scaled by the same factor. However, using ln is the convention, so we'll use it here as well.


The logarithm of a number between zero and one is always a negative number, so that means all of our probabilities (which are between zero and one) will give us negative results when we take their logarithm. Thus, we will want to take the negative of each of these results (i.e., multiply each one by -1).

In the end, what we calculate is a sum of negative logarithms of our probabilities, like this:

-log(0.6) - log(0.2) - log(0.1) - log(0.7) = 4.8

This is called the cross-entropy. A good model gives a high probability and the negative of a logarithm of a large number is a small number—thus, in the end:

- A high cross-entropy indicates a bad model
- A low cross-entropy indicates a good model

We can think of the negatives of these logarithms as errors at each point. The higher the probability, the lower the error—and the lower the cross-entropy. So now our goal has shifted from maximizing the probability to minimizing the cross-entropy.


Think of cross-entropy as a measure of how surprised your model is when seeing the true answers - like a test score where you get penalized more harshly for being confidently wrong than for being unsure. When you're very confident (90%) but wrong, you get a bigger penalty than when you're unsure (51%) and wrong.

## Technical Details

### Basic Formula
```
H(y,ŷ) = -Σ y_i * log(ŷ_i)
```
Where:
- y_i are true values
- ŷ_i are predicted probabilities
- Σ sums over all classes


Formula for cross-entropy

Cross-entropy = -Σ[y_i ln(p_i) + (1 - y_i)ln(1 - p_i)]

From i=1 to m

Where:
- y_i is the true label (0 or 1)
- p_i is the predicted probability
- ln is the natural logarithm
- m is the number of training examples
- Σ represents the sum over all training examples

This is the binary cross-entropy formula specifically used for binary classification problems.


### Key Properties
1. Always positive
2. Zero only when prediction equals reality
3. Higher when predictions are confidently wrong
4. Lower when predictions match true labels

### Examples
- Perfect prediction (1.0) → CE ≈ 0
- Wrong prediction (0.0) → CE → ∞
- Uncertain prediction (0.5) → Moderate CE

### Applications
- Classification problems
- Neural networks
- Model evaluation
- Information theory
- Deep learning

### Important Points
- Used as loss function
- Measures prediction quality
- Related to maximum likelihood
- Good for gradient descent
- Works with multiple classes

Remember: The goal is to minimize cross-entropy, as lower values indicate better model predictions.


# Multi-Class Cross-Entropy

Imagine a weather prediction system that needs to decide between sunny, rainy, or cloudy - not just a simple yes/no. Multi-class cross-entropy helps measure how well our predictions match reality when we have multiple possible outcomes, penalizing the model more heavily when it's confidently wrong about any of the classes.

## Technical Details

### Basic Formula
```
H(y,p) = -Σ Σ y_ij * log(p_ij)
```
Where:
- y_ij is 1 if sample i belongs to class j, else 0
- p_ij is predicted probability that sample i belongs to class j
- First Σ sums over all samples
- Second Σ sums over all classes

### Properties
1. Extends binary cross-entropy to multiple classes
2. Works with one-hot encoded labels
3. Each class contributes to total loss
4. Always non-negative
5. Zero only for perfect predictions

### Common Applications
- Deep learning classification
- Natural language processing
- Image recognition
- Speech recognition
- Document classification

### Important Considerations
- Requires normalized probabilities (sum to 1)
- Often paired with softmax activation
- More computationally intensive than binary
- Handles class imbalance naturally
- Differentiable (good for gradient descent)

Remember: The goal remains minimizing the cross-entropy, which happens when predicted probabilities match true distributions across all classes.


# Logistic Regression

Think of logistic regression like a sophisticated yes/no decision maker - similar to how a doctor might determine if a patient has a condition based on various symptoms. Despite its name, it's actually used for classification, not regression, and predicts the probability of an outcome being in a particular category.

## Technical Details

### Basic Formula
```
P(y=1) = 1 / (1 + e^(-z))
where z = wx + b
```
Where:
- w = weights
- x = input features
- b = bias term
- e = Euler's number

### Key Components
1. Sigmoid Function
   - Transforms linear input to [0,1] range
   - Creates S-shaped curve
   - Output interpreted as probability

2. Cost Function (Binary Cross-Entropy)
```
J(w,b) = -(1/m)Σ[y_i log(ŷ_i) + (1-y_i)log(1-ŷ_i)]
```

### Formula for the error function (for binary classification problems)

Error function = -(1/m)Σ[(1 - y_i)(ln(1 - ŷ_i)) + y_i ln(ŷ_i)]
From i=1 to m

And the total formula for the error is then:

E(W,b) = -(1/m)Σ[(1 - y_i)(ln(1 - σ(Wx^(i) + b))) + y_i ln(σ(Wx^(i) + b))]
From i=1 to m

For multiclass problems, the error function is:

Error function = -(1/m)Σ Σ y_ij ln(ŷ_ij)
From i=1 to m, j=1 to n

Now that we know how to calculate the error, our goal will be to minimize it.

Where:
- m is number of samples
- n is number of classes
- y_i are true values
- ŷ_i are predicted values
- W is weight matrix
- b is bias term 
- σ is sigmoid function
- x^(i) is input vector for sample i


### Properties
- Binary classification (usually)
- Outputs probabilities
- Requires numeric input features
- Assumes linear decision boundary
- Easy to interpret coefficients

### Common Applications
- Medical diagnosis
- Credit risk assessment
- Email spam detection
- Customer churn prediction
- Marketing response prediction

### Advantages
- Simple to implement
- Fast to train
- Probabilistic interpretation
- Works well with linear boundaries
- Less prone to overfitting


# Gradient Descent

Imagine rolling a ball down a hill - it naturally finds the lowest point by following the steepest path downward. That's essentially what gradient descent does in machine learning: it finds the minimum of a function by repeatedly taking steps in the direction where the function decreases most quickly (the steepest descent).

## The Calculation Process

### Step 1: The Error Function
The error function E tells us how wrong our predictions are:
```
E = -(1/m)Σ(y_i ln(ŷ_i) + (1 - y_i)ln(1 - ŷ_i))
```

### Step 2: Finding the Direction (Gradient)
We calculate partial derivatives to find which direction leads downhill:
- For weights: ∂E/∂w_j = -(y - ŷ)x_j
- For bias: ∂E/∂b = -(y - ŷ)

### Step 3: Update Rule
We update parameters by moving in the opposite direction of the gradient:
```
w'_i ← w_i + α(y - ŷ)x_i
b' ← b + α(y - ŷ)
```
Where:
- α is the learning rate (step size)
- (y - ŷ) is the prediction error
- x_i are the input features

### Significance
1. Larger errors cause bigger steps
2. Direction depends on whether prediction was too high or too low
3. Step size is controlled by learning rate α
4. Process repeats until convergence

The beauty of this calculation is its elegant form: the gradient turns out to be just the error (y - ŷ) times the input features, making it computationally efficient and intuitively meaningful.


# Gradient calculation

In the last few videos, we learned that in order to minimize the error function, we need to take some derivatives. So let's get our hands dirty and actually compute the derivative of the error function. The first thing to notice is that the sigmoid function has a really nice derivative. Namely,

σ'(x) = σ(x)(1 - σ(x))

The reason for this is the following, we can calculate it using the quotient formula:

σ'(x) = ∂/∂x 1/(1+e^-x)
       = e^-x/(1+e^-x)²
       = 1/(1+e^-x) · e^-x/(1+e^-x)
       = σ(x)(1 - σ(x))

And now, let's recall that if we have m points labelled x^(1), x^(2),...,x^(m), the error formula is:

E = -(1/m)Σ(y_i ln(ŷ_i) + (1 - y_i)ln(1 - ŷ_i))

where the prediction is given by ŷ_i = σ(Wx^(i) + b).

Our goal is to calculate the gradient of E, at a point x = (x₁,...,x_n), given by the partial derivatives

∇E = (∂/∂w₁E,...,∂/∂w_n E,∂/∂b E)

To simplify our calculations, we'll actually think of the error that each point produces, and calculate the derivative of this error. The total error, then, is the average of the errors at all the points. The error produced by each point is, simply,

E = -y ln(ŷ) - (1 - y)ln(1 - ŷ)

In order to calculate the derivative of this error with respect to the weights, we'll first calculate ∂/∂w_j ŷ. Recall that ŷ = σ(Wx + b), so:

∂/∂w_j ŷ = ∂/∂w_j σ(Wx + b)
          = σ(Wx + b)(1 - σ(Wx + b)) · ∂/∂w_j(Wx + b)
          = ŷ(1 - ŷ) · ∂/∂w_j(Wx + b)
          = ŷ(1 - ŷ) · ∂/∂w_j(w₁x₁ + ... + w_jx_j + ... + w_nx_n + b)
          = ŷ(1 - ŷ) · x_j

The last equality is because the only term in the sum which is not a constant with respect to w_j is precisely w_jx_j, which clearly has derivative x_j.

Now, we can go ahead and calculate the derivative of the error E at a point x, with respect to the weight w_j.

∂/∂w_j E = ∂/∂w_j[-y log(ŷ) - (1 - y)log(1 - ŷ)]
          = -y ∂/∂w_j log(ŷ) - (1 - y)∂/∂w_j log(1 - ŷ)
          = -y · 1/ŷ · ∂/∂w_j ŷ - (1 - y) · 1/(1-ŷ) · ∂/∂w_j(1 - ŷ)
          = -y · 1/ŷ · ŷ(1 - ŷ)x_j - (1 - y) · 1/(1-ŷ) · (-1)ŷ(1 - ŷ)x_j
          = -y(1 - ŷ) · x_j + (1 - y)ŷ · x_j
          = -(y - ŷ)x_j

A similar calculation will show us that

∂/∂b E = -(y - ŷ)

This actually tells us something very important. For a point with coordinates (x₁,...,x_n), label y, and prediction ŷ, the gradient of the error function at that point is (-(y - ŷ)x₁,...,-(y - ŷ)x_n,-(y - ŷ)). In summary, the gradient is

∇E = -(y - ŷ)(x₁,...,x_n,1).

If you think about it, this is fascinating. The gradient is actually a scalar times the coordinates of the point! And what is the scalar? Nothing less than a multiple of the difference between the label and the prediction. What significance does this have?

So, a small gradient means we'll change our coordinates by a little bit, and a large gradient means we'll change our coordinates by a lot.

If this sounds anything like the perceptron algorithm, this is no coincidence! We'll see it in a bit.

# Gradient descent step

Therefore, since the gradient descent step simply consists in subtracting a multiple of the gradient of the error function at every point, then this updates the weights in the following way:

w'_i ← w_i - α[-(y - ŷ)x_i],

which is equivalent to

w'_i ← w_i + α(y - ŷ)x_i.

Similarly, it updates the bias in the following way:

b' ← b + α(y - ŷ),

Note: Since we've taken the average of the errors, the term we are adding should be 1/m · α instead of α, but as α is a constant, then in order to simplify calculations, we'll just take 1/m · α to be our learning rate, and abuse the notation by just calling it α.


# Logistic Regression Algorithm

Think of logistic regression algorithm like teaching a computer to make yes/no decisions by gradually adjusting its decision-making weights until it gets better at predicting - similar to how you might adjust the temperature knob on an oven until you get it just right.

## Algorithm Steps

Here are our steps for logistic regression:

1. Start with random weights: w₁,...,w_n, b

2. For every point (x₁,...,x_n):
   - For i = 1...n:
     - Update w'_i ← w_i - a(ŷ - y)x_i
     - Update b' ← b - a(ŷ - y)

3. Repeat until the error is small

## Key Components

### Initial Setup
- Begin with random weights and bias
- These are the starting parameters that will be refined

### Update Process
- For each data point:
  - Calculate predicted value (ŷ)
  - Compare with actual value (y)
  - Adjust weights and bias accordingly

### Convergence
- Continue updating until error becomes acceptably small
- Error measures the difference between predictions and actual values

### Learning Rate (a)
- Controls how big steps we take in updating
- Too large: might overshoot
- Too small: slow learning



Here are the key points of contrast between gradient descent and the perceptron algorithm that Luis mentioned in the video:

# Gradient Descent

With gradient descent, we change the weights from w_i to w_i + a(y - ŷ)x_i.

# Perceptron Algorithm

With the perceptron algorithm we only change the weights on the misclassified points. If a point x is misclassified:

- We change w_i:
  - To w_i + ax_i if positive
  - To w_i - ax_i if negative

- If correctly classified: y - ŷ = 0

- If misclassified:
  - y - ŷ = 1 if positive
  - y - ŷ = -1 if negative


# PERCEPTRON ALGORITHM:

If x is misclassified:

Change w_i to {
    w_i + α x_i if positive
    w_i - α x_i if negative
}

If correctly classified: y-ŷ=0

If misclassified: {
    y-ŷ = 1 if positive
    y-ŷ = -1 if negative
}


### Neural Network Architecture

We will combine two linear models to get our non-linear model. Essentially the steps to do this are:

Calculate the probability for each model
Apply weights to the probabilities
Add the weighted probabilities
Apply the sigmoid function to the result

Multiple layers
Now, not all neural networks look like the one above. They can be way more complicated! In particular, we can do the following things:

Add more nodes to the input, hidden, and output layers.
Add more layers.
We'll see the effects of these changes in the next video.

Neural networks have a certain special architecture with layers:

The first layer is called the input layer, which contains the inputs.
The next layer is called the hidden layer, which is the set of linear models created with the input layer.
The final layer is called the output layer, which is where the linear models get combined to obtain a nonlinear model.
Neural networks can have different architectures, with varying numbers of nodes and layers:

Input nodes. In general, if we have n nodes in the input layer, then we are modeling data in n-dimensional space (e.g., 3 nodes in the input layer means we are modeling data in 3-dimensional space).
Output nodes. If there are more nodes in the output layer, this simply means we have more outputs—for example, we may have a multiclass classification model.
Layers. If there are more layers then we have a deep neural network. Our linear models combine to create nonlinear models, which then combine to create even more nonlinear models!

#### Multi-Class Classification

And here we elaborate a bit more into what can be done if our neural network needs to model data with more than one output.

When we have three or more classes, we could construct three separate neural networks—one for predicting each class. However, this is not necessary. Instead, we can add more nodes in the output layer. Each of these nodes will give us the probability that the item belongs to the given class.


# Feedforward Neural Networks

Imagine a production line where raw materials (input) move through multiple processing stations, each station transforming the material a bit, until you get the final product (output). That's how feedforward works in neural networks - information flows forward through layers, each layer transforming the data in specific ways.

## Technical Details

### Process Steps
1. Take the input vector
2. Apply a sequence of linear models and sigmoid functions
3. Combine maps to create a highly non-linear map

### Mathematical Formula
ŷ = σ ∘ W^(2) ∘ σ ∘ W^(1)(x)

Where:
- σ is the sigmoid function
- W^(1) is first layer weights
- W^(2) is second layer weights
- ∘ represents function composition

### Key Features
- One-way flow (forward only)
- Layer-by-layer processing
- Non-linear transformations
- Sequential computation
- No cycles or loops

### Applications
- Pattern recognition
- Classification tasks
- Regression problems
- Feature learning
- Function approximation

Remember: The power of feedforward networks comes from their ability to transform input through multiple layers, creating increasingly complex representations of the data.

# Backpropagation

Think of backpropagation like tracing back your steps after making a mistake to figure out exactly where things went wrong. In neural networks, it's how we calculate which weights need adjusting and by how much, by working backwards from the output error to determine each layer's contribution to that error.


Now, we're ready to get our hands into training a neural network. For this, we'll use the method known as backpropagation. In a nutshell, backpropagation will consist of:

Doing a feedforward operation.
Comparing the output of the model with the desired output.
Calculating the error.
Running the feedforward operation backwards (backpropagation) to spread the error to each of the weights.
Use this to update the weights, and get a better model.
Continue this until we have a model that is good.
Sounds more complicated than what it actually is. Let's take a look in the next few videos. The first video will show us a conceptual interpretation of what backpropagation is.


# Calculation of the derivative of the sigmoid function

Recall that the sigmoid function has a beautiful derivative, which we can see in the following calculation. This will make our backpropagation step much cleaner.

σ'(x) = ∂/∂x 1/(1+e^-x)
      = e^-x/(1+e^-x)²
      = 1/(1+e^-x) · e^-x/(1+e^-x)
      = σ(x)(1 - σ(x))

## Technical Details

### Basic Process
1. Forward Pass:
   - Input goes through network
   - Calculate predicted output
   - Measure error

2. Backward Pass:
   - Start from output error
   - Calculate gradients layer by layer
   - Propagate error backwards

### Key Components
```
Error gradient = Local gradient × Upstream gradient
```

### Chain Rule Application
- Output layer: δ = (y - ŷ) × σ'(z)
- Hidden layers: δ = (W^T δ_next) × σ'(z)
where:
- δ is error term
- σ' is derivative of activation function
- W^T is transposed weight matrix

### Update Rules
- Weights: W_new = W_old - α × (input × δ)
- Biases: b_new = b_old - α × δ

### Key Features
- Efficient gradient computation
- Layer-wise updates
- Uses chain rule of calculus
- Enables deep learning
- Computationally efficient

Remember: Backpropagation is what makes deep learning possible by efficiently computing how each weight contributes to the overall error.






# Implementing Gradient Descent


# Mean Squared Error (MSE)

Imagine measuring how far off your darts are from the bullseye by measuring the distance of each throw, squaring those distances (to make all errors positive), and then taking the average. That's essentially what MSE does - it measures the average squared difference between predictions and actual values.

## Technical Definition
```
MSE = (1/n) Σ(y_i - ŷ_i)²
```
Where:
- n is number of samples
- y_i is actual value
- ŷ_i is predicted value
- Σ sums over all samples

## Key Properties
1. Always non-negative (due to squaring)
2. Perfect predictions yield MSE = 0
3. Larger errors are penalized more heavily
4. Unit of measurement is squared
5. Sensitive to outliers

## Advantages
- Simple to compute
- Easy to differentiate
- Clear interpretation
- Good for regression problems
- Penalizes large errors more than small ones

## Disadvantages
- Scale-dependent
- Can be dominated by outliers
- Squared units make interpretation less intuitive
- May not be ideal for classification tasks

Remember: MSE is particularly useful when larger errors are more problematic than smaller ones, as the squaring effect emphasizes bigger differences.


# Gradient Descent with Squared Errors

We want to find the weights for our neural networks. Let's start by thinking about the goal. The network needs to make predictions as close as possible to the real values. To measure this, we use a metric of how wrong the predictions are, the error. A common metric is the sum of the squared errors (SSE):

E = 1/2 ΣΣ[y_j^μ - ŷ_j^μ]²
        μ  j

where ŷ is the prediction and y is the true value, and you take the sum over all output units j and another sum over all data points μ. This might seem like a really complicated equation at first, but it's fairly simple once you understand the symbols and can say what's going on in words.

First, the inside sum over j. This variable j represents the output units of the network. So this inside sum is saying for each output unit, find the difference between the true value y and the predicted value from the network ŷ, then square the difference, then sum up all those squares.

Then the other sum over μ is a sum over all the data points. So, for each data point you calculate the inner sum of the squared differences for each output unit. Then you sum up those squared differences for each data point. That gives you the overall error for all the output predictions for all the data points.

The SSE is a good choice for a few reasons. The square ensures the error is always positive and larger errors are penalized more than smaller errors. Also, it makes the math nice, always a plus.

Remember that the output of a neural network, the prediction, depends on the weights

ŷ_j^μ = f(Σ w_ij x_i^μ)
           i

and accordingly the error depends on the weights

E = 1/2 ΣΣ[y_j^μ - f(Σ w_ij x_i^μ)]²
        μ  j            i

We want the network's prediction error to be as small as possible and the weights are the knobs we can use to make that happen. Our goal is to find weights w_ij that minimize the squared error E. To do this with a neural network, typically you'd use gradient descent.


As Luis said, with gradient descent, we take multiple small steps towards our goal. In this case, we want to change the weights in steps that reduce the error. Continuing the analogy, the error is our mountain and we want to get to the bottom. Since the fastest way down a mountain is in the steepest direction, the steps taken should be in the direction that minimizes the error the most. We can find this direction by calculating the gradient of the squared error.

Gradient is another term for rate of change or slope. If you need to brush up on this concept, check out Khan Academy's [great lectures] on the topic.

To calculate a rate of change, we turn to calculus, specifically derivatives. A derivative of a function f(x) gives you another function f'(x) that returns the slope of f(x) at point x. For example, consider f(x) = x². The derivative of x² is f'(x) = 2x. So, at x = 2, the slope is f'(2) = 4. Plotting this out, it looks like:

[Note: The text contains a link that I cannot process, but I've included it in brackets to maintain the original format]

The gradient is just a derivative generalized to functions with more than one variable. We can use calculus to find the gradient at any point in our error function, which depends on the input weights. You'll see how the gradient descent step is derived on the next page.

Below I've plotted an example of the error of a neural network with two inputs, and accordingly, two weights. You can read this like a topographical map where points on a contour line have the same error and darker contour lines correspond to larger errors.

At each step, you calculate the error and the gradient, then use those to determine how much to change each weight. Repeating this process will eventually find weights that are close to the minimum of the error function, the black dot in the middle.


Caveats
Since the weights will just go wherever the gradient takes them, they can end up where the error is low, but not the lowest. These spots are called local minima. If the weights are initialized with the wrong values, gradient descent could lead the weights into a local minimum, illustrated below.

If we want our neural network to make reasonable predictions, we need to have a way of setting the weights. To address this, we can present the neural network with data that we know to be true and then set the model parameters (the weights) to match that data.

The most essential component we need for this is some measure of how bad our predictions are. The measure we'll use is the sum of the squared errors (SSE), which looks like this:

E = 1/2 Σ(y^μ - ŷ^μ)²
       μ

The SSE is a measure of our network's performance. If it's high, the network is making bad predictions. If it's low, the network is making good predictions. Minimizing the SSE is our goal in gradient descent:

- Starting at some random weight, we make a step in the direction towards the minimum, opposite to the gradient or slope.

- If we take many steps, always descending down a gradient, eventually the weights will find the minimum of the error function.


# Mean Square Error

We're going to make a small change to how we calculate the error here. Instead of the SSE, we're going to use the mean of the square errors (MSE). Now that we're using a lot of data, summing up all the weight steps can lead to really large updates that make the gradient descent diverge. To compensate for this, you'd need to use a quite small learning rate. Instead, we can just divide by the number of records in our data, m to take the average. This way, no matter how much data we use, our learning rates will typically be in the range of 0.01 to 0.001. Then, we can use the MSE (shown below) to calculate the gradient and the result is the same as before, just averaged instead of summed.

E = 1/2m Σ(y^μ - ŷ^μ)²
          μ

Here's the general algorithm for updating the weights with gradient descent:

- Set the weight step to zero: Δw_i = 0

- For each record in the training data:
  - Make a forward pass through the network, calculating the output ŷ = f(Σ_i w_ix_i)
  - Calculate the error term for the output unit, δ = (y - ŷ) * f'(Σ_i w_ix_i)
  - Update the weight step Δw_i = Δw_i + δx_i

- Update the weights w_i = w_i + ηΔw_i/m where η is the learning rate and m is the number of records. Here we're averaging the weight steps to help reduce any large variations in the training data.

- Repeat for e epochs.

You can also update the weights on each record instead of averaging the weight steps after going through all the records.

Remember that we're using the sigmoid for the activation function, f(h) = 1/(1 + e^-h)

And the gradient of the sigmoid is f'(h) = f(h)(1 - f(h))

where h is the input to the output unit,

h = Σ_i w_ix_i


