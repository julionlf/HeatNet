# HeatNet: Neural Networks to do Heat Transfer

## Introduction

In this project I wanted to train a Shallow Network to predict the temperature profile of a square surface (you can imagine a table) that's heated at the edges. Below is a simple example where most of the heat in the surface is coming from the bottom most edge.

![image](https://user-images.githubusercontent.com/24802860/210691139-5b447c57-7b8c-42a7-ad9c-d9a18eb9fae7.png)

## The Physics

Engineers make simulations like the one shown above by numerically solving the 2D Diffusion Equation over any given domain, which in this case is a square surface. For those who aren't familiar with the latter, the 2D Diffusion Equation is a Parabolic Partial Differential Equation (PDE), and in most cases it's numerically solved using either the Finite Volume Method (FVM) or the Finite Element Method (FEM). For a steady-state case (results do not vary with time) and no heat generation (there is no internal heat source in the domain) the 2D Diffusion Equation is defined as shown below:

$$ \frac{\partial^2 T}{\partial x^2} + \frac{\partial^2 T}{\partial y^2} = 0 $$

There are analytical solutions for the 2D Diffusion equation in simple cases such as the square domain of interest in this project, but analytically solving this PDE quickly becomes unfeasible with just slight increases in complexity. Instead I implemented a numerical PDE solver in MATLAB using the FVM approach. Both the FEM and FVM approaches are based on the same philosophy: take the original continuous domain (i.e.: the square surface) and break it up into small cells where the complicated PDE can be approximated by an algebraic equation in each one.

![image](https://user-images.githubusercontent.com/24802860/210691203-9c39111e-0c96-4746-8ddb-e00e94930917.png)

Wis the domain discretization shown above, we can convert the original PDE:

$$ \frac{\partial^2 T}{\partial x^2} + \frac{\partial^2 T}{\partial y^2} = 0 $$

to a new discretized form:

$$ T_{i+1,j} + T_{i-1,j} + T_{i,j+1} + T_{i,j-1} - 4T_{i,j} = 0 $$

The i and j subscripts refer to the temperature in a particular X and Y location of the mesh, i'e.: a particular cell. How this mathematical magic happens is beyond the scope of this post, but those interested can check-out how I made the FVM implementation here. What's important now is that we went from having a complicated PDE for the entire continuous domain to a system of linear algebraic equations where each equation predicts the temperature of each cell in the mesh.

There are two important features of this numerical approach. Firstly, the method solves for the temperature at the dot in the each cell and then assumes that temperature for the rest of the cell. Secondly, and particularly relevant for the selection of the Machine Learning model, is the fact that the temperature at each (i,j) location depends on the temperature on the neighboring locations, namely the locations (i+1,j), (i-1,j), (i,j+1) and (i,j-1).

![image](https://user-images.githubusercontent.com/24802860/210691430-6748643b-db29-487c-aafd-a7a5bdaecc46.png)
$$ T_{i+1,j} + T_{i-1,j} + T_{i,j+1} + T_{i,j-1} - 4T_{i,j} = 0 $$

The arrows in the illustration above are intended to provide only an idea of how the neighboring cells interact in the FVM numerical scheme, but in practice heat can also flow in the opposite direction depending on the way the system is defined. If most of the heat were coming form the top most edge in our domain, then most likely the arrows would be in the opposite direction.

To solve the system we need to specify the temperature (or heat-flux) at each edge of the domain. We call the temperatures specified at the edges of the domain "boundary conditions", because they are precisely at the boundaries of our domain. So, these boundary conditions are the inputs of our system, our Xs, and the temperatures we need to solve for at each location are the outputs, the Ys. 

The computation of the temperature is slightly different for the cells next to the edges where the boundary conditions are applied, but again this is beyond the sco

## The Data

The whole point of this project is to make the Shallow Net predict the temperature at each location of our mesh given a set of boundary conditions. In other words, the inputs to our Machine Learning model are the boundary conditions and the outputs the temperature at each location.

![image](https://user-images.githubusercontent.com/24802860/210691523-ac5323ec-e542-4c25-b0ad-82bc0beaddaf.png)

To generate the dataset, I ran the FVM code Nx10 times using a Latin Hypercube Design of Experiments (DOE) where N is the number of cells in the mesh and in each trial the boundary conditions are varied randomly. Accordingly, the structure of the dataset consists of a 4 by Nx10 table of inputs, where each columns corresponds to a boundary condition, and of a N by Nx10 table of outputs, where each columns corresponds to the temperature at each cell. I scaled the data so that all the values would lie between 0 and 1. I didn't allow any negative values since none of the boundary conditions are negative.

![image](https://user-images.githubusercontent.com/24802860/210691552-2b54521f-95f6-411c-938a-059be2ce83fd.png)

## The Machine Learning Model

I decided to build a Shallow Neural network using Tensorflow and Keras as my machine learning model for a few reasons: 

It's relatively simple model to implement. There are other alternatives, such as Regularized Multiple Regression, SVMs, etc., but I would like to start with a Shallow Net in the hope of extending this to Deep Learning.

The structure of the activation functions used to train Networks on tabular data such as the one in this model is analogous to the FVM system of equations in the physical model (or at least I will try to make the argument that it is).

As previously mentioned, the system of algebraic equations in the numerical model has an interconnected structure because heat is conducted from one cell to another, and how much heat flows through each depends on their respective temperatures. In order for the Shallow Net to make successful predictions, it needs to correctly identify this structure of inter-connectivity.

A Shallow Neural Network has three layers: the input layer, the hidden layer, and the output layer. Designing the network architecture is pretty much simplified to deciding how many neurons we want in the hidden layer since the amount of input neurons is given by the amount of boundary conditions and the amount of output neurons is given by the amount of cells.

![image](https://user-images.githubusercontent.com/24802860/210691721-35669328-c60c-4940-b1e6-24c47cd76d90.png)

I decided to go with a guesstimate of Nx4 neurons in the hidden layer based on the notion that the temperature at the neighboring four locations are needed to compute the temperature at the location of interest, so an argument can be made that each output layer neuron might require input from 4 hidden neurons to make an accurate prediction.This a relatively bold assumption given that I haven't rigorously considered what information is being conveyed by the hidden layer neurons. 

![image](https://user-images.githubusercontent.com/24802860/210691766-fffc04cb-2df1-4012-9c8a-58f39ee473c0.png)

For this model, I'm using ReLU activation functions for the hidden layer neurons, and simple Linear activation functions for the output layer neurons (both are pretty standard assumptions for tabular data). 

$$
ReLU: 
\begin{cases} 
x > 0: Y = \Sigma^N_i \beta_i x_i + \beta_0 \\
x \leq 0: Y = 0\\
\end{cases}
$$

$$ Linear: Y = \Sigma^N_i \beta_i x_i + \beta_0 $$

The system of equations in our physical model is almost entirely analogous to the activation function at the output layer: a linear combination of input values weighed by some constant coefficients. In the physical model, the input values are the temperatures at the neighboring nodes and the weights are a product of the thermal conductivity and cell size; in the Shallow Network model, the inputs to the output layer activation functions are the output signals emitted by the hidden layer neurons and its weights are also constant coefficients. So both the activation functions and the FVM equations are linear combinations of variables and constant weights (or coefficients).

![image](https://user-images.githubusercontent.com/24802860/210692198-8ae4eda3-9647-4d47-85d3-85515ab444b7.png)

![image](https://user-images.githubusercontent.com/24802860/210691858-29260ec9-ebaa-43b7-ae20-7452dec8658a.png)

The same assumption can be made with the hidden layer activation functions with the input layer neurons, although the mapping of the signals between these two layers is not as intuitive because each neuron in the hidden layer is receiving much less diverse input, i.e.: 4 boundary condition values. This complexity, however, doesn't prove or disprove the hypothesis that the same analogous computation can happen at the hidden layers. Again, these are all notions at this point, I have yet to look under the hood and study the weights in the neurons to see if there is any correspondence between them and the coefficients on the FVM model, but that's a topic for another post.

I randomly sampled 70% of my dataset for training and used the remaining 30% for testing. I also added a dropout layer with a dropout rate of 10% to control for over-fitting, and a validation loss split of 5%. I played around with this for a bit, but there seemed to be a point where no additional dropout amount would reduce over-fitting. I assumed a Mean Square Error loss function to fit the model using ADAM. Lastly, I trained the model for 50 epochs. I also player around with this, but 50 seemed to be right at the point of diminishing returns (and, sadly, over-fitting).

## The Results

A first glance at the correlation plot between the test predictions and ground-truth results seems to show very good agreement. Frankly, much more than what I was expecting. However, there is pretty severe over-fitting. We can see that in how the data is so tightly packed around the diagonal and in how the validation loss spikes and oscillates near the last few epochs.

![image](https://user-images.githubusercontent.com/24802860/210691912-a878879c-7654-4db9-b778-bf7d18e77ed4.png)

![image](https://user-images.githubusercontent.com/24802860/210691925-e3292048-dd63-412c-adbb-37ff4cda500a.png)

However, keep in mind that our model is making predictions for multiple output values that are all correlated because there is a physical structure that ties them together (heat diffusion). So the model not only needs to make a good "aggregate prediction", it also needs to now "where" each temperature value goes on the mesh. I was pleasantly surprised to see that it did a very respectable job at that in many cases.

![image](https://user-images.githubusercontent.com/24802860/210691959-5940b791-e13f-417c-a108-ea7895f8c823.png)

From a birdseye point of view, we can certainly make the argument that the Shallow Net "gets it". It knows that there is some structure in the data, and that there is structure between the output values. We can see that in how the contour lines generally flow in the same direction. However, upon closer inspection we can see in some cases that the lines aren't smooth.

![image](https://user-images.githubusercontent.com/24802860/210691999-eaf02186-47da-4a15-a1eb-92c33962faa6.png)

This, I believe, also is the result of pretty severe over-fitting. Furthermore, there are some cases where the model randomly predicts a small batch of cooler or warmer temperature. This sort of nonphysical result is to be expected in a "naive" and simple model such as a Shallow Net; it is, after all, a statistical approximation of the physical model (a Surrogate Model). But it almost certainly is also a result of over-fitting.

![image](https://user-images.githubusercontent.com/24802860/210692030-93414e42-5026-467a-9a48-f7edc206149a.png)

## Conclusion

This is a pretty good start for such a simple model. Although it showed that it can capture the interconnected structure of the temperature predictions at each location, it suffered from severe over-fitting. The latter means that this model is incapable of generalizing it's predictions. For example, it might be the case that if we apply a heat flux boundary condition, instead of a temperature boundary condition, the model would output nonsense.

Furthermore, the nonphysical predictions violate the energy conservation principle. Mathematically speaking, the model can make reasonably good predictions to match the ground truth, but it also needs to satisfy the condition that the total amount of heat coming into each cell matches the amount of heat leaving it. Still, I think this is a wonderfully surprising experiment that showed how much such a simple model can learn from physics.

## Further Work

If the Shallow Net really is learning the physics is a legitimate question to ask. One way to know is to look at the weights and see if there is any correspondence with the coefficients in the physical FVM model. Another interesting approach is to directly "teach" it physics.

The immediate task at hand is to handle the over-fitting. Standard approaches can be easily applied to see if they produce any improvements, such as lowering the number of epochs and feeding it less data. Other less traditional approaches can be implemented, such as adding a penalty term to the loss function.

Teaching physics to the network and adding a penalty term to the loss function can be made in a single effort if we add the conservation equation as the penalty term. We can also change the architecture of a Deep Network to make the flow of information resemble more that of the FVM approach; it would be a way of embedding the physics into the model. If you're interested in any of this, check out this other post.
