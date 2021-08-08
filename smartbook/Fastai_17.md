# A Neural Net from the Foundations

# 基础的神经网络

This chapter begins a journey where we will dig deep into the internals of the models we used in the previous chapters. We will be covering many of the same things we've seen before, but this time around we'll be looking much more closely at the implementation details, and much less closely at the practical issues of how and why things are as they are.

本章节会开始一段路程，在这里我们将使用之前章节提供的内容深入挖掘模型的内部。我们会解释之前我们已经学习过的很多相同内容，但这次的范围我们会更仔细的关注于细节的实现，同时更少贴近于为什么事情是那样的实际问题。

We will build everything from scratch, only using basic indexing into a tensor. We'll write a neural net from the ground up, then implement backpropagation manually, so we know exactly what's happening in PyTorch when we call `loss.backward`. We'll also see how to extend PyTorch with custom *autograd* functions that allow us to specify our own forward and backward computations.

