# Attention zoo
This library contains a collection of some state-of-the-art linear attention mechanisms.

## Usage
In order to understand the proper usage of these implementations, we refer the user to the provided [test.py](test.py) file, where we show a naive example.

In any case, this is a rather general framework, where we define an abstract class [abstract_attention.py](attentions/abstract_attention.py) that applies the standard linear projections. We then define an abstract method *apply_attention* that you implement in each of the concrete attention. Thus, implementing a new attention mechanism of your choice requires only implementing this function.

## TODO:
- The current implementation makes use of OmegaConf configuartion files. If this is not the configuration system that you want to use, I suggest rewriting all the default configuration hyperameters into standard arguments.