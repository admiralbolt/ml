# Neural Style Transfer

This is done mostly by following the colab tutorial at:
https://colab.research.google.com/github/tensorflow/models/blob/master/research/nst_blogpost/4_Neural_Style_Transfer_with_Eager_Execution.ipynb#scrollTo=lqTQN1PjulV9

I did a parameter sweep to experiment with different style_layers & weights.
Some summary images are included. It seems like the 4th / 5th layer have a lot
of random-ish noise in them. The best style transfers seem to come from 2nd / 
3rd layers only.
