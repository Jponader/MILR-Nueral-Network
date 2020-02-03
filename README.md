# MILR-Nueral-Network
Creates a custom Python library that implements MILR (Mathematically Induced Layer Recovery) technique, for error recovery. This can be added to tensorflow based Neural Networks for imporved resillancy from memoery corruption, as it adds a way to recover tampered with or corrupted weight in memory.

## Folder Hierarchy

### MILR
  MILR is a framework that can be added the a Tensorflow nueral networks and allows them to recovery thier weight to their previouse state if they have been tampered with.


### Testing
  Holds several deep neural networks used for testing and debugging the development process.
  
### Old_Base
  The starting point for development of a standalone nueral network framework for the purpose of implementing MILR. This was abondend for the purpose of intergrating it into a commonly used ML framework.
