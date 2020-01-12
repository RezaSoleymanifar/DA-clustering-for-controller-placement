This is the Python implementation of the 
"A Clustering Approach to Edge Controller Placement in Software Defined Networks with Cost Balancing" paper.
You can find the paper [here](https://arxiv.org/abs/1912.02915?context=cs).

This project is done from scratch and without use of any ML packages. Feel free to load your own datasets to train your own models. Two training methods are provided: (1) training using barrier interior point method with Newton steps, for optimization of quadratic programs and (2) ellipsoid method. Information on the details of implementations of these algorithms is provided in `Training Documentation.ipynb` notebook.

To clone the project use:
```bash
git clone https://github.com/RezaSoleymanifar/DA-SDN.git
```

To install required packages run:

```bash
pip install -r requirements.txt
```

- Find the placement of controllers:

```bash
python train_ipm.py 'data_file.npy' max_m gamma interactive mode
```

`'data_file.npy` contains the coordinates of the network nodes as rows of the input data matrix. `max_m' is the hyperparameter
that restricts the maximum number of controllers to be placed in the network.
`gamma' denotes the synchronization cost coefficient, `interactive' can take any
vlaue in `1, 2, 3' and each corresponds to a dynamic plot that displays the progress of the algorithm.
`mode' can be either `'ll'' or `'lb'' standing for leader-less and leader-based topologies respectively.

The results are saved to `results.npy' file as a dictionary containing the clusters and their corresponding controllers.

### Reference

[A Clustering Approach to Edge Controller Placement in Software Defined Networks with Cost Balancing]
(https://arxiv.org/abs/1912.02915?context=cs)
