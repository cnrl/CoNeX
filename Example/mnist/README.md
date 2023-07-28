- install the latest version of `pymonntorch` from github. [user guide](https://pymonntorch.readthedocs.io/en/latest/installation.html#from-sources)
- install the latest version of `CoNeX` from github.
- Load your own clasfication image dataset with pytorch documentation. e.g. Mnist, eth-80
  - For faster benchmark, you can set `subset_selected_labels` to subset of target classes
- update `config` and `NUM_ITERATION` based on your need.

- Run the simulation
- the ouput recorded for the timing window in the all simulation is recorded in the `net.metric_.output`
  - NOTE: The ouputs are single item tensors use `tensor.item()` to get the values
  - NOTE: the ouput shapes is `[(predicted_label_0, actual_label_0) ... (predicted_label_m, actual_label_m)]`
  - NOTE: `predicted_label_i` is index of neuron in the ouput which has been spiked the most in the timing window
  - NOTE: `actual_label_i` is the actual dataset.target for has been showed to network for in the timing windonw

if you want to gain more performance use cuda for device and use `dtype=torch.float16`
change it back to `torch.float32` if you face the following error

```bash
RuntimeError: "slow_conv2d_cpu" not implemented for 'Half'
```
