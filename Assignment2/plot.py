import matplotlib.pyplot as plt
import numpy as np

def plot_loss_vs_epoch(filename, epochs):


  # Read data from the text file
  data = np.loadtxt(filename)
  epochs = list(data[0 : epochs])
#   losses = list(data[2001 : 3001])
#   vallos = list(data[4000 : 5001])
  losses = list(data[70 : 105])
  vallos = list(data[140 : 175])
  print(len(epochs))
  print(len(losses))
  print(len(vallos))
#   print(epochs) # Extract losses from the second column

  # Plot the data
  plt.plot(epochs, losses, label="Training Loss")
  plt.plot(epochs, vallos, label="Validation Loss")
  plt.xlabel("Epoch")
  plt.ylabel("Loss")
  plt.title("Training/Validation Loss vs Epoch")
  plt.legend()
  plt.grid(True)

  # Show the plot
  plt.show()


filename = "./epochs.txt"
plot_loss_vs_epoch(filename, 35)