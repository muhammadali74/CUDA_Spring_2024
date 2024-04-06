import matplotlib.pyplot as plt

# Data
samples = [4, 400, 1000, 2000, 5000]
gpu = [5.473, 12.42, 30.66, 61.15, 152.6]
cpu = [5.012, 532, 922.224, 3369.33, 6595.34]

# Plot
plt.figure(figsize=(10, 6))
plt.plot(samples, gpu, marker='o', label='GPU')
plt.plot(samples, cpu, marker='s', label='CPU')
plt.xlabel('Number of Samples')
plt.ylabel('Time (s)')
plt.title('Time Taken by CPU vs GPU for Path Tracing')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()