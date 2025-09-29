import numpy as np
import matplotlib.pyplot as plt

# Número de épocas
epochs = np.arange(1, 51)

# Error de entrenamiento: disminuye suavemente
train_error = 1 / np.sqrt(epochs) + 0.01 * np.random.randn(len(epochs))

# Error de validación: disminuye al principio, luego aumenta notablemente para evidenciar sobreajuste
val_error = 1 / np.sqrt(epochs) + 0.01 * np.random.randn(len(epochs)) + 0.15 * (epochs/50)**2

# Mejor época según el error de validación mínimo
best_epoch = np.argmin(val_error) + 1

# Crear plot
plt.figure(figsize=(10,6))
plt.plot(epochs, train_error, label='Error de entrenamiento', linewidth=2.5, color='blue')
plt.plot(epochs, val_error, label='Error de validación', linewidth=2.5, color='orange')
plt.axvline(best_epoch, color='red', linestyle='--', label=f'Mejor época = {best_epoch}', linewidth=2)
plt.scatter(best_epoch, val_error[best_epoch-1], color='red', s=100)  # marcar el mínimo
plt.xlabel('Épocas', fontsize=14)
plt.ylabel('Error', fontsize=14)
#plt.title('Dinámica de entrenamiento y sobreajuste', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()
