# Tarea2_ComputacionParalela

Para compilar se utilizó la siguiente linea:

```ruby
$nvcc -o [metodo] reduction_max_[metodo].cu
```
Donde [metodo] es CG (cooperative groups) o Stream. El programa ./[metodo] se lanza de la siguiente manera

```ruby
$./[metodo] N_bits k_bits blockSize
```
Donde

```
N_bits : Log2 de cantidad de listas.
k_bits : Log2 de tamaño de cada lista.
blockSize : Cantidad de threads por bloque.
```
