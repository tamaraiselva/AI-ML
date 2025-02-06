import numpy as np

# Step 1: Create two small matrices
matrix_A = np.array([[1, 2], 
                      [3, 4]])

matrix_B = np.array([[5, 6], 
                      [7, 8]])

print("Matrix A:")
print(matrix_A)

print("\nMatrix B:")
print(matrix_B)

# Step 2: Perform basic operations

# Matrix addition
matrix_sum = matrix_A + matrix_B
print("\nSum of Matrix A and B:")
print(matrix_sum)

# Matrix subtraction
matrix_diff = matrix_A - matrix_B
print("\nDifference of Matrix A and B:")
print(matrix_diff)

# Element-wise multiplication
matrix_product = matrix_A * matrix_B
print("\nElement-wise Product of Matrix A and B:")
print(matrix_product)

# Matrix multiplication
matrix_mul = np.dot(matrix_A, matrix_B)
print("\nMatrix A multiplied by Matrix B:")
print(matrix_mul)

# Step 3: Transpose of a matrix
matrix_A_T = matrix_A.T
print("\nTranspose of Matrix A:")
print(matrix_A_T)

# Step 4: Determinant of a square matrix
determinant_A = np.linalg.det(matrix_A)
print("\nDeterminant of Matrix A:", determinant_A)

# Step 5: Inverse of a matrix
matrix_A_inv = np.linalg.inv(matrix_A)
print("\nInverse of Matrix A:")
print(matrix_A_inv)

# Verification: Matrix A multiplied by its inverse
identity_matrix = np.dot(matrix_A, matrix_A_inv)
print("\nMatrix A multiplied by its Inverse (should be Identity Matrix):")
print(identity_matrix)