import numpy as np

def roundupA(A, n):
    for i in range(n):
        for j in range(n):
            A[i, j] = round(A[i, j], 1)
    return A

def roundupB(B, n):
    for i in range(n):
        for j in range(n):
            B[i, j] = round(B[i, j], 1)
    return B


n = int(input("Enter the size of the square matrix you want : "))
X = np.random.randint(10, size=n)
A = np.random.randint(10, size=(n, n))
B = np.random.randint(10, size=(n, n))

# Displaying A and B transpose
print("Transpose of A : \n", A.transpose())
print("Transpose of B : \n", B.transpose())

# Displaying A*B
print("The matrix multiplication of A and B : \n", np.matmul(A, B))
AB = np.matmul(A, B)

# Verify (AB)t = BtAt
print("AB transpose:\n ", (np.matmul(A, B)).transpose())
print("B transpose * A transpose : \n", np.matmul(B.transpose(), A.transpose()))

# Calculate Determinant of A and B
detA = np.linalg.det(A)
detB = np.linalg.det(B)
print("Determinant of A : " + str(round(np.linalg.det(A))) + " and Determinant of B : " + str(round(np.linalg.det(B))))

# To verify singularity of MatrixA and MatrixB
if detA == 0 and detB == 0:
    print("The matrix A and B are singular matrix ")
else:
    print("The matrix A and B are not singular matrix")

if detA == 0 and detB == 0:
    print("The inverse of the matrix A and matrix B is not Possible")
else:
    print("The inverse of matrix A is : \n" + str(
        roundupA(np.linalg.inv(A), n)) + " \nThe inverse of matrix B is :\n " + str(roundupB(np.linalg.inv(B), n)))

inverseA = np.linalg.inv(A)
inverseB = np.linalg.inv(B)

# Verify A*transpose(A) == transpose(A)*A
print("A*inverse(A) =  \n " + str(roundupA(np.matmul(A, inverseA), n)))
print("inverse(A)*A =  \n " + str(roundupB(np.matmul(inverseA, A), n)))

# Trace of A and B,inverse(A) and inverse(B), transpose(A) and transpose(B)
print("Trace of A is : " + str(A.trace()))
print("Trace of B is : " + str(B.trace()))
print("Trace of inverse of A is : " + str(roundupA(inverseA, n).trace()))
print("Trace of inverse of B is : " + str(roundupB(inverseB, n).trace()))
print("Trace of transpose of A is : " + str(A.transpose().trace()))
print("Trace of transpose of B is : " + str(B.transpose().trace()))

# Verify trace of A*B and Trace of B*A
print("Trace of A*B is : " + str(AB.trace()))
print("Trace of B*A is : " + str(AB.trace()))

# Verify trace(transpose(A)*B) = trace(A*transpose(B)) = trace(B*transpose(A)) = trace(transpose(B)*A)
print("Trace of transpose(A)*B : ", np.matmul(A.transpose(), B).trace())
print("Trace of A*transpose(B): ", np.matmul(B.transpose(), A).trace())
print("Trace of B*transpose(A) : ", np.matmul(A.transpose(), B).trace())
print("Trace of transpose(B)*A : ", np.matmul(B.transpose(), A).trace())

# y = Ax

m =  int(input("Enter rows of matrix: "))
n = int(input("Enter columns of matrix: "))

A2 = np.random.randint(10, size=(m, n))
X2 = np.random.randint(10, size=n)
Y = np.matmul(A2, X2)

print("The obtained x2 from y is : \n" + str(X2))

# Inner product of X2 and Y. Also check their orthogonality
print("Inner product of X2 and Y is : " + str(np.matmul(X2.transpose(), Y)))
if np.matmul(X2, Y) == 0:
    print("The matrix X2 and Y are orthogonal matrices")
else:
    print("The matrix X2 and Y are not orthogonal matrices")

# Normalize X2 and Y
print("Normalizing X2 and Y and their values are: " + str(np.matmul(X2, X2.transpose())) + " , " + str(
    np.matmul(Y, Y.transpose())))

# Verify Cauchy-Swartz theorem
print("|X2.Y| : " + str(np.matmul(X2.transpose(), Y)))
print("||X2||.||Y|| : " + str(np.matmul(X2, X2.transpose()) * np.matmul(Y, Y.transpose())))

# Verify tranpose(x)*y = transpose(y)*x
print("X2 Transpose * Y : " + str(np.matmul(X2.transpose(), Y)))
print("Y Transpose * X2 : " + str(np.matmul(Y.transpose(), X2)))

# Verify tranpose(y)*A*X2 = transpose(x)*tranpose(A2)*Y
print("Tranpose(y)*A*X2 : " + str(np.matmul(Y.transpose(), Y)))
print("Transpose(x)*Tranpose(A2)*Y : " + str(np.matmul(Y, Y.transpose())))

# Eigen Values and Eigen Vectors of A and B
eigenvalA, eigenvecA = np.linalg.eig(A)
eigenvalB, eigenvecB = np.linalg.eig(B)
print("Eigen value of A : " + str(eigenvalA))
print("Eigen vector of A : " + str(eigenvecA))
print("Eigen value of B : " + str(eigenvalB))
print("Eigen vector of B : " + str(eigenvecB))
