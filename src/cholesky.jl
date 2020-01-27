export cholesky, cholesky_back

# For real matrix, complex matrix is under development ......
function cholesky(A)
	C = LinearAlgebra.cholesky(A)
	Matrix(C.L)
end

function cholesky_back(A, L, dL)
	N = size(A, 1)
	dA = zero(A)
	for i in N:-1:1
		for j in i:-1:1
			if j==i
				dA[i,i] = 0.5*dL[i,i]/L[i,i]
			else
				dA[i,j] = dL[i,j]/L[j,j]
				dL[j,j] = dL[j,j] - dL[i,j]*L[i,j]/L[j,j]
			end
			for k in j-1:-1:1
				dL[i,k] = dL[i,k] - dA[i,j]*L[j,k]
				dL[j,k] = dL[j,k] - dA[i,j]*L[i,k]
			end
		end
	end
	dA
end

