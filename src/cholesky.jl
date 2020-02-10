export cholesky, cholesky_back!

# For real matrix, complex matrix is under development ......
function cholesky(A)
	C = LinearAlgebra.cholesky(A)
	Matrix(C.L)
end

function level2partition(X, i)
    N = size(X, 1)
    r = view(X, i, 1:i-1)
    c = view(X, i+1:N, i)
    d = X[i,i]
    B = view(X, i+1:N, 1:i-1)
    r, d, B, c
end

function cholesky_back!(L̄::AbstractMatrix{T}, L::AbstractMatrix{T}) where {T}
    N = size(L, 1)
    
    # N-th element
    rN, dN = view(L, N, 1:N-1), L[N,N]
    r̄N, d̄N = view(L̄, N, 1:N-1), L̄[N,N]
    d̄N = d̄N / dN
    LinearAlgebra.BLAS.axpy!(-d̄N, rN, r̄N)
    d̄N = d̄N / T(2.0)
    L̄[N,N] = d̄N
    
    # N-1 to 2
    @inbounds for i in N-1:-1:2
        r, d, B, c = level2partition(L, i)
        r̄, d̄, B̄, c̄ = level2partition(L̄, i)
        d̄ -= (T(1.0)/d)*LinearAlgebra.BLAS.dot(N-i, c, c.stride1, c̄, c̄.stride1)
        d̄ = d̄ / d; LinearAlgebra.BLAS.scal!(N-i, T(1.0)/d, c̄, c̄.stride1)
        r̄ .= r̄ - (d̄*r + LinearAlgebra.BLAS.gemv('T', B, c̄))
        LinearAlgebra.BLAS.ger!(T(-1.0), c̄, r, B̄)
        d̄ = d̄ / T(2.0)
        L̄[i,i] = d̄
    end
    
    # 1st element
    d1, c1 = L[1,1], view(L, 2:N, 1)
    d̄1, c̄1 = L̄[1,1], view(L̄, 2:N, 1)
    d̄1 -= (T(1.0)/d1)*LinearAlgebra.BLAS.dot(N-1, c1, 1, c̄1, 1)
    d̄1 = d̄1/d1; LinearAlgebra.BLAS.scal!(N-1, T(1.0)/d1, c̄1, c̄1.stride1)
    d̄1 = d̄1/T(2)
    L̄[1,1] = d̄1
    
    L̄
end
