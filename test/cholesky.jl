using BackwardsLinalg
using Test, Random


@testset "cholesky" begin
    Random.seed!(3)
    T = Float64
		for N in [2, 4, 7]
        @show N
        x = randn(T, N, N)

        op = randn(N, N)
        op += op'

        function tfunc(x)
            A = x' * x
						L = cholesky(A)
            v = L[:,1]
            (v'*op*v)[] |> real
        end
        @test gradient_check(tfunc, x)
    end
end


